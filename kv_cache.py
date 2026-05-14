# kv_cache.py
import torch
import torch.nn.functional as F


def get_kv_cache_budget(method=None, **kwargs):
    if method is None:
        return None

    if method == "truncate":
        return kwargs.get("max_length", 256)

    if method in {"streaming", "rkv"}:
        return kwargs.get(
            "max_total_len",
            kwargs.get("sink_size", 4) + kwargs.get("window_size", 256)
        )

    if method == "snapkv":
        return kwargs.get("max_total_len", 256)

    if method in {"snapkv_rkv", "snapkv_pp", "snapkvpp_rkv"}:
        return kwargs.get("max_total_len", 512)

    return None


def should_optimize_kv_cache(
    past_key_values,
    method=None,
    step_idx=0,
    last_compress_len=0,
    compress_interval=256,
    min_seq_growth=64,
    **kwargs
):
    if method is None or past_key_values is None:
        return False

    budget = get_kv_cache_budget(method=method, **kwargs)
    if budget is None:
        return False

    seq_len = past_key_values[0][0].size(2)
    return (
        seq_len > budget and
        step_idx % compress_interval == 0 and
        seq_len - last_compress_len > min_seq_growth
    )




def rkv_dedup_kv_cache(
    past_key_values,
    sink_size=4,
    window_size=256,
    proj_dim=64,
    keep_ratio=0.9,
    max_total_len=None
):
    new_past = []

    for k, v in past_key_values:
        B, H, T, D = k.shape
        total_budget = max_total_len if max_total_len is not None else sink_size + window_size

        if T <= total_budget:
            new_past.append((k, v))
            continue

        sink_len = min(sink_size, total_budget, T)
        remaining_budget = max(total_budget - sink_len, 0)
        window_len = min(window_size, max(T - sink_len, 0), remaining_budget)
        mid_budget = max(total_budget - sink_len - window_len, 0)

        k_sink = k[:, :, :sink_len, :]
        v_sink = v[:, :, :sink_len, :]

        if window_len > 0:
            k_window = k[:, :, -window_len:, :]
            v_window = v[:, :, -window_len:, :]
            mid_end = T - window_len
        else:
            k_window = k[:, :, :0, :]
            v_window = v[:, :, :0, :]
            mid_end = T

        k_mid = k[:, :, sink_len:mid_end, :]
        v_mid = v[:, :, sink_len:mid_end, :]

        if mid_budget <= 0 or k_mid.size(2) == 0:
            k_mid = k_mid[:, :, :0, :]
            v_mid = v_mid[:, :, :0, :]
        elif k_mid.size(2) < 2:
            k_mid = k_mid[:, :, -mid_budget:, :]
            v_mid = v_mid[:, :, -mid_budget:, :]
        else:
            k_proj = k_mid[..., :proj_dim] if proj_dim < D else k_mid
            k_norm = F.normalize(k_proj, dim=-1)

            sim = F.cosine_similarity(
                k_norm[:, :, :-1, :],
                k_norm[:, :, 1:, :],
                dim=-1
            )

            sim_mean = sim.mean(dim=1)

            # Keep enough candidates to fully use the shared cache budget.
            keep_k = int(k_mid.size(2) * keep_ratio)
            keep_k = max(keep_k, mid_budget)
            keep_k = min(keep_k, k_mid.size(2))
            score = -sim_mean  # low similarity better

            idx = torch.topk(score, keep_k, dim=-1).indices[0]
            idx = torch.sort(idx).values

            k_mid = k_mid.index_select(2, idx)
            v_mid = v_mid.index_select(2, idx)

            if k_mid.size(2) > mid_budget:
                k_mid = k_mid[:, :, -mid_budget:, :]
                v_mid = v_mid[:, :, -mid_budget:, :]

        k_new = torch.cat([k_sink, k_mid, k_window], dim=2)
        v_new = torch.cat([v_sink, v_mid, v_window], dim=2)

        new_past.append((k_new, v_new))

    return tuple(new_past)





def snapkv_plus_plus_cache(
    past_key_values,
    window_size=256,
    max_total_len=512,
    temperature=1.0
):
    new_past = []

    for k, v in past_key_values:
        B, H, T, D = k.shape

        if T <= max_total_len:
            new_past.append((k, v))
            continue

        window = min(window_size, T)
        prefix_budget = max_total_len - window

        k_window = k[:, :, -window:, :]
        v_window = v[:, :, -window:, :]
        k_prefix = k[:, :, :-window, :]
        v_prefix = v[:, :, :-window, :]

        if prefix_budget > 0 and k_prefix.size(2) > 0:

            q = k_window[:, :, -1:, :]  # query

            # ===== head-wise score =====
            # [B,H,T]
            score = torch.matmul(k_prefix, q.transpose(-1, -2)).squeeze(-1)

            # ❗ 不做 mean，改 rank fusion
            score = score  # keep [B,H,T]

            # rank aggregation（关键）
            score_rank = score.argsort(dim=-1).argsort(dim=-1).float()
            score = score_rank.mean(dim=1)  # [B,T]

            # mild recency bias（弱化）
            t = score.size(-1)
            recency = torch.linspace(0.8, 1.0, t, device=k.device)
            score = score * recency

            idx = torch.topk(score, min(prefix_budget, t), dim=-1).indices[0]
            idx = torch.sort(idx).values

            k_prefix = k_prefix.index_select(2, idx)
            v_prefix = v_prefix.index_select(2, idx)

        else:
            k_prefix = k_prefix[:, :, :0, :]
            v_prefix = v_prefix[:, :, :0, :]

        k_new = torch.cat([k_prefix, k_window], dim=2)
        v_new = torch.cat([v_prefix, v_window], dim=2)

        new_past.append((k_new, v_new))

    return tuple(new_past)

def snapkvpp_rkv_cache(
    past_key_values,
    keep_ratio=0.5,
    threshold=0.92,   # ⭐比0.95更宽松
    proj_dim=64,
    window_size=256,
    max_total_len=512
):

    new_past = []

    for k, v in past_key_values:
        B, H, T, D = k.shape

        if T <= max_total_len:
            new_past.append((k, v))
            continue

        # =====================
        # 1. split
        # =====================
        window = min(window_size, T)

        k_window = k[:, :, -window:, :]
        v_window = v[:, :, -window:, :]

        k_prefix = k[:, :, :-window, :]
        v_prefix = v[:, :, :-window, :]

        prefix_budget = max_total_len - window

        if prefix_budget <= 0 or k_prefix.size(2) == 0:
            k_prefix = k_prefix[:, :, :0, :]
            v_prefix = v_prefix[:, :, :0, :]
        else:

            # =====================
            # 2. Query-aware score（核心）
            # =====================
            q = k_window[:, :, -1:, :]  # [B,H,1,D]

            score = torch.matmul(
                k_prefix, q.transpose(-1, -2)
            ).squeeze(-1)  # [B,H,T]

            score = score.mean(dim=1)  # [B,T]

            # =====================
            # 3. weak recency bias（可控）
            # =====================
            t = score.size(-1)
            recency = torch.linspace(0.85, 1.0, t, device=k.device)
            score = score * recency

            # =====================
            # 4. NO softmax（关键修复）
            # =====================
            k_keep = min(prefix_budget * 2, k_prefix.size(2))

            idx = torch.topk(score, k_keep, dim=-1).indices[0]
            idx = torch.sort(idx).values

            k_prefix = k_prefix.index_select(2, idx)
            v_prefix = v_prefix.index_select(2, idx)

            # =====================
            # 5. RKV (light version)
            # =====================
            if k_prefix.size(2) >= 2:

                if proj_dim < D:
                    k_proj = k_prefix[..., :proj_dim]
                else:
                    k_proj = k_prefix

                k_norm = F.normalize(k_proj, dim=-1)

                sim = F.cosine_similarity(
                    k_norm[:, :, :-1, :],
                    k_norm[:, :, 1:, :],
                    dim=-1
                )

                sim_mean = sim.mean(dim=1)

                # ⭐ 只去掉“极高冗余”
                keep_mask = sim_mean <= threshold

                keep_mask = torch.cat([
                    keep_mask,
                    torch.ones(B, 1, device=k.device, dtype=torch.bool)
                ], dim=-1)

                idx2 = torch.nonzero(keep_mask[0], as_tuple=False).squeeze(-1)

                k_prefix = k_prefix.index_select(2, idx2)
                v_prefix = v_prefix.index_select(2, idx2)

                # =====================
                # 6. budget control
                # =====================
                if k_prefix.size(2) > prefix_budget:
                    k_prefix = k_prefix[:, :, -prefix_budget:, :]
                    v_prefix = v_prefix[:, :, -prefix_budget:, :]

        # =====================
        # 7. merge
        # =====================
        k_new = torch.cat([k_prefix, k_window], dim=2)
        v_new = torch.cat([v_prefix, v_window], dim=2)

        new_past.append((k_new, v_new))

    return tuple(new_past)


def apply_kv_optimization(past_key_values, method=None, **kwargs):
    if method == "truncate":
        return truncate_kv_cache(past_key_values, **kwargs)

    elif method == "streaming":
        return streaming_kv_cache(past_key_values, **kwargs)

    elif method == "rkv":
        return rkv_dedup_kv_cache(past_key_values, **kwargs)

    elif method == "snapkv":
        return snapkv_cache(past_key_values, **kwargs)

    elif method == "snapkv_rkv":
        return snapkv_rkv_cache(past_key_values, **kwargs)
    
    elif method == "snapkv_pp":
        return snapkv_plus_plus_cache(past_key_values, **kwargs)
    
    elif method == "snapkvpp_rkv":
        return snapkvpp_rkv_cache(past_key_values, **kwargs)

    return past_key_values
