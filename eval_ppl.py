import torch
from datasets import load_dataset
from tqdm import tqdm
from kv_cache import apply_kv_optimization, should_optimize_kv_cache


# ===== WikiText =====
def get_wikitext():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    texts = []
    for t in dataset["text"]:
        t = t.strip()
        if len(t) > 0:
            texts.append(t)

    return texts


# ===== PG19（本地文件）=====
def get_pg19():
    with open("pg19_sample.txt", "r", encoding="utf-8") as f:
        return f.read()


# ===== Sliding window raw =====
def compute_ppl_sliding_raw(model, input_ids, window_size=512, stride=256):
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    count = 0

    for start in range(0, seq_len - 1, stride):
        end = min(start + window_size, seq_len)

        input_chunk = input_ids[:, start:end]

        if input_chunk.size(1) < 2:
            continue

        with torch.no_grad():
            outputs = model(input_chunk, use_cache=False)

        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_chunk[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        nll = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )

        nll_sum += nll.item()
        count += shift_labels.numel()

    return nll_sum, count


# ===== WikiText baseline =====
def compute_ppl_wikitext(model, tokenizer, device, texts):
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts):
        enc = tokenizer(text, return_tensors="pt").input_ids.to(device)

        if enc.size(1) < 10:
            continue

        nll, count = compute_ppl_sliding_raw(model, enc)

        total_nll += nll
        total_tokens += count

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item()


# ===== PG19 baseline =====
def compute_ppl_pg19(model, tokenizer, device, text):
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)

    nll, count = compute_ppl_sliding_raw(model, enc)

    ppl = torch.exp(torch.tensor(nll / count))
    return ppl.item()


# =========================================================
# ⭐ 核心修复：KV Cache PPL（PG19）
# =========================================================
def compute_ppl_pg19_with_cache(
    model, tokenizer, device, text,
    kv_method=None,
    kv_params=None,
    max_tokens=4096
):
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)

    end = min(enc.size(1), max_tokens)

    nll_sum = 0.0
    count = 0
    past_key_values = None

    # ⭐ 新增：压缩控制
    compress_interval = 256
    last_compress_len = 0

    for i in range(end - 1):
        input_token = enc[:, i:i+1]
        target_token = enc[:, i+1:i+2]

        with torch.no_grad():
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # =========================
        # ⭐ 统一触发逻辑（关键修复）
        # =========================
        if should_optimize_kv_cache(
            past_key_values,
            method=kv_method,
            step_idx=i,
            last_compress_len=last_compress_len,
            compress_interval=compress_interval,
            **(kv_params or {})
        ):
            seq_len = past_key_values[0][0].size(2)
            past_key_values = apply_kv_optimization(
                past_key_values,
                method=kv_method,
                **(kv_params or {})
            )
            last_compress_len = seq_len

        # ===== loss =====
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, target_token).item()

        nll_sum += nll
        count += 1

    return torch.exp(torch.tensor(nll_sum / count)).item()


# =========================================================
# ⭐ 核心修复：KV Cache PPL（WikiText）
# =========================================================
def compute_ppl_wikitext_with_cache(
    model, tokenizer, device, texts,
    kv_method=None,
    kv_params=None,
    max_tokens=512
):
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt").input_ids.to(device)

        if enc.size(1) < 10:
            continue

        end = min(enc.size(1), max_tokens)

        past_key_values = None

        # ⭐ 新增：压缩控制
        compress_interval = 256
        last_compress_len = 0

        for i in range(end - 1):
            input_token = enc[:, i:i+1]
            target_token = enc[:, i+1:i+2]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # =========================
            # ⭐ 统一触发逻辑（关键修复）
            # =========================
            if should_optimize_kv_cache(
                past_key_values,
                method=kv_method,
                step_idx=i,
                last_compress_len=last_compress_len,
                compress_interval=compress_interval,
                **(kv_params or {})
            ):
                seq_len = past_key_values[0][0].size(2)
                past_key_values = apply_kv_optimization(
                    past_key_values,
                    method=kv_method,
                    **(kv_params or {})
                )
                last_compress_len = seq_len

            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, target_token).squeeze()

            total_nll += nll.item()
            total_tokens += 1

    if total_tokens == 0:
        return float("inf")

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()
