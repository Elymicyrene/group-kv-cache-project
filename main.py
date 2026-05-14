import math

from model import load_model
from eval_ppl import (
    get_wikitext, get_pg19,
    compute_ppl_wikitext, compute_ppl_pg19,
    compute_ppl_wikitext_with_cache,
    compute_ppl_pg19_with_cache
)
from generate import generate


# ===== 工具函数1：多次运行取平均 =====
def run_generation_avg(model, tokenizer, device, prompt,
                       kv_method, kv_params,
                       max_new_tokens=640,
                       repeat=3):

    # ===== 预热：消除首次运行的额外开销 =====
    print(f"  Warmup (discarded)...")
    generate(model, tokenizer, device, prompt,
             max_new_tokens=max_new_tokens,
             kv_method=kv_method,
             kv_params=kv_params)

    all_tpot = []
    ttft_list, first_list, tpot_avg_list, th_list = [], [], [], []

    for i in range(repeat):
        print(f"  Run {i+1}/{repeat}...")

        ttft, first_tok, avg_tpot, throughput, tpot = generate(
            model, tokenizer, device, prompt,
            max_new_tokens=max_new_tokens,
            kv_method=kv_method,
            kv_params=kv_params
        )

        all_tpot.append(tpot)
        ttft_list.append(ttft)
        first_list.append(first_tok)
        tpot_avg_list.append(avg_tpot)
        th_list.append(throughput)

        print(f"    TTFT: {ttft:.4f}s, First: {first_tok:.4f}s, "
              f"TPOT: {avg_tpot:.4f}s, Throughput: {throughput:.2f} tok/s")

    # ===== 对齐长度 =====
    min_len = min(len(t) for t in all_tpot)
    avg_tpot_curve = []

    for i in range(min_len):
        avg_tpot_curve.append(sum(run[i] for run in all_tpot) / repeat)

    # ===== 标量取平均 & 标准差 =====
    avg_ttft = sum(ttft_list) / repeat
    avg_first = sum(first_list) / repeat
    avg_tpot = sum(tpot_avg_list) / repeat
    avg_throughput = sum(th_list) / repeat

    std_ttft = math.sqrt(sum((x - avg_ttft) ** 2 for x in ttft_list) / repeat)
    std_first = math.sqrt(sum((x - avg_first) ** 2 for x in first_list) / repeat)
    std_tpot = math.sqrt(sum((x - avg_tpot) ** 2 for x in tpot_avg_list) / repeat)
    std_throughput = math.sqrt(sum((x - avg_throughput) ** 2 for x in th_list) / repeat)

    print(f"  => TTFT: {avg_ttft:.4f}±{std_ttft:.4f}s, "
          f"First: {avg_first:.4f}±{std_first:.4f}s, "
          f"TPOT: {avg_tpot:.4f}±{std_tpot:.4f}s, "
          f"Throughput: {avg_throughput:.2f}±{std_throughput:.2f} tok/s")

    return (avg_ttft, avg_first, avg_tpot, avg_throughput, avg_tpot_curve,
            std_ttft, std_first, std_tpot, std_throughput)


# ===== 工具函数2：汇总表格 =====
def print_summary_table(pg_results, wiki_results, cache_methods, pg_len, wiki_len):
    metrics = [
        ("TTFT (s)",      0, 5),
        ("First (s)",     1, 6),
        ("TPOT (s)",      2, 7),
        ("Throughput",    3, 8),
    ]

    def fmt_method(name, result, is_baseline):
        if result is None:
            return [name] + ["—"] * len(metrics)
        parts = [name]
        for m in metrics:
            mi, si = m[1], m[2]
            if m[0] == "Throughput":
                parts.append(f"{result[mi]:.2f}±{result[si]:.2f}")
            else:
                parts.append(f"{result[mi]:.4f}±{result[si]:.4f}")
        return parts

    def print_section(title, results):
        print(f"\n{title}")
        header = ["Method"] + [m[0] for m in metrics]
        col_widths = [max(len(x), 18) for x in header[:1]] + [max(len(x), 20) for x in header[1:]]

        def fmt_row(row):
            return "  ".join(f"{c:<{w}}" for c, w in zip(row, col_widths))

        print(fmt_row(header))
        print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

        baseline = results.get("Baseline")
        for name, _, _ in cache_methods:
            row = fmt_method(name, results.get(name), name == "Baseline")
            print(fmt_row(row))

    print("\n" + "=" * 90)
    print("  Generation Performance Summary".center(86))
    print("=" * 90)

    print_section(f"PG19 ({pg_len} tokens prompt)", pg_results)
    print_section(f"WikiText ({wiki_len} tokens prompt)", wiki_results)


# ===== 主函数 =====
def run(max_total_len=512):
    print("Loading model...")
    model, tokenizer, device = load_model()

    print("Loading datasets...")
    wiki_texts = get_wikitext()
    pg_text = get_pg19()

    """
    # =========================
    # 1. Sliding Window PPL
    # =========================
    print("\n===== Sliding Window PPL =====")
    wiki_subset = wiki_texts[:100]

    ppl_wiki = compute_ppl_wikitext(model, tokenizer, device, wiki_subset)
    print(f"WikiText (baseline): {ppl_wiki:.2f}")

    ppl_pg = compute_ppl_pg19(model, tokenizer, device, pg_text)
    print(f"PG19 (baseline): {ppl_pg:.2f}")
    """

    # =========================
    # 2. KV Cache PPL
    # =========================
    print("\n===== KV Cache PPL =====")

    cache_methods = [
        ("Baseline", None, {}),
        ("RKV-only", "rkv", {
            "sink_size": 4,
            "window_size": 256,
            "proj_dim": 64,
            "keep_ratio": 0.90,
            "max_total_len": max_total_len
        }),
        ("SnapKV++", "snapkv_pp", {
            "temperature": 1.0,
            "window_size": 256,
            "max_total_len": max_total_len
        }),
        ("SnapKV++ + RKV", "snapkvpp_rkv", {
            "keep_ratio": 0.5,
            "threshold": 0.95,
            "proj_dim": 64,
            "window_size": 256,
            "max_total_len": max_total_len
        }),
    ]
    """
    for name, method, params in cache_methods:
        ppl = compute_ppl_wikitext_with_cache(
            model, tokenizer, device, wiki_subset,
            kv_method=method, kv_params=params
        )
        print(f"WikiText - {name}: {ppl:.2f}")
    """

    for name, method, params in cache_methods:
        ppl = compute_ppl_pg19_with_cache(
            model, tokenizer, device, pg_text,
            kv_method=method, kv_params=params
        )
        print(f"PG19 - {name}: {ppl:.2f}")

    # =========================
    # 3. Generation (PG19)
    # =========================
    print("\n===== PG19 Generation =====")

    pg_tokens = tokenizer(pg_text, return_tensors="pt").input_ids.to(device)
    pg_len = 9600
    prompt_pg = tokenizer.decode(pg_tokens[0, :pg_len])

    pg_results = {}

    for name, method, params in cache_methods:
        print(f"\nRunning {name}...")

        result = run_generation_avg(
            model, tokenizer, device,
            prompt_pg,
            kv_method=method,
            kv_params=params,
            max_new_tokens=640,
            repeat=5
        )

        pg_results[name] = result

    # =========================
    # 4. Generation (WikiText)
    # =========================
    print("\n===== WikiText Generation =====")

    wiki_tokens = tokenizer(wiki_texts[2226], return_tensors="pt").input_ids.to(device)
    wiki_len = 800
    prompt_wiki = tokenizer.decode(wiki_tokens[0, :wiki_len])

    wiki_results = {}

    for name, method, params in cache_methods:
        print(f"\nRunning {name}...")

        result = run_generation_avg(
            model, tokenizer, device,
            prompt_wiki,
            kv_method=method,
            kv_params=params,
            max_new_tokens=640,
            repeat=5
        )

        wiki_results[name] = result

    # ===== 汇总表格 =====
    print_summary_table(pg_results, wiki_results, cache_methods, pg_len, wiki_len)


if __name__ == "__main__":
    run()
