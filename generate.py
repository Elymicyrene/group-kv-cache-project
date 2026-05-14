# generate.py
import time
import torch
from kv_cache import apply_kv_optimization, should_optimize_kv_cache


def generate(model, tokenizer, device, prompt,
             max_new_tokens=500,
             kv_method=None,
             kv_params=None):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # ===== Prefill =====
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    prefill_time = time.time() - t0

    # ===== First token =====
    t1 = time.time()
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    first_token_time = time.time() - t1

    ttft = prefill_time + first_token_time

    tpot_list = [first_token_time]

    compress_interval = 256
    last_compress_len = 0

    for step in range(max_new_tokens - 1):
        step_start = time.time()

        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)

        step_time = time.time() - step_start
        tpot_list.append(step_time)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        # ✅ 关键优化
        # ===== KV compression trigger =====
        if should_optimize_kv_cache(
            past_key_values,
            method=kv_method,
            step_idx=step,
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

    avg_tpot = sum(tpot_list) / len(tpot_list)
    throughput = 1.0 / avg_tpot

    return ttft, first_token_time, avg_tpot, throughput, tpot_list
