# 代码说明文档

本项目实现了多种无训练、仅修改 KV Cache 的 Transformer 推理加速方法，并在 Pythia-70M 模型上评估其语言建模质量（困惑度）与生成速度。  
当前版本重点包含 **SnapKV++（query‑aware selection + rank fusion）** 以及 **RKV 余弦去冗余** 等算法，并提供了完整的 PPL 评估与生成速度测试流程。

---

## 1. 项目结构

```
project/
├── main.py          # 实验主入口，控制评估流程
├── model.py         # 模型与分词器加载
├── kv_cache.py      # KV Cache 压缩算法实现
├── generate.py      # 自回归生成与计时
├── eval_ppl.py      # 困惑度评估函数（含带 cache 版本）
└── pg19_sample.txt  # PG19 数据集样本（需自行准备）
```

---

## 2. 环境依赖与运行方式

### 2.1 依赖库
- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers ≥ 4.40
- Datasets
- tqdm
- matplotlib（仅绘图需要）

安装示例：
```bash
pip install torch transformers datasets tqdm matplotlib
```

### 2.2 数据准备
- PG19 数据集需下载单个样本，命名为 `pg19_sample.txt` 放在项目根目录。  
- WikiText 数据集会自动通过 HuggingFace `datasets` 库下载，无需额外准备。

### 2.3 运行
修改 `main.py` 末尾的 `run(max_total_len=...)` 参数（默认 512），然后执行：
```bash
python main.py
```
程序将自动完成以下操作：
1. 加载 Pythia-70M 模型
2. 对 PG19 文本进行带 KV Cache 的 PPL 测试
3. 分别在 PG19 长提示和 WikiText 短提示上进行生成速度测试，并输出指标和 TPOT 曲线图

---

## 3. 模块说明

### 3.1 `model.py` – 模型加载
- **函数** `load_model(device=None)`  
  加载 `EleutherAI/pythia-70m` 模型及其分词器，自动选择设备（GPU/CPU），并将模型设为 `eval` 模式。  
  *注意*：代理设置已注释，如需要可通过环境变量 `HTTP_PROXY` 自行配置。

### 3.2 `kv_cache.py` – KV Cache 压缩算法
所有压缩方法通过统一接口 `apply_kv_optimization(past_key_values, method, **kwargs)` 调用。

| 方法名 (`method`) | 函数 | 核心原理 |
|-------------------|------|----------|
| `"truncate"` | `truncate_kv_cache` | 仅保留最后 `max_length` 个 token |
| `"streaming"` | `streaming_kv_cache` | 保留前 `sink_size` 个 token + 最后 `window_size` 个 token |
| `"rkv"` | `rkv_dedup_kv_cache` | 余弦相似度去冗余（软剪枝），控制保留比例 |
| `"snapkv"` | `snapkv_cache` | 基于 key 模长的全局 top‑k 选择 (前缀压缩) |
| `"snapkv_rkv"` | `snapkv_rkv_cache` | SnapKV + 余弦去重 |
| `"snapkv_pp"` | `snapkv_plus_plus_cache` | **SnapKV++**：query‑aware 评分 + rank fusion + top‑k 选择 |
| `"snapkvpp_rkv"` | `snapkvpp_rkv_cache` | SnapKV++ + 余弦去重 |

**关键设计**：
- 所有方法遵循“**完整保留最近 `window_size` 个 token，仅压缩更早的历史前缀**”的策略，以保证近期上下文完整。
- 压缩仅在缓存长度超过 `max_total_len` 时触发（生成与 PPL 评估一致）。
- `snapkv_plus_plus_cache` 使用窗口最后一个 token 的 key 作为 query，计算与前缀 key 的点积得分，并通过 head 维度的 rank fusion 聚合，最后加入温和的 recency bias；不进行 softmax 温度归一化。
- RKV 类方法在 SnapKV 选择后，对前缀按相邻余弦相似度去掉极高冗余的 token。

### 3.3 `generate.py` – 自回归生成与速度测量
- **函数** `generate(model, tokenizer, device, prompt, max_new_tokens, kv_method, kv_params)`  
  - **Prefill**：一次性输入全部提示，获得初始 KV Cache。
  - **首 token 生成**：单独计时，计入 TTFT。
  - **后续生成**：逐 token 循环，每步记录时间，遇到 EOS 提前终止。
  - **压缩触发**：对于 SnapKV 系列方法，每 `compress_interval` 步检查一次，若累计增长超过 64 且总长度超过 `max_total_len`，则调用压缩。
  - 返回：`TTFT`, `first_token_time`, `avg_tpot`, `throughput`, `tpot_list`。

### 3.4 `eval_ppl.py` – 困惑度评估
提供两种 PPL 计算方式：

#### A. 滑动窗口 PPL（无缓存，理想参考）
- `compute_ppl_sliding_raw`：将文本切分为定长窗口，独立前向计算交叉熵，取加权平均。
- `compute_ppl_wikitext` / `compute_ppl_pg19`：基于上述函数计算整个数据集的 PPL（在 `main.py` 中被注释，可按需启用）。

#### B. 带 KV Cache 的逐 token PPL（真实推理模拟）
- `compute_ppl_pg19_with_cache` / `compute_ppl_wikitext_with_cache`：逐 token 输入，维护 `past_key_values`，并按照与生成阶段一致的压缩触发逻辑进行 KV Cache 压缩，最终计算整体 PPL。

**数据集辅助函数**：
- `get_wikitext()`：返回 WikiText-2 测试集的非空文档列表。
- `get_pg19()`：读取本地 `pg19_sample.txt` 文件内容。

### 3.5 `main.py` – 实验主控
`run(max_total_len=512)` 函数按顺序执行：
1. **加载模型与数据**。
2. **KV Cache PPL 测试**（PG19）：对 Baseline、RKV-only、SnapKV++、SnapKV+++RKV 四种配置分别计算带缓存的 PPL。
3. **PG19 生成速度测试**：使用长提示（前 3000 字符）生成 300 token，重复 3 次取平均，输出 TTFT、平均 TPOT、吞吐量，并绘制 TPOT 曲线图 (`tpot_pg19.png`)。
4. **WikiText 生成速度测试**：类似步骤 3，但使用短提示（第一篇文档前 1000 字符），结果保存为 `tpot_wikitext.png`。

**工具函数**：
- `smooth(data, window)`：滑动平均，用于曲线平滑。
- `run_generation_avg(...)`：多次运行生成测试并平均，返回标量指标和平均后的 TPOT 序列。
- `plot_curve(results, title, save_name)`：绘制多条 TPOT 曲线对比图。

**注释部分**：
- 滑动窗口 PPL 测试已注释，如需对比请取消注释并传入 `wiki_subset`（第 79‑85 行）。
- WikiText 带缓存 PPL 也暂时注释。

---

## 4. 关键参数说明

在 `main.py` 中可调整以下常用压缩参数：

```python
max_total_len = 512        # KV Cache 最大总长度
window_size = 256          # 始终完整保留的最近窗口大小
sink_size = 4              # StreamingLLM 的 sink token 数
keep_ratio = 0.5           # top‑k 保留比例（部分方法）
temperature = 1.0          # SnapKV++ 的温度系数
threshold = 0.95 / 0.92    # 余弦去重阈值
proj_dim = 64              # 降维维度（RKV 中可选）
```

---

## 5. 输出结果解读

程序运行后将在终端打印：

- **PG19 PPL**：各压缩方法在长文本文档上的困惑度（数值越低越好）。
- **生成速度指标**：
  - `TTFT`：从输入提示到生成第一个新 token 的总时间。
  - `First token`：第一个 token 单独耗时。
  - `Avg TPOT`：后续 token 的平均生成时间。
  - `Throughput`：每秒生成的 token 数。

同时生成两张 TPOT 曲线图：
- `tpot_pg19.png`：长文本生成时不同方法的 TPOT 变化趋势。
- `tpot_wikitext.png`：短文本生成时的对比。

---

## 6. 注意事项

- 首次运行会自动下载 Pythia-70M 模型（约 350 MB），请确保网络通畅。
- 如需在离线环境运行，可提前下载模型并修改 `model_name` 为本地路径。
- CPU 推理较慢，建议使用较短的测试文本（如已设 `max_tokens=4096`）。
- 代理设置已注释，如遇下载问题请自行配置 `HTTP_PROXY` 环境变量。
- `pg19_sample.txt` 文件需用户自行从 PG19 数据集中抽取一段长文本（不少于 5000 字符）放入根目录，否则程序无法读取 PG19 数据。

---

以上即本项目的完整代码说明，可供组员快速了解架构、运行实验及自定义修改。