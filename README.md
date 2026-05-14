# 代码说明文档

本项目实现了多种无训练、仅修改 KV Cache 的 Transformer 推理加速方法，并在 Pythia-70M 模型上评估其语言建模质量（困惑度）与生成速度。  
当前版本重点包含 **SnapKV++（query‑aware selection + rank fusion）** 以及 **RKV 余弦去冗余** 等算法，并提供了完整的 PPL 评估与生成速度测试流程。

---

## 1. 项目结构

```text
group_project/
├── README.md
├── main.py
├── model.py
├── kv_cache.py
├── generate.py
├── eval_ppl.py
├── requirements.txt
└── pg19_sample.txt
```

各文件作用：

- `main.py`：实验总入口，负责加载数据、运行 PPL 测试与生成性能测试
- `model.py`：加载 `pythia-70m` 模型和 tokenizer
- `kv_cache.py`：实现各类 KV Cache 压缩与压缩触发逻辑
- `generate.py`：执行自回归生成，并统计生成延迟与吞吐
- `eval_ppl.py`：提供基线 PPL 与带缓存 PPL 的评估函数
- `pg19_sample.txt`：本地 PG19 长文本样本



## 2. 环境依赖与运行方式

### 2.1 依赖库
- Python>=3.8
- torch>=2.0
- transformers==4.46.0
- datasets>=2.14
- tqdm>=4.65
- numpy>=1.23
- httpx[socks]==0.27.0


安装示例：
```bash
pip install -r requirements.txt
```

### 2.2 数据准备
项目当前使用两类数据：

- `PG19`：从本地 `pg19_sample.txt` 读取长文本样本
- `WikiText-2`：通过 `datasets` 自动下载测试集

操作方式：
- 仓库当前已附带一段 PG19 数据集中的文本，存储在 pg19_sample.txt。如需更换，可直接用相应的 PG19 长文本样本覆盖同名文件。  
- WikiText 数据集会自动通过 HuggingFace `datasets` 库下载，无需额外准备。

### 2.3 运行

在项目根目录执行：
```bash
python main.py
```
程序默认会完成以下流程：

1. 加载 `pythia-70m` 模型
2. 加载 `WikiText-2` 与本地 `PG19` 样本
3. 对 `PG19` 文本进行带 KV Cache 的 PPL 测试
4. 在 `PG19` 长提示和 `WikiText` 短提示上进行生成速度和性能测试，并输出指标
5. 输出不同方法的性能汇总表



## 3. 当前优化算法

`main.py` 中默认启用了以下四组实验配置：

- `Baseline`：不做 KV Cache 压缩
- `RKV-only`：基于余弦相似度进行冗余去除
- `SnapKV++`：基于 query-aware 打分和 rank fusion 的前缀选择
- `SnapKV++ + RKV`：先做前缀选择，再做轻量冗余去除

其中默认关键参数包括：

- `max_total_len = 512`
- `window_size = 256`
- `sink_size = 4`
- `proj_dim = 64`

如需调整实验设置，可直接修改 `main.py` 中 `cache_methods` 或 `run(max_total_len=...)` 的参数。



## 4. 核心方法说明

### 1. RKV

`rkv_dedup_kv_cache` 通过计算相邻 token 表示的余弦相似度，删除冗余较高的历史缓存项，同时保留最近窗口，以控制总缓存长度。

### 2. SnapKV++

`snapkv_plus_plus_cache` 在 SnapKV 的基础上引入更强的 query-aware 选择机制：

- 使用窗口最后一个 token 的 key 作为 query
- 对历史前缀做逐 head 打分
- 通过 rank fusion 聚合不同 head 的重要性排序
- 引入轻量 recency bias，使更近的历史 token 略有优势

### 3. SnapKV++ + RKV

`snapkvpp_rkv_cache` 先执行 query-aware 前缀筛选，再使用余弦相似度做二次去冗余，在相同预算下尽量保留更有信息量的历史上下文。



## 5. 项目评估内容

### PPL 评估

`eval_ppl.py` 提供两种评估方式：

- 滑动窗口 PPL：作为不依赖 cache 的参考基线
- 带 KV Cache 的逐 token PPL：更贴近真实推理过程

当前 `main.py` 默认运行的是 **PG19 上的带缓存 PPL 测试**。

### 生成性能评估

`generate.py` 会统计以下指标：

- `TTFT`：从输入提示到首个新 token 生成完成的总时间
- `First`：第一个生成 token 的单独耗时
- `TPOT`：平均每个 token 的生成时间
- `Throughput`：平均每秒生成 token 数

`main.py` 会对每种方法重复运行多次，输出均值和标准差，并分别在 PG19 与 WikiText 数据集上做对比。

### 压缩触发逻辑

项目在生成和带缓存 PPL 评估中采用统一的压缩触发策略：

- 只有当 KV Cache 长度超过预算时才考虑压缩
- 按固定步长检查是否触发压缩
- 仅在距离上次压缩后缓存有足够增长时执行压缩

这使得不同评估场景下的压缩行为更加一致，便于公平比较方法效果。



## 6. 其他注意事项

- 首次运行需要下载模型与 WikiText 数据集
- 在 CPU 上运行时，实验耗时会明显增加；且在CPU与GPU上分别运行时，各压缩算法之间的相对效率也不相同。
- `PG19` 样本文本过短时，测试结果的区分度会下降
- 如需启用滑动窗口 PPL 或 WikiText 的带缓存 PPL，可取消 `main.py` 中对应注释

