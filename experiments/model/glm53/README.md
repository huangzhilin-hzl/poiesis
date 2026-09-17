# GLM-5.3 CP8 sparse MLA：TRTLLM / DeepSeek FlashMLA 对照

问题：在 baseline prefill 的单卡 attention 工作量下，DeepSeek 的 SM100 sparse MLA 是否比 TRTLLM-GEN 更快？本实验只测 attention API，不测模型整层、CP 通信、Indexer、Top-K 生成或 TTFT。

## Workload 定义

| 项目 | 所有 backend 共用的定义 |
| --- | --- |
| 默认模型请求 | 一个 128k 请求，4 个 32768-token chunk |
| CP | 固定 CP8，query 按 token 交错分配 |
| chunk size | `--chunk-size`，默认 32768，必须是正的 64 倍数 |
| 本卡 query 数 | `--local-tokens`；未指定时为 `chunk_size / 8`，默认 4096 |
| chunk | `--chunk` 是从 0 开始的编号，默认 3；KV context 为 `(chunk + 1) * chunk_size` |
| 本卡第 i 个 query 的因果长度 | `chunk * chunk_size + rank + i * 8 + 1` |
| Attention | 64 Q heads、1 KV head、QK=576、V=512、Top-K=2048、page size=64 |
| 有效 key 数 | `min(causal_length, 2048)`，不足部分索引为 -1 |
| QK scale | `(192 + 64) ** -0.5 = 0.0625`，不能换成 `576 ** -0.5` |
| 原始输入 | 同一随机种子生成的 FP8 E4M3 Q/KV，量化 scale 为 1 |
| 稀疏索引 | 所有 backend 复用同一张索引表和有效长度；在因果范围内无放回随机采样 |

这里是 geometry 与因果约束一致的合成输入，不是原模型真实 Q/KV 和 Indexer Top-K 的重放。各 backend 会有不同的计算精度、访存布局和调度方式。

`local_tokens` 可以小于 `chunk_size / 8`：此时只取该 rank 当前 chunk 的前 N 个 query，KV 容量仍为完整 context；JSON 标记为 `rank-chunk-prefix`。等于上限时为完整 rank workload，标记为 `full-rank-chunk`。超过上限会越出当前 chunk，因此单组运行报错，矩阵运行记录到 `skipped` 并跳过。以下形状中的 N 表示实际 `local_tokens`。

## 三条执行路径

| 名称 | 实际 API | Q | KV |
| --- | --- | --- | --- |
| `trtllm` | FlashInfer `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` | FP8 `[N,1,64,576]` | FP8 `[pages,1,64,576]` |
| `flashmla-prefill` | DeepSeek `flash_mla_sparse_fwd` | BF16 `[N,64,576]` | BF16 `[context,1,576]` |
| `flashmla-decode` | DeepSeek `flash_mla_with_kvcache(is_fp8_kvcache=True, indices=...)` | BF16 `[N,1,64,576]` | uint8 `[pages,64,1,656]`，V3.2 FP8 cache 格式 |

FlashMLA 的 SM100 系列实现也支持 B300/SM103。这里 `flashmla-decode` 表示用 sparse decode API 承担同一批 prefill query 的计算，4096 是独立 sparse query 行数，不代表重新构造了 4096 个不同的模型请求。两条 FlashMLA 路径均保留 576 维，未切换到 DeepSeek V4 的 512 维 NoPE workload。

656 字节/token 的打包方式：前 512 字节保留原 FP8 NoPE，后接 4 个值为 1 的 FP32 scale，再接 64 个 BF16 RoPE（128 字节）。原 FP8 RoPE 到 BF16 是精确转换；Q 转 BF16 也不引入新的输入舍入。这样三条路径表示的是同一个数学 attention 问题。GPU 运行时通过 FP32 参考检验 kernel 数值误差。

稀疏因果关系由索引和有效长度表达，所以 FlashMLA decode 必须传 `causal=False`。FlashMLA scheduler metadata 在 warmup 中初始化，固定输入下复用，计时不包含首次调度准备。

接口依据：[FlashMLA interface（固定 commit）](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/flash_mla/flash_mla_interface.py)、[SM100 sparse prefill dispatch](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/csrc/api/sparse_prefill.cpp)、[SM100 sparse decode dispatch](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/csrc/api/sparse_decode.cpp)。

## 计时与正确性

默认 `--backends trtllm flashmla-prefill flashmla-decode --scope both`，产生 5 行 case：

- `trtllm/native`：现有 baseline API 参数，预分配 output/workspace。
- `flashmla-prefill/native` / `flashmla-decode/native`：输入已转换好，只计 API 执行。
- `flashmla-prefill/adapted` / `flashmla-decode/adapted`：每次调用都从原始 FP8 Q 和**整个历史 KV cache**转换/打包，再执行 API。

`adapted` 用来评估直接接入现有输入格式的成本。它不是维护原生 KV cache、仅写入新 token 的实现，也不是整模型端到端时间；真实集成应单独测增量写 cache 的成本。`native` 也不是只测主 kernel：FlashMLA API 的辅助 GPU 操作仍包含在内；eager 调用还可能包含输出分配和 host launch 造成的空隙。FlashMLA 额外计算 LSE/max logits 等 API 输出，这些成本未剔除。

- 默认 8 个 query 行与独立 FP32 sparse attention 参考比较，覆盖首尾和 chunk 0 的 Top-K 长度边界；所有输出检查 finite。绝对/相对容差默认 0.01/0.05，输出 max abs、RMSE 和 relative RMSE。失败即退出；这不代替真实模型精度评估。
- `--check-rows` 大于等于本卡 query 数时全量检查，代价较高；`--check-rows 0` 显式跳过。小 batch 会自动限制抽样数，边界行仅在有效范围内抽取。
- `--timing cuda-event` 默认，接近 baseline 的 eager 调用方式。
- `--timing cuda-graph` 捕获一次调用后 replay，不与 eager 数字混算。
- `--cache warm` 表示重复复用同一输入，不保证全部数据放得进 L2。
- `--cache cold` 在计时区间前写入至少 256 MiB、且不小于已报告 L2 大小两倍的 buffer，作尽力而为的 L2 eviction；未暴露 L2 大小时按 128 MiB 估计。cold flush 不计入耗时。
- 两种 timing 使用同一套 CUDA Events harness。与旧 `modal_result.txt` 使用的 FlashInfer timing helper 不完全一致，应重新测 TRTLLM 作为同轮 baseline。
- 表格输出 median/p05/p95 和 `TRT median / 当前 median`；比值 >1 表示当前 case 更快。JSON 另含 mean 和逐次样本。无 TRTLLM 对照时比值为 null。
- `profile` 单独运行，trace 内用 `mla_probe/<backend>/<scope>/<iteration>` 标记，并按 case 输出 kernel 名、耗时、占该 case kernel 总时间比例和 launch 配置。比例是累计 kernel 时长之比，不是 wall-time 占比。Profiler 数字仅用于诊断，不作为加速比。

## 运行

在仓库根目录执行。已有 CUDA 13.2 / PyTorch 2.12 / FlashInfer 0.6.18.post1 / 编译好的 FlashMLA 环境时：

```bash
python experiments/model/glm53/sm103_mla.py --chunk 3 --scope both --output-json /tmp/mla_chunk3.json
python experiments/model/glm53/sm103_mla.py --chunk 0 --scope native --timing cuda-graph --output-json /tmp/mla_chunk0_graph.json
python experiments/model/glm53/sm103_mla.py --mode profile --chunk 3 --scope native --trace-path /tmp/mla_trace.json --output-json /tmp/mla_profile_summary.json
```

Modal runner 会在 B300 上执行，自动安装固定版本的 FlashInfer，并源码编译 FlashMLA commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`。上游 build 同时生成 sm_100a / sm_103a，禁用无关 SM90 target；首次构建时间较长。构建显式设置 `CC=gcc`、`CXX=g++`，使用镜像内已安装的 GCC，避免链接阶段调用不存在的 `clang++`。源码下载与编译分层缓存，`pip -v` 保留完整编译日志。已有 Modal 登录态可直接运行：

```bash
uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py --chunk 3 --scope both --output-json /tmp/mla_chunk3.json
```

先跑 chunk 0（覆盖 partial Top-K）和 chunk 3（128k KV）。完整四段对比：

```bash
for chunk in 0 1 2 3; do
  uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py --chunk "$chunk" --scope both --output-json "/tmp/mla_chunk${chunk}.json"
done
```

## 扩展测试矩阵

单组指定 chunk size 与本卡 query 数：

```bash
uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py \
  --chunk 3 --chunk-size 16384 --local-tokens 1024 \
  --scope native --output-json /tmp/mla_c3_s16384_n1024.json
```

使用任意一个复数参数即进入矩阵模式，按 `chunks × chunk_sizes × local_tokens_list` 的笛卡尔积执行，重复值只测一次。未给 `--local-tokens-list` 或 `--local-tokens` 时，每个 chunk size 自动取完整 CP8 份额。

```bash
uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py \
  --chunks 0 3 \
  --chunk-sizes 8192 16384 32768 65536 \
  --local-tokens-list 512 1024 2048 4096 8192 \
  --scope native --timing cuda-event --cache both \
  --output-json /tmp/mla_matrix.json
```

这条命令有 28 组有效 workload、12 组超出 CP8 份额而跳过的组合；每组比较 3 条 native 路径和 2 种 cache，共 168 行计时结果。一次 Modal 调用在同一块 GPU 上顺序执行，每组后释放显存；镜像与已加载的依赖复用。可加 `--timing cuda-graph` 进行另一轮对比，或用 `--scope both` 包含输入适配成本。

矩阵终表列出 chunk 编号、chunk size、local tokens、KV 长度、backend、cache、median、query/s 和 `TRT/this`。query/s 为本卡 `local_tokens / API 耗时`，不是整模型生成吞吐。`TRT/this` 只和**同一 workload、同一 cache** 的 TRT 耗时相比。

JSON 包含 `runs`（各组完整配置、正确性和原始样本）、`summary`（可直接转成表格）、`skipped` 和 `status`。每完成一组就保存一次；遇到正确性或 CUDA 错误立即停止并标记 `failed_workload`，Modal runner 会先下载已有 JSON，再以非零状态退出。容器强制终止或客户端断连时不保证下载。

固定 chunk 编号时，改变 chunk size 也会改变 KV 长度；本矩阵并非固定 128k context。若要固定 131072-token KV，四种 chunk size 分别需要 `--chunk 15/7/3/1`。固定同一 chunk size 与 chunk 编号，改变 local tokens，才是在相同 KV 容量下改变 query 数。

本机先预览组合，不占用 GPU（需要 NumPy 和 PyTorch，但无需 CUDA）：

```bash
python experiments/model/glm53/sm103_mla.py --dry-run \
  --chunks 0 3 --chunk-sizes 8192 16384 32768 65536 \
  --local-tokens-list 512 1024 2048 4096 8192
```

矩阵模式只支持 bench。Profile 时用单数参数选定一组配置，避免把多组 trace 混在一起。

提取 kernel 耗时与 launch 参数：

```bash
uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py --mode profile --chunk 3 --scope native --profile-iters 5 --output-json /tmp/mla_profile_summary.json
```

Modal runner 只下载 `--output-json` 指定的结果 JSON。Profile trace 默认在远端临时目录生成，用于提取并打印 kernel 统计，运行结束后删除，不回传本机。如果给 runner 传 `--trace-path`，它表示远端路径。直接在本机运行 `sm103_mla.py` 时，仍可通过 `--trace-path` 保存本机 trace。单测原 baseline 可加 `--backends trtllm`；本地只测它时不要求安装 FlashMLA。

## B300 矩阵实测（2026-09-17）

上述 28 组 native 矩阵已在 B300 完成，使用 CUDA Event、warmup 20 次、每组/cache 采样 100 次、每条路径对照 8 行 FP32 参考。84 次正确性检查全部通过，168 行计时结果见 [完整 JSON](results/matrix_native_cuda_event_20260917.json) 和 [汇总 CSV](results/matrix_native_cuda_event_20260917.csv)。

TRTLLM-GEN 在本轮全部 56 个「workload × cache」组合中中位耗时最低。以下摘取 `chunk=3`、完整 CP8 份额、warm cache 的结果（单位 μs）：

| chunk size | local tokens | KV context | TRTLLM-GEN | FlashMLA prefill | FlashMLA decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1024 | 32768 | 215.82 | 269.12 | 400.26 |
| 16384 | 2048 | 65536 | 418.27 | 550.38 | 810.59 |
| 32768 | 4096 | 131072 | 909.60 | 1228.98 | 1668.61 |
| 65536 | 8192 | 262144 | 2017.12 | 2847.82 | 3289.84 |

这里的输入精度、API 分配行为仍按前文定义，且跨行同时改变了 query 数和 KV 长度；不能将差距解释为相同精度下的纯 kernel 优劣或整模型加速比。`modal_result.txt` 是旧 TRTLLM 实验记录，应与本轮数据分开使用。

补充边界检查中，`chunk=0, chunk_size=64, local_tokens=8, rank=0, seed=1234` 的 TRT 路径未通过默认 FP32 容差：最大绝对误差约 0.04353。此配置已在计时前停止，不作为有效性能结果；默认容差没有放宽。它不在上述 28 组矩阵中。

另有 4 组 CUDA Graph 回归（`chunk=3`、chunk size 为 8192/16384、本卡 query 数为 512/1024、`scope=both`），20 次正确性检查全部通过。该回归每项只采样 3 次，用于验证矩阵执行与输入适配，未混入上述性能结果。

CPU 回归检查（需要 pytest、NumPy、PyTorch）：

```bash
python -m pytest -q experiments/model/glm53/test_sm103_mla.py
```
