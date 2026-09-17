"""Benchmark sparse MLA without profiling, or inspect its GPU trace separately.

Default inputs model rank-local CP8 prefill after KV gathering. They are
synthetic inputs, not a replay of the model's actual Top-K selections.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import flashinfer
import numpy as np
import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as mla
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import (
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)

# ref python/sglang/srt/layers/attention/dsa_backend.py
SGLANG_FLASHINFER_WORKSPACE_SIZE = 384 * 1024 * 1024

# ref config
num_attention_heads = 64
kv_lora_rank = 512
qk_nope_head_dim = 192
qk_rope_head_dim = 64
index_topk = 2048

TRACE_KERNEL_NAME = (
    "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512"
    "PagedKvDenseStaticTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--chunk", type=int, choices=range(4), default=3)
    parser.add_argument("--rank", type=int, choices=range(8), default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument(
        "--repeat-iters", type=int, default=100, help="Bench iterations"
    )
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument(
        "--timing",
        choices=("cuda-event", "cuda-graph"),
        default="cuda-event",
        help="Bench only: eager CUDA Events, or Events around CUDA Graph replay",
    )
    parser.add_argument(
        "--cache", choices=("warm", "cold", "both"), default="both", help="Bench only"
    )
    parser.add_argument(
        "--trace-path", type=Path, help="Profile only: output Chrome trace JSON path"
    )
    parser.add_argument(
        "--with-stack", action="store_true", help="Profile only: collect Python stacks"
    )
    args = parser.parse_args()
    for name in ("warmup_iters", "repeat_iters", "profile_iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def warmup(kwargs, iters):
    for _ in range(iters):
        mla(**kwargs)
    torch.cuda.synchronize()


def print_config(kwargs, args):
    device = kwargs["query"].device
    print(
        f"GPU: {torch.cuda.get_device_name(device)}, "
        f"compute capability: {torch.cuda.get_device_capability(device)}"
    )
    print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"mode={args.mode}, chunk={args.chunk}, rank={args.rank}, seed={args.seed}")
    for name in ("query", "kv_cache", "block_tables", "seq_lens", "out"):
        tensor = kwargs[name]
        print(
            f"{name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, stride={tensor.stride()}"
        )
    print(
        f"backend={kwargs['backend']}, enable_dcp={kwargs['enable_dcp']}, "
        f"enable_pdl={kwargs['enable_pdl']}, max_seq_len={kwargs['max_seq_len']}, "
        f"topk={kwargs['sparse_mla_top_k']}, bmm1_scale={kwargs['bmm1_scale']}"
    )


def bench(kwargs, args):
    print(f"[BENCH] profiler=off, timing={args.timing}")
    print("Measures GPU elapsed time for the API call, not an individual kernel.")
    if args.timing == "cuda-event":
        print("Eager timing can include host launch gaps between CUDA Events.")
    else:
        print(
            "CUDA Graph replay reduces host launch overhead; "
            "the prefill trace is eager."
        )
    cache_modes = {"warm": (False,), "cold": (True,), "both": (False, True)}
    for cold in cache_modes[args.cache]:
        times_ms = bench_gpu_time(
            mla,
            input_kwargs=kwargs,
            enable_cupti=False,
            use_cuda_graph=args.timing == "cuda-graph",
            cold_l2_cache=cold,
            dry_run_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
        )
        print(
            f"[BENCH] cold_l2={cold}, samples={len(times_ms)}: "
            f"mean={statistics.mean(times_ms) * 1000:.2f} us, "
            f"median={statistics.median(times_ms) * 1000:.2f} us"
        )


def profile_mla(kwargs, args):
    # Import and enable the profiler only in the separate profile mode.
    from torch.profiler import ProfilerActivity, profile, record_function

    print("[PROFILE] Diagnostic timings only: profiling can perturb kernel execution.")
    print("Eager calls, no L2 flush; --timing and --cache apply only to bench.")
    trace_path = args.trace_path or Path(
        f"sm103_mla_chunk{args.chunk}_rank{args.rank}_{time.time_ns()}.json"
    )
    trace_path = trace_path.expanduser().resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as prof:
        for i in range(args.profile_iters):
            with record_function(f"mla_probe_{i}"):
                mla(**kwargs)
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    print(f"[PROFILE] Trace saved to: {trace_path}")
    events = json.loads(trace_path.read_text())["traceEvents"]
    groups = defaultdict(list)
    for event in events:
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}:
            if "dur" in event:
                groups[(event["cat"], event["name"])].append(event)
    if not any(category == "kernel" for category, _ in groups):
        raise RuntimeError(
            "No GPU kernel events captured; check profiler/CUPTI support."
        )

    for (category, name), rows in sorted(
        groups.items(),
        key=lambda item: sum(event["dur"] for event in item[1]),
        reverse=True,
    ):
        durations = [event["dur"] for event in rows]  # Chrome trace uses microseconds.
        print(f"\n[PROFILE/{category}] {name}")
        print(
            f"count={len(rows)}, mean={statistics.mean(durations):.2f} us, "
            f"median={statistics.median(durations):.2f} us"
        )
        configs = {
            json.dumps(
                {
                    key: event.get("args", {}).get(key)
                    for key in (
                        "grid",
                        "block",
                        "registers per thread",
                        "shared memory",
                    )
                },
                sort_keys=True,
            )
            for event in rows
        }
        for config in sorted(configs):
            print(config)

    matched = ("kernel", TRACE_KERNEL_NAME) in groups
    print(f"\n[PROFILE] Original prefill kernel name matched: {matched}")
    print(
        "Reference: grid=[1,1,4096], block=[512,1,1], "
        "registers=128, smem=219776 bytes"
    )


def make_inputs(
    chunk: int,
    rank: int = 0,
    workspace_bytes: int = SGLANG_FLASHINFER_WORKSPACE_SIZE,
    seed: int = 1234,
):
    assert 0 <= chunk < 4
    assert 0 <= rank < 8

    device = torch.device("cuda:0")
    batch, heads, topk, page_size = 4096, 64, 2048, 64
    context_len = (chunk + 1) * 32768

    causal_lens = chunk * 32768 + rank + np.arange(batch, dtype=np.int64) * 8 + 1
    valid_lens = np.minimum(causal_lens, topk).astype(np.int32)

    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    slots = np.full((batch, topk), -1, dtype=np.int32)
    for i, causal_len in enumerate(causal_lens):
        n = int(valid_lens[i])
        slots[i, :n] = rng.choice(int(causal_len), n, replace=False)

    def random_fp8(shape):
        return torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        ).to(torch.float8_e4m3fn)

    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch, heads, get_device_sm_count(device)
    )

    return dict(
        query=random_fp8((batch, 1, heads, 576)),
        kv_cache=random_fp8((context_len // page_size, 1, page_size, 576)),
        workspace_buffer=torch.zeros(workspace_bytes, device=device, dtype=torch.uint8),
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.from_numpy(slots).to(device).unsqueeze(1),
        seq_lens=torch.from_numpy(valid_lens).to(device),
        max_seq_len=context_len,
        sparse_mla_top_k=topk,
        out=torch.empty(
            (batch, 1, heads, 512),
            device=device,
            dtype=torch.bfloat16,
        ),
        bmm1_scale=(qk_nope_head_dim + qk_rope_head_dim) ** -0.5,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        is_var_seq=True,
        enable_dcp=False,
        enable_pdl=None,
        multi_ctas_kv_counter_buffer=torch.zeros(
            counter_bytes, device=device, dtype=torch.uint8
        ),
    )


# def test():
#     torch.manual_seed(42)
#     device = "cuda"

#     ## prefill
#     batch_size = 4096
#     q_len_per_request = 1
#     num_pages = 65535
#     page_size = 64

#     query = torch.randn(
#         batch_size,
#         q_len_per_request,
#         num_attention_heads,
#         kv_lora_rank + qk_rope_head_dim,
#         device=device,
#         dtype=torch.float8_e4m3fn,
#     )
#     kv_cache = torch.randn(
#         num_pages,
#         1,
#         page_size,
#         kv_lora_rank + qk_rope_head_dim,
#         device=device,
#         dtype=torch.float8_e4m3fn,
#     )
#     workspace_buffer = torch.empty(
#         SGLANG_FLASHINFER_WORKSPACE_SIZE, device=device, dtype=torch.uint8
#     )
#     sm_count = flashinfer.utils.get_device_sm_count(device)
#     required_bytes = flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
#         batch_size, num_attention_heads, sm_count
#     )
#     multi_ctas_kv_counter_buffer = torch.zeros(
#         required_bytes, device=device, dtype=torch.uint8
#     )
#     block_tables = torch.zeros(
#         (batch_size, 1, index_topk), sdevice=device, dtype=torch.int8
#     )
#     seq_lens = [batch_size]
#     max_seq_len=32768
#     bmm1_scale = 1.0 * 1.0 * (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

#     flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
#         query=query,
#         kv_cache=kv_cache,
#         workspace_buffer=workspace_buffer,
#         qk_nope_head_dim=qk_nope_head_dim,
#         kv_lora_rank=kv_lora_rank,
#         qk_rope_head_dim=qk_rope_head_dim,
#         block_tables=block_tables,
#         seq_lens=seq_lens,
#         max_seq_len=,
#         sparse_mla_top_k=index_topk,
#         bmm1_scale=bmm1_scale,
#         backend="trtllm-gen",
#         skip_softmax_threshold_scale_factor=False,
#         multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
#     )


if __name__ == "__main__":
    args = parse_args()
    kwargs = make_inputs(chunk=args.chunk, rank=args.rank, seed=args.seed)
    print_config(kwargs, args)
    warmup(kwargs, args.warmup_iters)
    if args.mode == "bench":
        bench(kwargs, args)
    else:
        profile_mla(kwargs, args)
