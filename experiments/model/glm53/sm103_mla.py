"""Compare TRTLLM and DeepSeek FlashMLA on rank-local CP8 sparse prefill.

Synthetic inputs reproduce the baseline geometry, not actual model Top-K values.
Native timing excludes input adaptation; adapted timing includes converting the
entire canonical FP8 Q/KV cache on every call. See README.md for interpretation.
"""

import argparse
import bisect
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import torch

SGLANG_FLASHINFER_WORKSPACE_SIZE = 384 * 1024 * 1024
HEADS, D_QK, D_V, TOPK, PAGE_SIZE = 64, 576, 512, 2048, 64
DEFAULT_CHUNK_SIZE, CP_SIZE = 32768, 8
# The absorbed QK width is 576, but the model's pre-absorption QK width is 256.
SOFTMAX_SCALE = (192 + 64) ** -0.5
BACKENDS = ("trtllm", "flashmla-prefill", "flashmla-decode")


@dataclass
class Case:
    name: str
    run: Callable[[], torch.Tensor]
    layout: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument(
        "--scope", choices=("native", "adapted", "both"), default="both"
    )
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--chunk", type=int, help="Zero-based chunk index; default: 3")
    position.add_argument("--chunks", type=int, nargs="+", help="Matrix: chunk indices")
    position.add_argument(
        "--total-tokens", type=int,
        help="Fix request length and test its last chunk: chunk = total-tokens / chunk-size - 1",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--local-tokens",
        type=int,
        help="Query rows on this rank; default: chunk-size / 8",
    )
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", help="Matrix: global chunk sizes"
    )
    parser.add_argument(
        "--local-tokens-list",
        type=int,
        nargs="+",
        help="Matrix: rank-local query counts",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print workload plan without using CUDA"
    )
    parser.add_argument("--rank", type=int, choices=range(8), default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument(
        "--timing", choices=("cuda-event", "cuda-graph"), default="cuda-event"
    )
    parser.add_argument("--cache", choices=("warm", "cold", "both"), default="both")
    parser.add_argument(
        "--check-rows", type=int, default=8, help="FP32 reference rows; 0 skips"
    )
    parser.add_argument("--check-atol", type=float, default=0.01)
    parser.add_argument("--check-rtol", type=float, default=0.05)
    parser.add_argument("--trace-path", type=Path, help="Profile: Chrome trace output")
    parser.add_argument(
        "--output-json", type=Path, help="Configuration, checks and timings"
    )
    parser.add_argument(
        "--with-stack", action="store_true", help="Profile: Python stacks"
    )
    args = parser.parse_args(argv)
    if args.chunk is None:
        args.chunk = 3
    for name in ("warmup_iters", "repeat_iters", "profile_iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.check_rows < 0:
        parser.error("seed and check-rows must be nonnegative")
    if any(c < 0 for c in [args.chunk, *(args.chunks or [])]):
        parser.error("chunk indices must be nonnegative")
    if any(
        s <= 0 or s % PAGE_SIZE for s in [args.chunk_size, *(args.chunk_sizes or [])]
    ):
        parser.error(f"chunk sizes must be positive multiples of {PAGE_SIZE}")
    if args.total_tokens is not None:
        if not 0 < args.total_tokens <= np.iinfo(np.int32).max:
            parser.error("total-tokens must be positive and fit int32")
        for size in args.chunk_sizes or [args.chunk_size]:
            if args.total_tokens % size:
                parser.error("total-tokens must be divisible by every chunk size")
    local_counts = [*(args.local_tokens_list or [])]
    if args.local_tokens is not None:
        local_counts.append(args.local_tokens)
    if any(n <= 0 for n in local_counts):
        parser.error("local token counts must be positive")
    if not all(np.isfinite(v) and v >= 0 for v in (args.check_atol, args.check_rtol)):
        parser.error("check tolerances must be finite and nonnegative")
    args.backends = list(dict.fromkeys(args.backends))
    args.matrix = any((args.chunks, args.chunk_sizes, args.local_tokens_list))
    if args.matrix and args.mode != "bench":
        parser.error(
            "matrix runs support --mode bench; profile one configuration at a time"
        )
    workloads, skipped = plan_workloads(args)
    if not workloads:
        parser.error("no valid workloads: " + skipped[0]["reason"])
    if skipped and not args.matrix:
        parser.error(skipped[0]["reason"])
    return args


def workload_error(chunk, chunk_size, local_tokens):
    if local_tokens > chunk_size // CP_SIZE:
        return f"local_tokens={local_tokens} exceeds chunk_size / CP8={chunk_size // CP_SIZE}"
    if (chunk + 1) * chunk_size > np.iinfo(np.int32).max:
        return "context length exceeds the int32 index range"
    return None


def plan_workloads(args):
    """Pair size/index for a fixed request, or use explicitly selected indices."""
    workloads, skipped = [], []
    chunks = list(dict.fromkeys(args.chunks or [args.chunk]))
    sizes = list(dict.fromkeys(args.chunk_sizes or [args.chunk_size]))
    positions = (
        [(args.total_tokens // size - 1, size) for size in sizes]
        if args.total_tokens is not None
        else [(chunk, size) for chunk in chunks for size in sizes]
    )
    for chunk, size in positions:
            counts = args.local_tokens_list or [
                args.local_tokens if args.local_tokens is not None else size // CP_SIZE
            ]
            for count in dict.fromkeys(counts):
                coordinates = dict(chunk=chunk, chunk_size=size, local_tokens=count)
                error = workload_error(chunk, size, count)
                if error:
                    skipped.append(dict(**coordinates, reason=error))
                    continue
                workloads.append(argparse.Namespace(**(vars(args) | coordinates)))
    return workloads, skipped


def make_sparse_indices(
    chunk, rank, seed, batch=None, topk=TOPK, chunk_size=DEFAULT_CHUNK_SIZE
):
    """Physical token slots in one shared, contiguous KV cache; CP interleaves Q."""
    batch = chunk_size // CP_SIZE if batch is None else batch
    causal_lens = (
        chunk * chunk_size + rank + np.arange(batch, dtype=np.int64) * CP_SIZE + 1
    )
    valid_lens = np.minimum(causal_lens, topk).astype(np.int32)
    rng = np.random.default_rng(seed)
    slots = np.full((batch, topk), -1, dtype=np.int32)
    for i, causal_len in enumerate(causal_lens):
        n = int(valid_lens[i])
        slots[i, :n] = rng.choice(int(causal_len), n, replace=False)
    return slots, valid_lens


def make_inputs(
    chunk, rank=0, seed=1234, chunk_size=DEFAULT_CHUNK_SIZE, local_tokens=None
):
    device = torch.device("cuda:0")
    local_tokens = chunk_size // CP_SIZE if local_tokens is None else local_tokens
    context_len = (chunk + 1) * chunk_size
    slots, valid_lens = make_sparse_indices(
        chunk, rank, seed, batch=local_tokens, chunk_size=chunk_size
    )
    generator = torch.Generator(device=device).manual_seed(seed)

    def random_fp8(shape):
        return torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        ).to(torch.float8_e4m3fn)

    return dict(
        query=random_fp8((local_tokens, 1, HEADS, D_QK)),
        kv_cache=random_fp8((context_len // PAGE_SIZE, 1, PAGE_SIZE, D_QK)),
        block_tables=torch.from_numpy(slots).to(device).unsqueeze(1),
        seq_lens=torch.from_numpy(valid_lens).to(device),
        max_seq_len=context_len,
    )


def pack_flashmla_kv(kv_cache):
    """Lossless V3.2-layout adapter for unit-scale FP8 baseline KV, NOT V4.

    656 bytes/token = 512 FP8 NoPE + 4 FP32 scales + 64 BF16 RoPE.
    NoPE bytes are preserved and scales are 1; FP8 -> BF16 RoPE is exact.
    """
    if kv_cache.dtype != torch.float8_e4m3fn or kv_cache.shape[-1] != D_QK:
        raise ValueError("Expected unit-scale E4M3 KV with width 576")
    pages, kv_heads, page_size, _ = kv_cache.shape
    if kv_heads != 1:
        raise ValueError("This benchmark requires one shared KV head")
    kv = kv_cache.view(pages, page_size, 1, D_QK)
    packed = torch.empty((*kv.shape[:-1], 656), dtype=torch.uint8, device=kv.device)
    packed[..., :512].copy_(kv[..., :512].view(torch.uint8))
    packed[..., 512:528].view(torch.float32).fill_(1.0)
    packed[..., 528:].copy_(kv[..., 512:].to(torch.bfloat16).view(torch.uint8))
    return packed


def make_trtllm_case(inputs):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    q = inputs["query"]
    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        q.shape[0], HEADS, get_device_sm_count(q.device)
    )
    kwargs = dict(
        **inputs,
        workspace_buffer=torch.zeros(
            SGLANG_FLASHINFER_WORKSPACE_SIZE, device=q.device, dtype=torch.uint8
        ),
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        sparse_mla_top_k=TOPK,
        out=torch.empty((*q.shape[:-1], D_V), device=q.device, dtype=torch.bfloat16),
        bmm1_scale=SOFTMAX_SCALE,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        is_var_seq=True,
        enable_dcp=False,
        enable_pdl=None,
        multi_ctas_kv_counter_buffer=torch.zeros(
            counter_bytes, device=q.device, dtype=torch.uint8
        ),
    )
    return Case(
        "trtllm/native",
        lambda: trtllm_batch_decode_with_kv_cache_mla(**kwargs),
        f"Q FP8 [{q.shape[0]},1,64,576]; KV FP8 [pages,1,64,576]; preallocated output",
    )


def make_flashmla_case(inputs, backend, scope):
    try:
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install pinned DeepSeek FlashMLA; see README.md / Modal runner"
        ) from error

    q_fp8, kv_fp8 = inputs["query"], inputs["kv_cache"]
    indices, lengths = inputs["block_tables"], inputs["seq_lens"]
    is_prefill = backend == "flashmla-prefill"

    def prepare():
        q = q_fp8.to(torch.bfloat16)
        if is_prefill:
            return q.squeeze(1), kv_fp8.view(-1, 1, D_QK).to(torch.bfloat16)
        return q, pack_flashmla_kv(kv_fp8)

    prepared = prepare() if scope == "native" else None
    # Metadata is initialized during correctness/warmup, before graph capture.
    # Reuse is valid: shapes and topk_length values do not change between calls.
    metadata, num_splits = get_mla_metadata() if not is_prefill else (None, None)

    def run():
        q, kv = prepared if prepared is not None else prepare()
        if is_prefill:
            return flash_mla_sparse_fwd(
                q=q,
                kv=kv,
                indices=indices,
                sm_scale=SOFTMAX_SCALE,
                d_v=D_V,
                attn_sink=None,
                topk_length=lengths,
            )[0]
        return flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=D_V,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            softmax_scale=SOFTMAX_SCALE,
            causal=False,  # Causality is already encoded in indices and lengths.
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=None,
            topk_length=lengths,
        )[0]

    layout = (
        f"Q BF16 [{q_fp8.shape[0]},64,576]; KV BF16 [context,1,576]"
        if is_prefill
        else f"Q BF16 [{q_fp8.shape[0]},1,64,576]; KV uint8 [pages,64,1,656] (V3.2 packed FP8)"
    )
    return Case(f"{backend}/{scope}", run, layout)


def make_cases(inputs, args):
    scopes = ("native", "adapted") if args.scope == "both" else (args.scope,)
    cases = []
    for backend in args.backends:
        if backend == "trtllm":
            cases.append(make_trtllm_case(inputs))
        else:
            cases.extend(make_flashmla_case(inputs, backend, scope) for scope in scopes)
    return cases


def reference_rows(inputs, rows):
    """Independent FP32 sparse softmax(QK * scale)V on selected query rows."""
    kv = inputs["kv_cache"].view(-1, D_QK).float()
    results = []
    for row in rows:
        n = int(inputs["seq_lens"][row].item())
        slots = inputs["block_tables"][row, 0, :n].long()
        slots = slots[(slots >= 0) & (slots < kv.shape[0])]
        if slots.numel() == 0:
            raise ValueError("Reference requires at least one valid key per row")
        selected_kv = kv[slots]
        q = inputs["query"][row, 0].float()
        probabilities = torch.softmax((q @ selected_kv.T) * SOFTMAX_SCALE, dim=-1)
        results.append(probabilities @ selected_kv[:, :D_V])
    return torch.stack(results)


def select_check_rows(args, batch):
    count = min(args.check_rows, batch)
    # First row with a full Top-K; it can be outside this chunk or local prefix.
    remaining = TOPK - (args.chunk * args.chunk_size + args.rank + 1)
    transition = -(-remaining // CP_SIZE)
    candidates = [0, batch - 1, transition - 1, transition]
    candidates += np.linspace(0, batch - 1, count, dtype=int).tolist()
    return list(dict.fromkeys(r for r in candidates if 0 <= r < batch))[:count]


def check_cases(cases, inputs, args):
    if args.check_rows == 0:
        print("[CHECK] SKIPPED by --check-rows 0")
        return {"status": "skipped"}
    batch = inputs["query"].shape[0]
    rows = select_check_rows(args, batch)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = reference_rows(inputs, rows)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    checks = {}
    for case in cases:
        output = case.run()
        expected = (batch, HEADS, D_V)
        if output.dtype != torch.bfloat16 or tuple(output.shape) not in (
            expected,
            (batch, 1, HEADS, D_V),
        ):
            raise AssertionError(
                f"{case.name}: unexpected output {output.shape}/{output.dtype}"
            )
        if not torch.isfinite(output).all().item():
            raise AssertionError(f"{case.name}: output contains nonfinite values")
        actual = output.reshape(expected)[rows].float()
        error = actual - reference
        rmse = error.square().mean().sqrt().item()
        checks[case.name] = dict(
            rows=rows,
            max_abs=error.abs().max().item(),
            rmse=rmse,
            relative_rmse=rmse / max(reference.square().mean().sqrt().item(), 1e-12),
            atol=args.check_atol,
            rtol=args.check_rtol,
        )
        torch.testing.assert_close(
            actual,
            reference,
            atol=args.check_atol,
            rtol=args.check_rtol,
            msg=lambda message: f"{case.name}: {message}",
        )
        print(f"[CHECK] {case.name}: PASS {json.dumps(checks[case.name])}")
    return checks


def warmup(case, iters):
    for _ in range(iters):
        case.run()
    torch.cuda.synchronize()


def measure_case(case, args, flush_buffer):
    warmup(case, args.warmup_iters)
    graph, graph_output = None, None
    if args.timing == "cuda-graph":
        # PyTorch recommends warming up on a side stream before graph capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.warmup_iters):
                case.run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = case.run()
        graph.replay()
        torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.repeat_iters)
    ]
    # Instantiate event handles before measuring; their first record can cost CPU time.
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    for start, end in events:
        if flush_buffer is not None:
            flush_buffer.zero_()  # Same stream, before start; excluded from measurement.
        start.record()
        if graph is None:
            output = case.run()
        else:
            graph.replay()
            output = graph_output
        end.record()
    torch.cuda.synchronize()
    # Keep the last output / graph allocations alive through synchronization.
    del output
    return [start.elapsed_time(end) * 1000 for start, end in events]


def bench(cases, args):
    print(f"[BENCH] profiler=off; API elapsed time; timing={args.timing}")
    print(
        "Eager Events include host launch gaps; Graph replay changes that execution regime."
    )
    print("warm=repeated input reuse; cold=best-effort L2 eviction before each call.")
    print("native excludes adapters; adapted converts FULL Q and KV on EVERY call.")
    print("FlashMLA APIs allocate output/auxiliary tensors; TRTLLM reuses its output.")
    cache_modes = ("warm", "cold") if args.cache == "both" else (args.cache,)
    properties = torch.cuda.get_device_properties(0)
    l2_bytes = getattr(properties, "L2_cache_size", 0) or 128 * 1024 * 1024
    flush_bytes = max(2 * l2_bytes, 256 * 1024 * 1024)
    records = []
    for cache in cache_modes:
        flush = (
            torch.empty(flush_bytes, device="cuda", dtype=torch.uint8)
            if cache == "cold"
            else None
        )
        for case in cases:
            samples = measure_case(case, args, flush)
            records.append(
                dict(
                    case=case.name,
                    cache=cache,
                    timing=args.timing,
                    mean_us=statistics.mean(samples),
                    median_us=statistics.median(samples),
                    queries_per_second=args.local_tokens
                    * 1e6
                    / statistics.median(samples),
                    p05_us=float(np.percentile(samples, 5)),
                    p95_us=float(np.percentile(samples, 95)),
                    samples_us=samples,
                    l2_flush_bytes=flush_bytes if cache == "cold" else 0,
                )
            )
        del flush
    baseline = {
        r["cache"]: r["median_us"] for r in records if r["case"] == "trtllm/native"
    }
    print(
        f"\n{'case':30} {'cache':5} {'median/us':>11} {'p05/us':>11} {'p95/us':>11} {'TRT/this':>10}"
    )
    for result in records:
        ratio = baseline.get(result["cache"], 0) / result["median_us"]
        result["speedup_vs_trtllm"] = ratio if result["cache"] in baseline else None
        ratio_text = (
            f"{ratio:.3f}x" if result["speedup_vs_trtllm"] is not None else "n/a"
        )
        print(
            f"{result['case']:30} {result['cache']:5} {result['median_us']:11.2f} "
            f"{result['p05_us']:11.2f} {result['p95_us']:11.2f} {ratio_text:>10}"
        )
    return records


def profile_mla(cases, args):
    from torch.profiler import ProfilerActivity, profile, record_function

    print("[PROFILE] Diagnostic eager timings; --timing/--cache only apply to bench.")
    trace_path = args.trace_path or Path(
        f"sm103_mla_chunk{args.chunk}_{time.time_ns()}.json"
    )
    trace_path = trace_path.expanduser().resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        warmup(case, args.warmup_iters)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as prof:
        for case in cases:
            for i in range(args.profile_iters):
                with record_function(f"mla_probe/{case.name}/{i}"):
                    output = case.run()
                    # The scope includes completion, so GPU timestamps belong to one case.
                    torch.cuda.synchronize()
                    del output
    prof.export_chrome_trace(str(trace_path))
    print(f"[PROFILE] Trace saved to: {trace_path}")
    events = json.loads(trace_path.read_text())["traceEvents"]
    scopes = sorted(
        (
            e["ts"],
            e["ts"] + e["dur"],
            e["name"].rsplit("/", 1)[0].removeprefix("mla_probe/"),
        )
        for e in events
        if e.get("name", "").startswith("mla_probe/") and "dur" in e
    )
    starts = [s[0] for s in scopes]
    groups = defaultdict(list)
    for event in events:
        if (
            event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}
            or "dur" not in event
        ):
            continue
        idx = bisect.bisect_right(starts, event["ts"]) - 1
        case_name = (
            scopes[idx][2]
            if idx >= 0 and event["ts"] + event["dur"] <= scopes[idx][1]
            else "unattributed"
        )
        groups[(case_name, event["cat"], event["name"])].append(event)
    if not any(category == "kernel" for _, category, _ in groups):
        raise RuntimeError("No GPU kernels captured; check CUPTI/profiler support")
    totals = defaultdict(float)
    for (case_name, category, _), rows in groups.items():
        if category == "kernel":
            totals[case_name] += sum(e["dur"] for e in rows)
    report = []
    for (case_name, category, name), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], -sum(e["dur"] for e in item[1]))
    ):
        durations = [e["dur"] for e in rows]
        total = sum(durations)
        configs = {
            json.dumps(
                {
                    k: e.get("args", {}).get(k)
                    for k in ("grid", "block", "registers per thread", "shared memory")
                },
                sort_keys=True,
            )
            for e in rows
        }
        row = dict(
            case=case_name,
            category=category,
            kernel=name,
            count=len(rows),
            total_us=total,
            mean_us=statistics.mean(durations),
            per_call_us=total / args.profile_iters,
            kernel_share_pct=(
                100 * total / totals[case_name] if category == "kernel" else None
            ),
            launch_configs=[json.loads(c) for c in sorted(configs)],
        )
        report.append(row)
        print(f"[PROFILE] {json.dumps(row)}")
    return dict(trace_path=str(trace_path), kernels=report)


def configuration(inputs, cases, args):
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for package in ("flashinfer-python", "flash-mla"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    versions["flashmla_source_revision"] = os.environ.get(
        "FLASH_MLA_SOURCE_REVISION", "unreported"
    )
    config = dict(
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        versions=versions,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        workload=dict(
            total_tokens=args.total_tokens,
            chunk=args.chunk,
            chunk_size=args.chunk_size,
            local_tokens=inputs["query"].shape[0],
            query_selection=(
                "full-rank-chunk"
                if args.local_tokens == args.chunk_size // CP_SIZE
                else "rank-chunk-prefix"
            ),
            cp=CP_SIZE,
            context_len=inputs["max_seq_len"],
            heads=HEADS,
            qk_dim=D_QK,
            v_dim=D_V,
            topk=TOPK,
            page_size=PAGE_SIZE,
            softmax_scale=SOFTMAX_SCALE,
            bmm2_scale=1.0,
            synthetic=True,
            enable_dcp=False,
        ),
        tensors={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), stride=list(v.stride()))
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        },
        cases={case.name: case.layout for case in cases},
    )
    print(json.dumps(config, indent=2))
    return config


def run_workload(args):
    inputs = make_inputs(
        chunk=args.chunk,
        rank=args.rank,
        seed=args.seed,
        chunk_size=args.chunk_size,
        local_tokens=args.local_tokens,
    )
    cases = make_cases(inputs, args)
    result = {"configuration": configuration(inputs, cases, args)}
    with torch.inference_mode():
        result["correctness"] = check_cases(cases, inputs, args)
        if args.mode == "bench":
            result["benchmarks"] = bench(cases, args)
        else:
            result["profile"] = profile_mla(cases, args)
    return result


def write_result(result, path):
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def workload_coordinates(args):
    return dict(
        chunk=args.chunk,
        chunk_size=args.chunk_size,
        local_tokens=args.local_tokens,
        context_len=(args.chunk + 1) * args.chunk_size,
    )


def print_matrix_summary(rows):
    print(
        f"\n[MATRIX SUMMARY]\n{'chunk':>5} {'chunk_size':>10} {'local':>6} "
        f"{'KV len':>8} {'case':30} {'cache':5} {'median/us':>11} "
        f"{'queries/s':>11} {'TRT/this':>10}"
    )
    for row in rows:
        ratio = row["speedup_vs_trtllm"]
        ratio_text = f"{ratio:.3f}x" if ratio is not None else "n/a"
        print(
            f"{row['chunk']:5} {row['chunk_size']:10} {row['local_tokens']:6} "
            f"{row['context_len']:8} {row['case']:30} {row['cache']:5} "
            f"{row['median_us']:11.2f} {row['queries_per_second']:11.0f} {ratio_text:>10}"
        )


def main(argv=None):
    args = parse_args(argv)
    workloads, skipped = plan_workloads(args)
    if args.dry_run:
        plan = dict(
            status="planned",
            workloads=[workload_coordinates(w) for w in workloads],
            skipped=skipped,
        )
        print(json.dumps(plan, indent=2))
        write_result(plan, args.output_json)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment needs an SM100/SM103 CUDA GPU")
    if torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3)):
        raise RuntimeError("This experiment targets SM100/SM103")
    if not args.matrix:
        result = run_workload(workloads[0])
    else:
        print(f"[MATRIX] {len(workloads)} valid workloads, {len(skipped)} skipped")
        for item in skipped:
            print(f"[SKIP] {json.dumps(item)}")
        result = dict(status="running", skipped=skipped, runs=[], summary=[])
        for index, workload in enumerate(workloads, 1):
            coordinates = workload_coordinates(workload)
            print(
                f"[MATRIX] {index}/{len(workloads)} {json.dumps(coordinates)}",
                flush=True,
            )
            try:
                run = run_workload(workload)
            except Exception as error:
                # Stop on a failed correctness check or CUDA error. Earlier runs
                # remain available; a poisoned CUDA context must not be reused.
                result.update(
                    status="failed",
                    failed_workload=dict(
                        **coordinates, error=f"{type(error).__name__}: {error}"
                    ),
                )
                write_result(result, args.output_json)
                print_matrix_summary(result["summary"])
                raise
            result["runs"].append(run)
            result["summary"].extend(
                coordinates | {k: v for k, v in row.items() if k != "samples_us"}
                for row in run["benchmarks"]
            )
            write_result(result, args.output_json)
            # Results contain only CPU scalars/JSON. Release each workload's Q,
            # KV, adapters and graph allocations before moving to the next size.
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
        print_matrix_summary(result["summary"])
    write_result(result, args.output_json)
    if args.output_json:
        print(f"[RESULT] {args.output_json.expanduser().resolve()}")


if __name__ == "__main__":
    main()
