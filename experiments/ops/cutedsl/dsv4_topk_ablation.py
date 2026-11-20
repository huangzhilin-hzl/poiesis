"""4-stage latency ablation for DeepSeek-V4 MoE top-6 routing in CuTe-DSL.

Decomposes single-kernel execution time of the CuTe-DSL ``redux.sync`` + PDL
DeepSeek-V4 router (``E in {256, 384}``, ``top_k = 6``) into four stages under
CUDA Graph replay (warm L2 and cold L2):

  1. Empty    : CUDA Graph empty-kernel launch + retire floor.
  2. MemOnly  : Read ``correction_bias`` and ``gating_output`` logits + write
                6 outputs (global memory latency floor).
  3. +Math    : Add ``sqrt(softplus(logit)) + bias``, NaN -> -1e30, -0.0 -> +0.0
                canonicalization, and float32 -> order-preserving uint32 key.
  4. FullTopK : Add top-6 ``warp_redux_sync`` selection (4-warp two-stage when
                ``M <= 4 * sm_count``, 1-warp zero-SMEM otherwise) and 8-lane
                butterfly renormalization.

Example usage:
    python experiments/ops/cutedsl/dsv4_topk_ablation.py
    python experiments/ops/cutedsl/dsv4_topk_ablation.py \
        --num-experts 384 --ms 1,4,8,16,64,256,512,4096
"""

import argparse
import json
import statistics
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

_TOPK = 6
_WARP_SCHEDULERS_PER_SM = 4
_SmemAllocator = getattr(cutlass, "memory", cutlass.utils).SmemAllocator


@cache
def _multi_warp_max_tokens(device_index: int) -> int:
    """Switch from 4-warp to 1-warp once 1-warp CTAs saturate SM schedulers."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return _WARP_SCHEDULERS_PER_SM * sm_count


def _asm(
    result,
    operands: list,
    text: str,
    constraints: str,
    side_effects: bool,
    loc,
    ip,
):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32(val, loc, ip):
    return Float32(val).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _load_f32_nc(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _store_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _f32(val, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_s32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


_SOFTPLUS_ARG_PTX = (
    "{\n"
    ".reg .f32 a;\n"
    "mul.f32 a, $1, 0f3FB8AA3B;\n"
    "ex2.approx.f32 a, a;\n"
    "add.f32 $0, a, 0f3F800000;\n"
    "}"
)

_SCORE_KEY_PTX = (
    "{\n"
    ".reg .pred p;\n"
    ".reg .f32 l, s, v;\n"
    ".reg .b32 b, n;\n"
    "setp.gt.f32 p, $2, 0f41A00000;\n"
    "selp.f32 l, $2, $3, p;\n"
    "sqrt.approx.ftz.f32 s, l;\n"
    "add.f32 v, s, $4;\n"
    "setp.num.f32 p, v, v;\n"
    "selp.f32 v, v, 0fF149F2CA, p;\n"
    "setp.eq.f32 p, v, 0f00000000;\n"
    "selp.f32 v, 0f00000000, v, p;\n"
    "mov.b32 b, v;\n"
    "setp.lt.s32 p, b, 0;\n"
    "not.b32 n, b;\n"
    "or.b32 b, b, 0x80000000;\n"
    "selp.b32 $1, n, b, p;\n"
    "mov.f32 $0, s;\n"
    "}"
)


@dsl_user_op
def _score_and_key(logit, bias, *, loc=None, ip=None):
    arg = Float32(
        _asm(
            T.f32(),
            [_f32(logit, loc, ip)],
            _SOFTPLUS_ARG_PTX,
            "=f,f",
            False,
            loc,
            ip,
        )
    )
    log_arg = cute.math.log(arg, fastmath=False)
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [_f32(logit, loc, ip), _f32(log_arg, loc, ip), _f32(bias, loc, ip)],
        _SCORE_KEY_PTX,
        "=f,=r,f,f,f",
        False,
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _add_f32(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _normalized_weight(scale, total, weight, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(scale, loc, ip), _f32(total, loc, ip), _f32(weight, loc, ip)],
            "{\n"
            ".reg .pred p;\n"
            ".reg .f32 d;\n"
            "setp.gt.f32 p, $2, 0f00000000;\n"
            "selp.f32 d, $2, 0f3F800000, p;\n"
            "div.full.f32 d, $1, d;\n"
            "mul.f32 $0, d, $3;\n"
            "}",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


class EmptyKernel:
    """Stage 1: empty kernel launch + retire floor."""

    def __init__(self, threads: int, use_pdl: bool) -> None:
        self.threads = threads
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(self, num_tokens: Int32, stream: cuda.CUstream):
        self.kernel().launch(
            grid=(num_tokens, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self):
        pass


class MemOnlyKernel:
    """Stage 2: bias + logits global loads and 6-slot stores only."""

    def __init__(self, num_experts: int, num_warps: int, use_pdl: bool) -> None:
        self.num_experts = num_experts
        self.threads = num_warps * 32
        self.experts_per_thread = num_experts // self.threads
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(gating_output, correction_bias, topk_weights, topk_ids).launch(
            grid=(num_tokens, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        experts_per_thread = self.experts_per_thread
        warp_base = warp * (32 * experts_per_thread)

        acc = Float32(0.0)
        row_base = gating_output + Int64(row) * (self.num_experts * 4)
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            bv = _load_f32_nc(correction_bias + Int64(e) * 4)
            lv = _load_f32_nc(row_base + Int64(e) * 4)
            acc = acc + bv + lv
        if tid < _TOPK:
            _store_f32(topk_weights + (Int64(row) * _TOPK + Int64(tid)) * 4, acc)
            _store_s32(topk_ids + (Int64(row) * _TOPK + Int64(tid)) * 4, Int32(tid))


class MathOnlyKernel:
    """Stage 3: loads + sqrt(softplus) + bias + float32->uint32 key + stores."""

    def __init__(self, num_experts: int, num_warps: int, use_pdl: bool) -> None:
        self.num_experts = num_experts
        self.threads = num_warps * 32
        self.experts_per_thread = num_experts // self.threads
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(gating_output, correction_bias, topk_weights, topk_ids).launch(
            grid=(num_tokens, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        experts_per_thread = self.experts_per_thread
        warp_base = warp * (32 * experts_per_thread)

        acc = Float32(0.0)
        kacc = Uint32(0)
        row_base = gating_output + Int64(row) * (self.num_experts * 4)
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            bv = _load_f32_nc(correction_bias + Int64(e) * 4)
            lv = _load_f32_nc(row_base + Int64(e) * 4)
            s, k = _score_and_key(lv, bv)
            acc = acc + s
            kacc = kacc ^ k
        if tid < _TOPK:
            _store_f32(topk_weights + (Int64(row) * _TOPK + Int64(tid)) * 4, acc)
            _store_s32(topk_ids + (Int64(row) * _TOPK + Int64(tid)) * 4, Int32(kacc))


class FullTopkKernel:
    """Stage 4: full CuTe-DSL redux.sync + PDL DSv4 top-6 router (1W or 4W)."""

    def __init__(self, num_experts: int, num_warps: int, use_pdl: bool) -> None:
        self.num_experts = num_experts
        self.num_warps = num_warps
        self.threads = num_warps * 32
        self.experts_per_thread = num_experts // self.threads
        self.num_merge_candidates = num_warps * _TOPK
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        num_tokens: Int32,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.num_warps == 1):
            self.kernel_1w(
                gating_output, correction_bias, topk_weights, topk_ids, scale
            ).launch(
                grid=(num_tokens, 1, 1),
                block=[32, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:

            @cute.struct
            class Storage:
                merge_keys: cute.struct.MemRange[Uint32, self.num_merge_candidates]
                merge_ids: cute.struct.MemRange[Int32, self.num_merge_candidates]
                merge_scores: cute.struct.MemRange[Float32, self.num_merge_candidates]

            self.kernel_4w(
                gating_output, correction_bias, topk_weights, topk_ids, scale, Storage
            ).launch(
                grid=(num_tokens, 1, 1),
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.jit
    def _normalize_and_store(
        self,
        slot_weight: Float32,
        slot_id: Int32,
        scale: Float32,
        topk_weights: Int64,
        topk_ids: Int64,
        row: Int32,
        lane: Int32,
    ) -> None:
        total = slot_weight
        for shift in cutlass.range_constexpr(2, -1, -1):
            total = _add_f32(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))
        if lane < _TOPK:
            _store_f32(
                topk_weights + (Int64(row) * _TOPK + Int64(lane)) * 4,
                _normalized_weight(scale, total, slot_weight),
            )
            _store_s32(topk_ids + (Int64(row) * _TOPK + Int64(lane)) * 4, slot_id)

    @cute.kernel
    def kernel_1w(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        scale: Float32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        experts_per_thread = self.experts_per_thread

        expert_bias = cute.make_rmem_tensor(
            cute.make_layout(experts_per_thread), Float32
        )
        for j in cutlass.range_constexpr(experts_per_thread):
            expert_bias[j] = _load_f32_nc(correction_bias + Int64(j * 32 + lane) * 4)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        score = cute.make_rmem_tensor(cute.make_layout(experts_per_thread), Float32)
        key = cute.make_rmem_tensor(cute.make_layout(experts_per_thread), Uint32)
        row_base = gating_output + Int64(row) * (self.num_experts * 4)
        for j in cutlass.range_constexpr(experts_per_thread):
            logit = _load_f32_nc(row_base + Int64(j * 32 + lane) * 4)
            score[j], key[j] = _score_and_key(logit, expert_bias[j])

        best_key = Uint32(key[0])
        best_local_idx = Int32(0)
        for j in cutlass.range_constexpr(1, experts_per_thread):
            if Uint32(key[j]) > best_key:
                best_key = Uint32(key[j])
                best_local_idx = Int32(j)

        slot_weight = Float32(0.0)
        slot_id = Int32(0)
        for slot in cutlass.range_constexpr(_TOPK):
            top = Uint32(cute.arch.warp_redux_sync(best_key, "umax"))
            tied_expert_id = Uint32(0xFFFFFFFF)
            if best_key == top:
                tied_expert_id = Uint32(lane + 32 * best_local_idx)
            winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
            won = Float32(0.0)
            if winner == lane + 32 * best_local_idx:
                for j in cutlass.range_constexpr(experts_per_thread):
                    if best_local_idx == j:
                        won = score[j]
                        key[j] = Uint32(0)
                best_key = Uint32(key[0])
                best_local_idx = Int32(0)
                for j in cutlass.range_constexpr(1, experts_per_thread):
                    if Uint32(key[j]) > best_key:
                        best_key = Uint32(key[j])
                        best_local_idx = Int32(j)
            weight = cute.arch.shuffle_sync(won, winner & 31)
            if lane == slot:
                slot_weight = weight
                slot_id = winner

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        self._normalize_and_store(
            slot_weight, slot_id, scale, topk_weights, topk_ids, row, lane
        )

    @cute.kernel
    def kernel_4w(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        scale: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        experts_per_thread = self.experts_per_thread
        warp_base = warp * (32 * experts_per_thread)

        smem = _SmemAllocator()
        storage = smem.allocate(Storage)
        smem_keys = storage.merge_keys.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )
        smem_ids = storage.merge_ids.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )
        smem_scores = storage.merge_scores.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )

        expert_bias = cute.make_rmem_tensor(
            cute.make_layout(experts_per_thread), Float32
        )
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            expert_bias[j] = _load_f32_nc(correction_bias + Int64(e) * 4)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        score = cute.make_rmem_tensor(cute.make_layout(experts_per_thread), Float32)
        key = cute.make_rmem_tensor(cute.make_layout(experts_per_thread), Uint32)
        row_base = gating_output + Int64(row) * (self.num_experts * 4)
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            logit = _load_f32_nc(row_base + Int64(e) * 4)
            score[j], key[j] = _score_and_key(logit, expert_bias[j])

        best_key = Uint32(key[0])
        best_local_idx = Int32(0)
        for j in cutlass.range_constexpr(1, experts_per_thread):
            if Uint32(key[j]) > best_key:
                best_key = Uint32(key[j])
                best_local_idx = Int32(j)

        out_key = Uint32(0)
        out_id = Int32(-1)
        out_score = Float32(0.0)
        for slot in cutlass.range_constexpr(_TOPK):
            top = Uint32(cute.arch.warp_redux_sync(best_key, "umax"))
            tied_expert_id = Uint32(0xFFFFFFFF)
            if best_key == top:
                tied_expert_id = Uint32(warp_base + lane + 32 * best_local_idx)
            winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
            won = Float32(0.0)
            if winner - warp_base == lane + 32 * best_local_idx:
                for j in cutlass.range_constexpr(experts_per_thread):
                    if best_local_idx == j:
                        won = score[j]
                        key[j] = Uint32(0)
                best_key = Uint32(key[0])
                best_local_idx = Int32(0)
                for j in cutlass.range_constexpr(1, experts_per_thread):
                    if Uint32(key[j]) > best_key:
                        best_key = Uint32(key[j])
                        best_local_idx = Int32(j)
            weight = cute.arch.shuffle_sync(won, (winner - warp_base) & 31)
            if lane == slot:
                out_key = top
                out_id = winner
                out_score = weight

        if lane < _TOPK:
            idx = warp * _TOPK + lane
            smem_keys[idx] = out_key
            smem_ids[idx] = out_id
            smem_scores[idx] = out_score

        cute.arch.barrier()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if warp == 0:
            cand_key = Uint32(0)
            cand_id = Int32(0x7FFFFFFF)
            cand_score = Float32(0.0)
            if lane < self.num_merge_candidates:
                cand_key = smem_keys[lane]
                cand_id = smem_ids[lane]
                cand_score = smem_scores[lane]

            slot_weight = Float32(0.0)
            slot_id = Int32(0)
            for slot in cutlass.range_constexpr(_TOPK):
                top = Uint32(cute.arch.warp_redux_sync(cand_key, "umax"))
                tied_expert_id = Uint32(0xFFFFFFFF)
                if cand_key == top:
                    tied_expert_id = Uint32(cand_id)
                chosen = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                holders = Uint32(cute.arch.vote_ballot_sync(cand_id == chosen))
                first = Int32(
                    cute.arch.popc((holders & (Uint32(0) - holders)) - Uint32(1))
                )
                weight = cute.arch.shuffle_sync(cand_score, first)
                if cand_id == chosen:
                    cand_key = Uint32(0)
                if lane == slot:
                    slot_weight = weight
                    slot_id = chosen

            self._normalize_and_store(
                slot_weight, slot_id, scale, topk_weights, topk_ids, row, lane
            )


@cache
def _compile_stages(num_experts: int, num_warps: int, use_pdl: bool):
    fake = make_fake_stream()
    c_empty = cute.compile(EmptyKernel(num_warps * 32, use_pdl), Int32(1), fake)
    c_mem = cute.compile(
        MemOnlyKernel(num_experts, num_warps, use_pdl),
        *(Int64(0),) * 4,
        Int32(1),
        fake,
    )
    c_math = cute.compile(
        MathOnlyKernel(num_experts, num_warps, use_pdl),
        *(Int64(0),) * 4,
        Int32(1),
        fake,
    )
    c_full = cute.compile(
        FullTopkKernel(num_experts, num_warps, use_pdl),
        *(Int64(0),) * 4,
        Int32(1),
        Float32(1.0),
        fake,
    )
    return c_empty, c_mem, c_math, c_full


def _torch_ref(logits: torch.Tensor, bias: torch.Tensor, scale: float):
    scores = F.softplus(logits.float()).sqrt()
    biased = scores + bias.unsqueeze(0)
    biased = torch.where(torch.isnan(biased), torch.full_like(biased, -1e30), biased)
    ids = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :_TOPK]
    w = scores.gather(1, ids)
    denom = w.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (w * (scale / denom)).to(torch.float32), ids.to(torch.int32)


def _bench_us(
    fn,
    input_args: tuple,
    *,
    cold_l2: bool,
    warmup: int,
    iters: int,
) -> float:
    """Measure median per-launch latency in microseconds via CUDA Graph replay."""
    try:
        from flashinfer.testing import bench_gpu_time_with_cudagraph

        return (
            statistics.median(
                bench_gpu_time_with_cudagraph(
                    fn,
                    cold_l2_cache=cold_l2,
                    dry_run_iters=warmup,
                    repeat_iters=iters,
                    input_args=input_args,
                )
            )
            * 1000.0
        )
    except ImportError:
        pass

    logits, bias = input_args
    if cold_l2:
        row_bytes = (
            logits.numel() * logits.element_size() + bias.numel() * bias.element_size()
        )
        slots = max(2, min(512, (128 * 1024 * 1024) // max(row_bytes, 1) + 1))
        bufs = [(logits.clone(), bias.clone()) for _ in range(slots)]
    else:
        slots = 1
        bufs = [(logits, bias)]

    for _ in range(warmup):
        fn(*bufs[0])
    torch.cuda.synchronize()

    inner_reps = max(slots, 32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(inner_reps):
            fn(*bufs[i % slots])
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner_reps)
    return statistics.median(samples)


def verify_correctness(num_experts: int, use_pdl: bool) -> None:
    torch.manual_seed(0)
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    for num_warps, m in ((4, 33), (1, 1024)):
        logits = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
        logits[0] = 0.0
        logits[1] = torch.linspace(19.0, 25.0, num_experts, device="cuda")
        ow = torch.empty((m, _TOPK), dtype=torch.float32, device="cuda")
        oi = torch.empty((m, _TOPK), dtype=torch.int32, device="cuda")
        _, _, _, c_full = _compile_stages(num_experts, num_warps, use_pdl)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        c_full(
            Int64(logits.data_ptr()),
            Int64(bias.data_ptr()),
            Int64(ow.data_ptr()),
            Int64(oi.data_ptr()),
            Int32(m),
            Float32(2.5),
            stream,
        )
        ref_w, ref_i = _torch_ref(logits, bias, 2.5)
        torch.testing.assert_close(oi, ref_i, atol=0, rtol=0)
        torch.testing.assert_close(ow, ref_w, atol=2e-5, rtol=2e-5)
        max_err = (ow - ref_w).abs().max().item()
        print(
            f"Correctness PASS: E={num_experts}, warps={num_warps}, M={m} "
            f"(exact top-6 ids, max_abs_err={max_err:.2e})"
        )


def _bench_row(
    num_experts: int,
    m: int,
    bias: torch.Tensor,
    *,
    use_pdl: bool,
    cold_l2: bool,
    warmup: int,
    iters: int,
) -> dict:
    num_warps = 4 if m <= _multi_warp_max_tokens(bias.device.index) else 1
    c_empty, c_mem, c_math, c_full = _compile_stages(num_experts, num_warps, use_pdl)
    logits = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
    ow = torch.empty((m, _TOPK), dtype=torch.float32, device="cuda")
    oi = torch.empty((m, _TOPK), dtype=torch.int32, device="cuda")

    def _st():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    t_emp = _bench_us(
        lambda x, b: c_empty(Int32(m), _st()),
        (logits, bias),
        cold_l2=cold_l2,
        warmup=warmup,
        iters=iters,
    )
    t_mem = _bench_us(
        lambda x, b: c_mem(
            Int64(x.data_ptr()),
            Int64(b.data_ptr()),
            Int64(ow.data_ptr()),
            Int64(oi.data_ptr()),
            Int32(m),
            _st(),
        ),
        (logits, bias),
        cold_l2=cold_l2,
        warmup=warmup,
        iters=iters,
    )
    t_math = _bench_us(
        lambda x, b: c_math(
            Int64(x.data_ptr()),
            Int64(b.data_ptr()),
            Int64(ow.data_ptr()),
            Int64(oi.data_ptr()),
            Int32(m),
            _st(),
        ),
        (logits, bias),
        cold_l2=cold_l2,
        warmup=warmup,
        iters=iters,
    )
    t_full = _bench_us(
        lambda x, b: c_full(
            Int64(x.data_ptr()),
            Int64(b.data_ptr()),
            Int64(ow.data_ptr()),
            Int64(oi.data_ptr()),
            Int32(m),
            Float32(2.5),
            _st(),
        ),
        (logits, bias),
        cold_l2=cold_l2,
        warmup=warmup,
        iters=iters,
    )
    print(
        f"{m:>5} | {num_warps:>2} | {t_emp:>9.2f} | {t_mem:>8.2f} | "
        f"{t_math:>8.2f} | {t_full:>8.2f} | {t_mem - t_emp:>+8.2f} | "
        f"{t_math - t_mem:>+8.2f} | {t_full - t_math:>+9.2f}"
    )
    return {
        "cold_l2": cold_l2,
        "E": num_experts,
        "M": m,
        "num_warps": num_warps,
        "empty_us": round(t_emp, 3),
        "mem_only_us": round(t_mem, 3),
        "math_only_us": round(t_math, 3),
        "full_topk_us": round(t_full, 3),
        "delta_mem_us": round(t_mem - t_emp, 3),
        "delta_math_us": round(t_math - t_mem, 3),
        "delta_topk_us": round(t_full - t_math, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="4-stage CuTe-DSL latency ablation for DSv4 top-6 router."
    )
    ap.add_argument("--num-experts", type=int, default=384, choices=[256, 384])
    ap.add_argument("--ms", type=str, default="1,4,8,16,64,256,512,4096")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--no-pdl", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    major, minor = torch.cuda.get_device_capability()
    use_pdl = (not args.no_pdl) and (major >= 9)
    ms_list = [int(x) for x in args.ms.split(",") if x.strip()]
    dev_name = torch.cuda.get_device_name(0)
    threshold = _multi_warp_max_tokens(0)
    print(
        f"GPU: {dev_name} (sm_{major}{minor}) | E={args.num_experts} | "
        f"use_pdl={use_pdl} | multi_warp_max_tokens={threshold}"
    )

    verify_correctness(args.num_experts, use_pdl)

    all_rows = []
    for cold in (False, True):
        print(
            f"\n=== 4-Stage Breakdown (E={args.num_experts}, cold_l2={cold}) ===\n"
            f"{'M':>5} | {'W':>2} | {'Empty(us)':>9} | {'MemOnly':>8} | {'+Math':>8} | "
            f"{'FullTopK':>8} | {'Mem-Emp':>8} | {'Math-Mem':>8} | {'TopK-Math':>9}"
        )
        print("-" * 85)
        bias = torch.randn(args.num_experts, dtype=torch.float32, device="cuda")
        for m in ms_list:
            all_rows.append(
                _bench_row(
                    args.num_experts,
                    m,
                    bias,
                    use_pdl=use_pdl,
                    cold_l2=cold,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"gpu": dev_name, "rows": all_rows}, f, indent=2)
        print(f"\nSaved JSON to {args.json_out}")


if __name__ == "__main__":
    main()
