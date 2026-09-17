import statistics

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


def mla_fwd():
    pass


def bench():
    kwargs = make_inputs(chunk=3)
    for _ in range(20):
        mla(**kwargs)
    torch.cuda.synchronize()

    for cold in (False, True):
        times_ms = bench_gpu_time(
            mla,
            input_kwargs=kwargs,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=cold,
            dry_run_iters=20,
            repeat_iters=100,
        )
        print(
            f"cold_l2={cold}: "
            f"mean={statistics.mean(times_ms) * 1000:.2f} us, "
            f"median={statistics.median(times_ms) * 1000:.2f} us"
        )


def make_inputs(
    chunk: int,
    rank: int = 0,
    workspace_bytes: int = 384 * 1024**2,
):
    assert 0 <= chunk < 4
    assert 0 <= rank < 8

    device = torch.device("cuda:0")
    batch, heads, topk, page_size = 4096, 64, 2048, 64
    context_len = (chunk + 1) * 32768

    causal_lens = chunk * 32768 + rank + np.arange(batch, dtype=np.int64) * 8 + 1
    valid_lens = np.minimum(causal_lens, topk).astype(np.int32)

    rng = np.random.default_rng(1234)
    slots = np.full((batch, topk), -1, dtype=np.int32)
    for i, causal_len in enumerate(causal_lens):
        n = int(valid_lens[i])
        slots[i, :n] = rng.choice(int(causal_len), n, replace=False)

    def random_fp8(shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )

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
    bench()
