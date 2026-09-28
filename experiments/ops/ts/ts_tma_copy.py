import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
import cutlass
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import (
    PipelineConfig
)


class InputGemmResource():
    pass

@cute.kernel
def tma_copy_kernel(tma_desc_a: cutlass.GridConstant[cuda.TensorMap], tma_desc_b: cutlass.GridConstant[cuda.TensorMap]):

    with prims.elect_sync():
        prims.prefetch_tensormap(tma_desc_a.get_ptr())
        prims.prefetch_tensormap(tma_desc_b.get_ptr())

    PipelineConfig.create_tma_async_pipeline_config(
        num_stages=3,
        num_bytes=128*128*4,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
    )



    

@cute.jit
def tma_copy_kernel_host(A_cute, B_cute):
    grid = cute.ceil_div((*A_cute.shape, 1), (128, 128))
    block = (32, 1, 1)

    tma_desc_a = cuda.create_tensor_map_tiled_from_view(
        A_cute,
        box_dims=(128, 128),
        swizzle=cuda.TensorMapSwizzle.none
    )
    tma_desc_b = cuda.create_tensor_map_tiled_from_view(
            B_cute,
            box_dims=(128, 128),
            swizzle=cuda.TensorMapSwizzle.none
    )

    tma_copy_kernel(tma_desc_a, tma_desc_b).launch(
        grid=grid,
        block=block
    )


def run_ts_tma_copy_kernel(A_cute, B_cute):
    func = cute.compile[cute.FrontendNext](tma_copy_kernel_host, A_cute, B_cute)
    func(A_cute, B_cute)

if __name__ == "__main__":
    torch.manual_seed(42)
    M, N = 4096, 8192
    device = "cuda"
    A = torch.randn(M, N, device, dtype=torch.float32)
    B = torch.empty(M, N, device, dtype=torch.float32)

    A_cute = from_dlpack(A)
    B_cute = from_dlpack(B)

    run_ts_tma_copy_kernel(A_cute, B_cute)

    torch.testing.assert_close(A, B)