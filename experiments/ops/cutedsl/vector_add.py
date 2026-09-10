import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.arch.constants import (
    WARP_SIZE,
)
from cutlass.cute.runtime import from_dlpack

THREAD_TILE_SIZE = 4


@cute.jit
def vector_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    vector_add_kernel(A, B, C).launch(grid=(A.shape[0] // 512, 1, 1), block=(128, 1, 1))


@cute.kernel
def vector_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tile_idx = bidx * bdim + tidx

    # if tidx == 0:
    #    cute.printf("[bid{}-tid{}] tile_idx={}\n", bidx, tidx, tile_idx)
    gA = cute.local_tile(A, (THREAD_TILE_SIZE,), (tile_idx,))
    gB = cute.local_tile(B, (THREAD_TILE_SIZE,), (tile_idx,))
    gC = cute.local_tile(C, (THREAD_TILE_SIZE,), (tile_idx,))

    rA = cute.make_rmem_tensor((THREAD_TILE_SIZE,), cutlass.Float32)
    rB = cute.make_rmem_tensor((THREAD_TILE_SIZE,), cutlass.Float32)

    cute.autovec_copy(gA, rA)
    cute.autovec_copy(gB, rB)

    rA.store(rA.load() + rB.load())

    cute.autovec_copy(rA, gC)


def test():
    total_size = 8192
    a = torch.randn(total_size, device="cuda")
    b = torch.randn(total_size, device="cuda")
    c = torch.zeros(total_size, device="cuda")

    a_cute = from_dlpack(a, assumed_align=16)
    b_cute = from_dlpack(b, assumed_align=16)
    c_cute = from_dlpack(c, assumed_align=16)

    compiled = cute.compile(vector_add, a_cute, b_cute, c_cute)

    compiled(a_cute, b_cute, c_cute)

    torch.testing.assert_close(c, a + b)


if __name__ == "__main__":
    test()
