import argparse

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


class Sm100SimpleCopyKernel:
    def __init__(self):
        self.tiler = (128, 128)
        self.tile_m, self.tile_n = self.tiler

    @cute.jit
    def __call__(self, A_cute: cute.Tensor, B_cute: cute.Tensor):
        self.dtype = A_cute.dtype

        grid = cute.ceil_div((*A_cute.shape, 1), self.tiler)
        block = (32, 1, 1)

        smem_layout = cute.make_layout(
            (self.tile_m, self.tile_n), stride=(self.tile_n, 1)
        )

        @cute.Struct
        class SharedStorage:
            barrier_storage: cute.struct.MemRange[cute.Int64, 1]
            smem_data: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)], 1024
            ]

        self.shared_storage = SharedStorage

        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), A_cute, smem_layout, self.tiler
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), B_cute, smem_layout, self.tiler
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout)

        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, smem_layout
        ).launch(grid=grid, block=block)

    @cute.kernel
    def kernel(self, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, smem_layout):
        bidx, bidy, _ = cute.block_idx()

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        barrier_ptr = storage.barrier_storage.data_ptr()

        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_ptr, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(
                barrier_ptr, self.num_tma_load_bytes
            )

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gA_tiled = cute.local_tile(
            tma_tensor_a, (self.tile_m, self.tile_n), (None, None)
        )
        gB_tiled = cute.local_tile(
            tma_tensor_b, (self.tile_m, self.tile_n), (None, None)
        )

        smem_tensor = storage.smem_data.get_tensor(smem_layout)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(gA_tiled, 0, 2),
        )
        _, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(gB_tiled, 0, 2),
        )

        tAgA_cta = tAgA[None, bidx, bidy]
        cute.copy(tma_atom_a, tAgA_cta, tAsA, tma_bar_ptr=barrier_ptr)

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier_ptr)

        cute.arch.mbarrier_wait(barrier_ptr, phase=0)

        tBgB_cta = tBgB[None, bidx, bidy]
        cute.copy(tma_atom_b, tAsA, tBgB_cta)

        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp.aysnc_bulk_wait_group(0)


def run():
    torch.manual_seed(442)
    device = "cuda"

    M, N = 4096, 8192

    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.empty(M, N, device=device, dtype=torch.float32)

    A_cute = from_dlpack(A)
    B_cute = from_dlpack(B)

    kerenl = Sm100SimpleCopyKernel()
    compiled = cute.compile(kerenl, A_cute, B_cute)

    compiled(A_cute, B_cute)

    torch.testing.assert_close(A, B)


if __name__ == "__main__":

    run()
