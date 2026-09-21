import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


class Sm100TransposeCopyKernel:
    def __init__(self):
        self.tiler = (128, 128)
        self.tile_m, self.tile_n = self.tiler

        self.num_trans_warps = 4  # Maximum number of transpose warps
        self.trans_warp_id = tuple(range(self.num_trans_warps))
        self.tma_load_warp_id = self.num_trans_warps
        self.tma_store_warp_id = self.num_trans_warps + 1
        self.threads_per_cta = 32 * len(
            (self.tma_store_warp_id, self.tma_load_warp_id, *self.trans_warp_id)
        )
        self.num_trans_threads = 32 * len(self.trans_warp_id)

        self.trans_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.trans_warp_id),
        )

    @cute.jit
    def __call__(self, A_cute: cute.Tensor, B_cute: cute.Tensor):
        self.dtype = A_cute.element_type

        grid = cute.ceil_div((*A_cute.shape, 1), self.tiler)
        block = (self.threads_per_cta, 1, 1)

        B_trans_cute = cute.make_tensor(
            B_cute.iterator,
            cute.make_layout(
                (B_cute.shape[1], B_cute.shape[0]),
                stride=(B_cute.stride[1], B_cute.stride[0]),
            ),
        )

        smem_layout_a = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(A_cute).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,
        )

        smem_layout_b = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(B_trans_cute).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,
        )

        @cute.struct
        class SharedStorage:
            load_barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]
            store_barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]
            smem_data_a: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_a)], 128
            ]
            smem_data_b: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_b)], 128
            ]

        self.shared_storage = SharedStorage

        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), A_cute, smem_layout_a, self.tiler
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), B_trans_cute, smem_layout_b, self.tiler
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout_a)

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            smem_layout_a,
            smem_layout_b,
        ).launch(grid=grid, block=block)

    @cute.kernel
    def kernel(
        self,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        smem_layout_a,
        smem_layout_b,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_barrier_ptr = storage.load_barrier_storage.data_ptr()
        store_barrier_ptr = storage.store_barrier_storage.data_ptr()

        smem_tensor_a = storage.smem_data_a.get_tensor(
            smem_layout_a.outer, smem_layout_a.inner
        )
        smem_tensor_b = storage.smem_data_b.get_tensor(
            smem_layout_b.outer, smem_layout_b.inner
        )

        if tidx == 0:
            cute.arch.mbarrier_init(load_barrier_ptr, 1)
            cute.arch.mbarrier_init(store_barrier_ptr, len(self.trans_warp_id))

            cute.arch.mbarrier_expect_tx(load_barrier_ptr, self.num_tma_load_bytes)

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx == self.tma_load_warp_id:

            gA_tiled = cute.local_tile(
                tma_tensor_a, (self.tile_m, self.tile_n), (None, None)
            )

            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(smem_tensor_a, 0, 2),
                cute.group_modes(gA_tiled, 0, 2),
            )

            tAgA_cta = tAgA[None, bidx, bidy]
            cute.copy(tma_atom_a, tAgA_cta, tAsA, tma_bar_ptr=load_barrier_ptr)

            with cute.elect_one():
                cute.arch.mbarrier_arrive(load_barrier_ptr)

        if warp_idx < self.tma_load_warp_id:
            trans_tid = tidx % self.num_trans_threads
            cute.arch.mbarrier_wait(load_barrier_ptr, 0)

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=self.dtype.width,
            )

            thread_layout = cute.make_layout(
                (self.num_trans_threads, 1), stride=(1, self.num_trans_threads)
            )
            value_layout = cute.make_layout(1)

            tiled_copy = cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)
            thr_copy = tiled_copy.get_slice(trans_tid)

            tCsA = thr_copy.partition_S(smem_tensor_a)

            tCrA = cute.make_fragment_like(tCsA)
            cute.copy(tiled_copy, tCsA, tCrA)

            tCsB = thr_copy.partition_D(smem_tensor_b)
            cute.copy(tiled_copy, tCrA, tCsB)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            self.trans_sync_barrier.arrive_and_wait()

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(store_barrier_ptr)

        if warp_idx == self.tma_store_warp_id:
            cute.arch.mbarrier_wait(store_barrier_ptr, 0)

            gB_tiled = cute.local_tile(
                tma_tensor_b, (self.tile_m, self.tile_n), (None, None)
            )

            tCsB, tCgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(smem_tensor_b, 0, 2),
                cute.group_modes(gB_tiled, 0, 2),
            )

            tCgB_cta = tCgB[None, bidx, bidy]
            cute.copy(tma_atom_b, tCsB, tCgB_cta)


def run():
    torch.manual_seed(442)
    device = "cuda"

    M, N = 4096, 8192

    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.empty(M, N, device=device, dtype=torch.float32)

    A_cute = from_dlpack(A, assumed_align=16)
    B_cute = from_dlpack(B, assumed_align=16)

    kerenl = Sm100SimpleCopyKernel()
    compiled = cute.compile(kerenl, A_cute, B_cute)

    compiled(A_cute, B_cute)
    # Report asynchronous kernel failures before starting the validation kernels.
    torch.cuda.synchronize()

    torch.testing.assert_close(A, B, rtol=0, atol=0)


if __name__ == "__main__":

    run()
