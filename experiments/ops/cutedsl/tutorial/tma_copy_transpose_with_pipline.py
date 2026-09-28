import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


class Sm100PiplineTransposeCopyKernel:
    def __init__(self):
        self.tiler = (128, 128)
        self.tile_m, self.tile_n = self.tiler

        self.num_trans_warps = 4  # Maximum number of transpose warps
        self.trans_warp_id = tuple(range(self.num_trans_warps))
        self.tma_load_warp_id = self.num_trans_warps
        self.threads_per_cta = 32 * len(
            (self.tma_store_warp_id, self.tma_load_warp_id, *self.trans_warp_id)
        )
        self.num_trans_threads = 32 * len(self.trans_warp_id)

        self.trans_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.trans_warp_id),
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _compute_stage(self):
        bytes_per_element = self.dtype.width // 8
        smem_a_bytes_per_satge = self.tile_m * self.tile_n * bytes_per_element
        smem_b_bytes_per_satge = self.tile_m * self.tile_n * bytes_per_element

        reserved_bytes = 1024

        max_stages = (self.smem_capacity - reserved_bytes) // (smem_a_bytes_per_satge + smem_b_bytes_per_satge)

        num_stages = max(2, min(max_stages, 8))

        return num_stages, num_stages

    def _compute_grid(self):

        


    @cute.jit
    def __call__(self, A_cute: cute.Tensor, B_cute: cute.Tensor, max_active_clusters: cutlass.Constexpr):
        self.dtype = A_cute.element_type

        self.tma_load_num_stages, self.tma_store_num_stages = self._compute_stage()

        tile_sched_params = utils.PersistentTileSchedulerParams(
            (cute.ceil_div((*A_cute.shape, 1), self.tiler), 1), (1, 1, 1)
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        

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
            self.tma_load_num_stages,
        )

        smem_layout_b = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(B_trans_cute).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            self.tma_store_num_stages,
        )

        @cute.struct
        class SharedStorage:
            load_barrier_storage: cute.struct.MemRange[cutlass.Int64, self.tma_load_num_stages]
            store_barrier_storage: cute.struct.MemRange[cutlass.Int64, self.tma_store_num_stages]
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
            tile_sched_params
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
        tile_sched_params
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


        load_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            1
        )
        load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_trans_threads
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            self.tma_load_num_stages,
            load_producer_group,
            load_consumer_group,
            self.num_tma_load_bytes,
            load_barrier_ptr,
            defer_sync=True
        )

        store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_trans_threads
        )

        store_pipeline = pipeline.PipelineTmaStore.create(
            self.tma_store_num_stages,
            store_producer_group
        )

        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()



        if warp_idx == self.tma_load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, 
                cute.arch.block_idx(), 
                cute.arch.grid_dim()
            )

            work_tile = tile_sched.initial_work_tile_info()
            load_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load_stages
            )

            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(smem_tensor_a, 0, 2),
                cute.group_modes(gA_tiled, 0, 2),
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                tAgA_cta = tAgA[(None, cur_tile_coord[0], cur_tile_coord[1])]

                cute.copy(tma_atom_a, tAgA_cta, tAsA[None, 0], 
                          tma_bar_ptr=load_pipeline.producer_get_barrier(load_producer_state))

                load_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            load_pipeline.producer_tail(load_producer_state)

            
            gA_tiled = cute.local_tile(
                tma_tensor_a, (self.tile_m, self.tile_n), (None, None)
            )

            

            

            with cute.arch.elect_one():
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
            cute.copy(tma_atom_b, tCsB[None, 0], tCgB_cta)


def run():
    torch.manual_seed(442)
    device = "cuda"
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)

    M, N = 4096, 8192

    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.empty(N, M, device=device, dtype=torch.float32)

    A_cute = from_dlpack(A, assumed_align=16)
    B_cute = from_dlpack(B, assumed_align=16)

    kerenl = Sm100PiplineTransposeCopyKernel()
    compiled = cute.compile(kerenl, A_cute, B_cute, max_active_clusters)

    compiled(A_cute, B_cute)
    # Report asynchronous kernel failures before starting the validation kernels.
    torch.cuda.synchronize()

    torch.testing.assert_close(A.T, B, rtol=0, atol=0)


if __name__ == "__main__":

    run()
