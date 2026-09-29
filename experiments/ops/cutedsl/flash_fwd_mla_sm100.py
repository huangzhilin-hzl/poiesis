import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx):
    """Reshape a tensor to fold qhead_per_kvhead into the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        (seqlen_q, headdim, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...)
    For LSE tensors (head_idx=1):
        (seqlen_q, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...)
    """
    head_stride = T.stride[head_idx]
    shape_packed = (
        (qhead_per_kvhead, T.shape[0]),
        *[T.shape[i] for i in range(1, head_idx)],
        nheads_kv,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_packed = (
        (head_stride, T.stride[0]),
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(
        T.iterator, cute.make_layout(shape_packed, stride=stride_packed)
    )


class FlashAttentionMLAForwardSm100:
    def __init__(self):
        self.hdim = 64
        self.hdimv = 512
        self.num_hdimv_splits = 2

        self.cta_group_size = 2
        self.cta_tile_m = 64
        self.tile_n = 128
        self.cluster_tile_m = self.cta_group_size * self.cta_tile_m
        self.qhead_per_kvhead = 128
        self.nheads_kv = 1

        self.cta_group = cute.nvgpu.tcgen05.CtaGroup.TWO
        self.dtype_acc = cute.Float32
        self.major_mode_Q = cute.nvgpu.OperatorMajorMode.K
        self.major_mode_Qvi = cute.nvgpu.OperatorMajorMode.K
        self.major_mode_K = cute.nvgpu.OperatorMajorMode.K
        self.major_mode_Vi = cute.nvgpu.OperatorMajorMode.K
        self.major_mode_P = cute.nvgpu.OperatorMajorMode.K
        self.major_mode_Vti = cute.nvgpu.OperatorMajorMode.MN

        self.oprand_source_Q = cute.nvgpu.tcgen05.OperandSource.SMEM
        self.oprand_source_Qvi = cute.nvgpu.tcgen05.OperandSource.SMEM
        self.oprand_source_P = cute.nvgpu.tcgen05.OperandSource.SMEM

        self.mma_tiler_QK = (self.cluster_tile_m, self.tile_n, self.hdim)
        self.mma_tiler_QvV = (
            self.cluster_tile_m,
            self.tile_n,
            self.hdimv // self.num_hdimv_splits,
        )
        self.mma_tiler_PVt = (
            self.cluster_tile_m,
            self.hdimv // self.num_hdimv_splits,
            self.tile_n,
        )

        self.num_stages_Q = 1

    @cute.jit
    def __call__(self, mQ, mQv, mK, mV, mO, mIndexTopk):
        """
        Args:
            mQ: [b, s_q, h, d]
            mQv: [b, s_q, h, dv]
            mK: [b, s_k, h_k, d]
            mV: [b, s_k, h_k, dv]
            mO: [b, s_q, h, dv]
            mIndexTopk: [b, s_q, topk]
        """
        # hint
        new_stride = lambda mX: (
            *(
                cute.assume(s, divby=128 // mX.element_type.width)
                for s in mX.stride[:-1]
            ),
            mX.stride[-1],
        )

        mQ, mQv, mK, mV, mO = [
            cute.make_tensor(
                mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX))
            )
            for mX in (mQ, mQv, mK, mV, mO)
        ]

        # layout require
        # Q/O [s_q, d, h, b]
        # k/v [s_k, d, h_k, b]
        # vt [d, s_k, h_k, b]
        # topk [topk, s_q, b]
        QO_layout_transpose = [1, 3, 2, 0]
        mQ, mQv, mK, mV, mO = [
            cute.make_tensor(
                mX.iterator, layout=cute.select(mX.layout, mode=QO_layout_transpose)
            )
            for mX in (mQ, mQv, mK, mV, mO)
        ]

        vt_layout_transpose = [1, 0, 2, 3]
        mVt = cute.make_tensor(
            mV.iterator, layout=cute.select(mV.layout, mode=vt_layout_transpose)
        )

        topk_layout_transpose = [2, 1, 0]
        mIndexTopk = cute.make_tensor(
            mIndexTopk.iterator,
            layout=cute.select(mIndexTopk.layout, mode=topk_layout_transpose),
        )

        # pad_qheads, head_first [h, d, s, b]
        pad_qheads_layout_transpose = [2, 1, 0, 3]
        mQ_valid, mQv_valid, mO_valid = [
            cute.make_tensor(
                mX.iterator,
                layout=cute.select(mX.layout, mode=pad_qheads_layout_transpose),
            )
            for mX in (mQ, mQv, mO)
        ]

        # pack_gqa
        pack_layout = cute.make_layout(((self.qhead_per_kvhead,)))
        mQ, mQv, mO = [
            pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
            for mX in (mQ, mQv, mO)
        ]

        # mma specs
        _mma_specs = [
            (
                "tiled_mma_QK",
                mQ.element.dtype,
                self.major_mode_Q,
                self.major_mode_K,
                self.mma_tiler_QK,
                self.oprand_source_Q,
            ),
            (
                "tiled_mma_QvV",
                mQv.element.dtype,
                self.major_mode_Qvi,
                self.major_mode_Vi,
                self.mma_tiler_QvV,
                self.oprand_source_Qvi,
            ),
            (
                "tiled_mma_PVt",
                mV.element.dtype,
                self.major_mode_P,
                self.major_mode_Vti,
                self.mma_tiler_PVt,
                self.oprand_source_P,
            ),
        ]

        tiled_mma_QK, tiled_mma_QvV, tiled_mma_PVt = [
            sm100_utils.make_trivial_tiled_mma(
                dtype_a,
                dtype_a,
                major_a,
                major_b,
                self.dtype_acc,
                self.cta_group,
                mma_tiler[:2],
                oprand_src_a,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, oprand_src_a in _mma_specs
        ]


        # smem sepc
        _smem_layout_specs = [
            ("sQ_layout", sm100_utils.make_smem_layout_a, tiled_mma_QK, self.mma_tiler_QK,  mQ.element.dtype,  self.num_stages_Q),
            ("sQ_layout", sm100_utils.make_smem_layout_a, tiled_mma_QK, self.mma_tiler_QK,  mQ.element.dtype,  self.num_stages_Q),
        ]


    @cute.kernel
    def __kernel__(self):
        pass
