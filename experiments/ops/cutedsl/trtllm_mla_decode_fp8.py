"""
ref: https://github.com/NVIDIA/TensorRT-LLM/pull/15138

Kernel Architecture
    Input:
        q_latent: [num_head, latent_dim, seq_len_q, batch_size] FP8
        q_rope: [num_head, rope_dim, seq_len_q, batch_size] FP8
        c_latent: [page_size, latent_dim, batch_size * num_pages] FP8
        c_rope: [page_size, latent_dim, batch_size * num_pages] FP8

    Output:
        O: [num_head, latent_dim, seq_len_q, batch_size] FP8
        lse: [num_head, seq_len_q, batch_size] FP32 

    key:

        mma:
            mma_qk_tiler = [128, 128, 128]
            mma_qk_rope_tiler = [128, 128, 64]
            mma_pv_tile = [128, 256, 64？]
            max_fold: ?
            2cta_instrs
        
        12 warp role:
            compute_warp: 0-3
            correction_warp: 4-7
            mma_warp: 8
            load_tma_k_warp: 9
            load_tma_v_warp: 10
            empty_warp: 11
        
        pipline:
            load_q_stage: 1
            load_k_stage: 3
            load_v_stage: 2
            mma_s_stage: 2
            p_mma_stage: 2
            p_cor_stage: 2
            mma_o_stage: 2
        Smem:
            smem_q_latent: [128, 128, load_q_stage * (128 / 32)]


        Schedule:
    
    Launch:

    impl:
        c_latent_transpose = [latent_dim, page_size, batch_size * num_pages] 


        

    


    

"""

import torch

def torch_reference_mla(
        q_latent,
        q_rope,
        c_latent,
        c_rope
):
    q_ref = torch.cat(q_latent, q_latent, dim=1).permute(3, 2, 0, 1)

