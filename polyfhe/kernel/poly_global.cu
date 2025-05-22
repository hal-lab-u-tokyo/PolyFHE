#include "polyfhe/kernel/ntt-phantom.hpp"
#include "polyfhe/kernel/polynomial.cuh"

__global__ void NTTPhase1_general(
    Params *params, int start_limb, int end_limb, int start_limb_original,
    int end_limb_original, int exclude_start, int exclude_end,
    uint64_t *edge_BConv_23_0_NTTPhase1_24_0,
    uint64_t *edge_NTTPhase1_24_0_NTTPhase2_25_0, const uint64_t *twiddles,
    const uint64_t *twiddles_shoup, const DModulus *modulus) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    uint64_t *in = edge_BConv_23_0_NTTPhase1_24_0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (params->N / 8 * (end_limb - start_limb));
         i += blockDim.x * gridDim.x) {
        const size_t n_twr = params->N / 8;
        const size_t n_idx = i % n_twr;
        const size_t twr_idx = i / n_twr + start_limb;
        const size_t group = params->n1 / 8;
        const size_t pad_tid = threadIdx.x % params->pad;
        const size_t pad_idx = threadIdx.x / params->pad;
        const size_t n_init = n_twr / group * pad_idx + pad_tid +
                              params->pad * (n_idx / (group * params->pad));
        uint64_t *out;
        // NTTPhase1_24
        if (twr_idx < exclude_end && twr_idx >= exclude_start) {
            continue;
        }

// Load to register
#pragma unroll
        for (int l = 0; l < 8; l++) {
            reg[l] = *(in + twr_idx * params->N + n_init + n_twr * l);
        }
        const uint64_t size_P = params->K;
        const uint64_t size_QP = params->KL;
        out = edge_NTTPhase1_24_0_NTTPhase2_25_0;
        size_t twr_idx2 =
            (twr_idx >= start_limb_original + end_limb_original - size_P
                 ? size_QP - (start_limb_original + end_limb_original - twr_idx)
                 : twr_idx);
        d_poly_fnwt_phase1(params, out, shared, reg, twiddles, twiddles_shoup,
                           modulus, twr_idx, twr_idx2, n_init, i);
    }
}

__global__ void NTTPhase1_general_part(Params *params, int start_limb,
                                       int end_limb, int start_limb_original,
                                       int end_limb_original, uint64_t *in,
                                       uint64_t *out, const uint64_t *twiddles,
                                       const uint64_t *twiddles_shoup,
                                       const DModulus *modulus) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (params->N / 8 * (end_limb - start_limb));
         i += blockDim.x * gridDim.x) {
        const size_t n_twr = params->N / 8;
        const size_t n_idx = i % n_twr;
        const size_t twr_idx = i / n_twr + start_limb;
        const size_t group = params->n1 / 8;
        const size_t pad_tid = threadIdx.x % params->pad;
        const size_t pad_idx = threadIdx.x / params->pad;
        const size_t n_init = n_twr / group * pad_idx + pad_tid +
                              params->pad * (n_idx / (group * params->pad));
        // NTTPhase1_24
// Load to register
#pragma unroll
        for (int l = 0; l < 8; l++) {
            reg[l] = *(in + twr_idx * params->N + n_init + n_twr * l);
        }
        size_t twr_idx2 =
            (twr_idx >= start_limb_original + end_limb_original - params->K
                 ? params->KL -
                       (start_limb_original + end_limb_original - twr_idx)
                 : twr_idx);
        d_poly_fnwt_phase1(params, out, shared, reg, twiddles, twiddles_shoup,
                           modulus, twr_idx, twr_idx2, n_init, i);
    }
}

__global__ void NTTP1_part_allbeta(Params *params, int start_limb, int end_limb,
                                   int start_limb_original,
                                   int end_limb_original,
                                   const uint64_t *twiddles,
                                   const uint64_t *twiddles_shoup,
                                   const DModulus *modulus,
                                   uint64_t **in_list) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (params->N / 8 * (end_limb - start_limb));
         i += blockDim.x * gridDim.x) {
        const size_t n_twr = params->N / 8;
        const size_t n_idx = i % n_twr;
        const size_t twr_idx = i / n_twr + start_limb;
        const size_t group = params->n1 / 8;
        const size_t pad_tid = threadIdx.x % params->pad;
        const size_t pad_idx = threadIdx.x / params->pad;
        const size_t n_init = n_twr / group * pad_idx + pad_tid +
                              params->pad * (n_idx / (group * params->pad));
        size_t twr_idx2 =
            (twr_idx >= start_limb_original + end_limb_original - params->K
                 ? params->KL -
                       (start_limb_original + end_limb_original - twr_idx)
                 : twr_idx);

        for (int beta_idx = 0; beta_idx < 5; beta_idx++) {
            if (twr_idx >= (beta_idx + 1) * 6 || twr_idx < beta_idx * 6) {
#pragma unroll
                for (int l = 0; l < 8; l++) {
                    reg[l] = *(in_list[beta_idx] + twr_idx * params->N +
                               n_init + n_twr * l);
                }
                d_poly_fnwt_phase1(params, in_list[beta_idx], shared, reg,
                                   twiddles, twiddles_shoup, modulus, twr_idx,
                                   twr_idx2, n_init, i);
            }
        }
    }
}

__global__ void NTTPhase2_general(
    Params *params, int start_limb, int end_limb, int start_limb_original,
    int end_limb_original, int exclude_start, int exclude_end,
    uint64_t *edge_NTTPhase1_24_0_NTTPhase2_25_0,
    uint64_t *edge_NTTPhase2_25_0_MultKeyAccum_8_0, const uint64_t *twiddles,
    const uint64_t *twiddles_shoup, const DModulus *modulus) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    const size_t n_tower = params->N / 8;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (params->N / 8 * (end_limb - start_limb));
         tid += blockDim.x * gridDim.x) {
        // NTTPhase2_25
        const uint64_t size_P = params->K;
        const uint64_t size_QP = params->KL;

        const size_t twr_idx = tid / n_tower + start_limb;
        size_t twr_idx2 =
            (twr_idx >= start_limb_original + end_limb_original - size_P
                 ? size_QP - (start_limb_original + end_limb_original - twr_idx)
                 : twr_idx);

        if (twr_idx < exclude_end && twr_idx >= exclude_start) {
            continue;
        }
        uint64_t n_init;
        d_poly_fnwt_phase2_debug(params, edge_NTTPhase1_24_0_NTTPhase2_25_0,
                                 shared, reg, twiddles, twiddles_shoup, modulus,
                                 end_limb, start_limb, twr_idx, twr_idx2,
                                 &n_init, tid);
        uint64_t *out_ptr =
            edge_NTTPhase2_25_0_MultKeyAccum_8_0 + twr_idx * params->N;
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(out_ptr + n_init + params->n2 / 8 * j) = reg[j];
        }
        __syncthreads();
    }
}

__global__ void NTTPhase2_general_part(Params *params, int start_limb,
                                       int end_limb, int start_limb_original,
                                       int end_limb_original, uint64_t *in,
                                       uint64_t *out, const uint64_t *twiddles,
                                       const uint64_t *twiddles_shoup,
                                       const DModulus *modulus) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    const size_t n_tower = params->N / 8;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (params->N / 8 * (end_limb - start_limb));
         tid += blockDim.x * gridDim.x) {
        // NTTPhase2_25
        const uint64_t size_P = params->K;
        const uint64_t size_QP = params->KL;

        const size_t twr_idx = tid / n_tower + start_limb;
        size_t twr_idx2 =
            (twr_idx >= start_limb_original + end_limb_original - size_P
                 ? size_QP - (start_limb_original + end_limb_original - twr_idx)
                 : twr_idx);

        uint64_t n_init;
        d_poly_fnwt_phase2_debug(params, in, shared, reg, twiddles,
                                 twiddles_shoup, modulus, end_limb, start_limb,
                                 twr_idx, twr_idx2, &n_init, tid);
        uint64_t *out_ptr = out + twr_idx * params->N;
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(out_ptr + n_init + params->n2 / 8 * j) = reg[j];
        }
        __syncthreads();
    }
}

__global__ void NTTP2_part_allbeta(Params *params, int start_limb, int end_limb,
                                   int start_limb_original,
                                   int end_limb_original,
                                   const uint64_t *twiddles,
                                   const uint64_t *twiddles_shoup,
                                   const DModulus *modulus,
                                   uint64_t **in_list) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (params->N / 8 * (end_limb - start_limb));
         tid += blockDim.x * gridDim.x) {
        const size_t l_idx = tid / (params->N / 8) + start_limb;
        size_t n_idx = tid % (params->N / 8);
        size_t m_idx = n_idx / (params->n2 / 8);
        size_t t_idx = n_idx % (params->n2 / 8);
        const size_t n_init = m_idx * params->n2 + t_idx;
        // NTTPhase2
        for (int beta_idx = 0; beta_idx < 5; beta_idx++) {
            size_t twr_idx2 =
                (l_idx >= start_limb_original + end_limb_original - params->K
                     ? params->KL -
                           (start_limb_original + end_limb_original - l_idx)
                     : l_idx);
            if (l_idx >= (beta_idx + 1) * 6 || l_idx < beta_idx * 6) {
                d_poly_fnwt_phase2_debug2(params, in_list[beta_idx], shared,
                                          reg, twiddles, twiddles_shoup,
                                          modulus, end_limb, start_limb, l_idx,
                                          twr_idx2, n_init, tid);
                uint64_t *out_ptr =
                    in_list[beta_idx] + l_idx * params->N + n_init;

#pragma unroll
                for (size_t j = 0; j < 8; j++) {
                    *(out_ptr + params->n2 / 8 * j) = reg[j];
                }
            }
        }
    }
}

// Define kernel for subgraph[18], type: ElemLimb2
__global__ void iNTTPhase2_general(Params *params, int start_limb, int end_limb,
                                   uint64_t *in, uint64_t *out) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    const size_t n_tower = params->N / 8;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (end_limb - start_limb) * n_tower;
         tid += blockDim.x * gridDim.x) {
        // Load data to register
        const int twr_idx = tid / params->N + start_limb;
        uint64_t *in_twr = in + twr_idx * params->N +
                           blockIdx.x * blockDim.x * 8 + threadIdx.x * 8;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            load_two_uint64_v2(in_twr + 2 * l, reg[2 * l], reg[2 * l + 1]);
            // reg[l] = *(in_twr + l);
        }
        __syncthreads();

        // iNTTPhase2_9
        d_poly_inwt_radix8_phase2(params, 30, shared, reg, tid);

        // Store data from register
        const size_t n_group = params->n2 / 8;
        const size_t idx_base =
            start_limb * params->N +
            blockIdx.x * blockDim.x * params->per_thread_ntt_size +
            (threadIdx.x / n_group) * n_group * params->per_thread_ntt_size +
            (threadIdx.x % n_group);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            *(out + idx_base + n_group * l) = reg[l];
        }
        __syncthreads();
    }
}

// Define kernel for subgraph[19], type: ElemLimb1
__global__ void iNTTPhase1_general(Params *params, int start_limb, int end_limb,
                                   uint64_t *in, uint64_t *out) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (params->N / 8 * (end_limb - start_limb));
         i += blockDim.x * gridDim.x) {
        const size_t n_twr = params->N / 8;
        const size_t n_idx = i % n_twr;
        const size_t twr_idx = i / n_twr + start_limb;
        const size_t group = params->n1 / 8;
        const size_t pad_tid = threadIdx.x % params->pad;
        const size_t pad_idx = threadIdx.x / params->pad;
        const size_t n_init = n_twr / group * pad_idx + pad_tid +
                              params->pad * (n_idx / (group * params->pad));
        // iNTTPhase1_10
        d_poly_inplace_inwt_radix8_phase1(in, params, start_limb, shared, reg,
                                          i);

        // Store data from register
        const size_t idx_out = twr_idx * params->N + n_init;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            *(out + idx_out + n_twr * l) = reg[l];
        }
        __syncthreads();
    }
}

__global__ void BConv_general_part_allbeta(
    Params *params, uint64_t **in_list, uint64_t **out_list,
    uint64_t **qiHat_mod_pj_list, uint64_t ibase_size, uint64_t obase_start,
    uint64_t obase_size, size_t alpha, size_t beta, const uint64_t *twiddles,
    const uint64_t *twiddles_shoup, const DModulus *modulus) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < beta * (params->N * obase_size) / 2;
         tid += blockDim.x * gridDim.x) {
        const size_t beta_idx = tid / (params->N * obase_size / 2);
        const size_t startPartIdx = alpha * beta_idx;
        if (startPartIdx == obase_start) {
            continue;
        }
        uint64_t obase_start_in30 =
            obase_start - obase_size * (obase_start > startPartIdx);
        const size_t tid_in_beta = tid % (params->N * obase_size / 2);
        const size_t n_idx = 2 * (tid_in_beta / obase_size);
        const size_t o_idx = tid_in_beta % obase_size;
        const size_t l_out_idx = o_idx + obase_start;
        // BConv_5
        BConvOpNoReg_debug2(
            params, out_list[beta_idx],
            in_list[beta_idx] + params->N * startPartIdx,
            qiHat_mod_pj_list[beta_idx] + obase_start_in30 * ibase_size, n_idx,
            o_idx, l_out_idx, ibase_size, twiddles, twiddles_shoup, modulus);
    }
}

// Define kernel for subgraph[17], type: Elem
__global__ void NTTP2_MultKeyAccum_part(
    Params *params, int start_limb, int end_limb, int start_limb_original,
    int end_limb_original, int beta,
    uint64_t *edge_NTTPhase1_24_0_NTTPhase2_25_0, const uint64_t *twiddles,
    const uint64_t *twiddles_shoup, const DModulus *modulus, uint64_t **in_list,
    uint64_t *edge_MultKeyAccum_8_0_iNTTPhase2_12_0,
    uint64_t *edge_MultKeyAccum_8_1_iNTTPhase2_9_0, uint64_t **relin_keys) {
    extern __shared__ uint64_t shared[];
    uint64_t reg[8];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (params->N / 8 * (end_limb - start_limb));
         tid += blockDim.x * gridDim.x) {
        const size_t l_idx = tid / (params->N / 8) + start_limb;
        size_t n_idx = tid % (params->N / 8);
        size_t m_idx = n_idx / (params->n2 / 8);
        size_t t_idx = n_idx % (params->n2 / 8);
        const size_t n_init = m_idx * params->n2 + t_idx;
        // NTTPhase2
        for (int beta_idx = 0; beta_idx < 5; beta_idx++) {
            size_t twr_idx2 =
                (l_idx >= start_limb_original + end_limb_original - params->K
                     ? params->KL -
                           (start_limb_original + end_limb_original - l_idx)
                     : l_idx);
            if (l_idx >= (beta_idx + 1) * 6 || l_idx < beta_idx * 6) {
                d_poly_fnwt_phase2_debug2(params, in_list[beta_idx], shared,
                                          reg, twiddles, twiddles_shoup,
                                          modulus, end_limb, start_limb, l_idx,
                                          twr_idx2, n_init, tid);
                uint64_t *out_ptr =
                    in_list[beta_idx] + l_idx * params->N + n_init;

#pragma unroll
                for (size_t j = 0; j < 8; j++) {
                    *(out_ptr + params->n2 / 8 * j) = reg[j];
                }
            }
        }
        __syncthreads();
        {
            size_t idx = l_idx * params->N + n_init;
#pragma unroll
            for (size_t j = 0; j < 8; j++) {
                MulKeyAccumOp_opt(params, edge_MultKeyAccum_8_0_iNTTPhase2_12_0,
                                  edge_MultKeyAccum_8_1_iNTTPhase2_9_0, in_list,
                                  relin_keys, beta, idx, l_idx, shared, reg, j);
                idx += params->n2 / 8;
            }
        }
    }
}

__global__ void MultKeyAccum_part(
    Params *params, int start_limb, int end_limb, int beta, uint64_t **in_list,
    uint64_t *edge_MultKeyAccum_8_0_iNTTPhase2_12_0,
    uint64_t *edge_MultKeyAccum_8_1_iNTTPhase2_9_0, uint64_t **relin_keys) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (params->N * (end_limb - start_limb));
         tid += blockDim.x * gridDim.x) {
        const size_t l_idx = tid / params->N + start_limb;
        MulKeyAccumOp_opt2(params, edge_MultKeyAccum_8_0_iNTTPhase2_12_0,
                           edge_MultKeyAccum_8_1_iNTTPhase2_9_0, in_list,
                           relin_keys, beta, tid + params->N * start_limb,
                           l_idx);
    }
}