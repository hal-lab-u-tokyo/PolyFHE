#pragma once

#include "polyfhe/kernel/device_context.hpp"

/**
 * \brief a = a * b % mod, Shoup's implementation
 * \param operand1 a, range [0, 2 * mod)
 * \param operand2 b, range [0, 2 * mod)
 * \param operand2_shoup shoup pre-computation of b
 * \param modulus mod
 * \return a * b % mod, range [0, 2 * mod)
 */
[[nodiscard]] __inline__ __device__ uint64_t multiply_and_reduce_shoup_lazy(
    const uint64_t& operand1, const uint64_t& operand2,
    const uint64_t& operand2_shoup, const uint64_t& modulus) {
    const uint64_t hi = __umul64hi(operand1, operand2_shoup);
    return operand1 * operand2 - hi * modulus;
}

__forceinline__ __device__ void modsub(uint64_t& operand,
                                       const uint64_t& modulus) {
    const uint64_t tmp = operand - modulus;
    operand = tmp + (tmp >> 63) * modulus;
}

/** Computer one butterfly in forward NTT
 * x[0] = x[0] + pow * x[1] % mod
 * x[1] = x[0] - pow * x[1] % mod
 */
__device__ __forceinline__ void ct_butterfly(uint64_t& x, uint64_t& y,
                                             const uint64_t& tw,
                                             const uint64_t& tw_shoup,
                                             const uint64_t& mod) {
    // const uint64_t tw_y = multiply_and_reduce_shoup_lazy(y, tw, tw_shoup,
    // mod);
    const uint64_t hi = __umul64hi(y, tw_shoup);
    const uint64_t tw_y = y * tw - hi * mod;
    // modsub(x, mod2);
    const uint64_t mod2 = 2 * mod;
    const uint64_t tmp = x - mod2;
    x = tmp + (tmp >> 63) * mod2;
    y = x + mod2 - tw_y;
    x += tw_y;
}

/** Computer one butterfly in inverse NTT
 * x[0] = (x[0] + pow * x[1]) / 2 % mod
 * x[1] = (x[0] - pow * x[1]) / 2 % mod
 */
__device__ __forceinline__ void gs_butterfly(uint64_t& x, uint64_t& y,
                                             const uint64_t& tw,
                                             const uint64_t& tw_shoup,
                                             const uint64_t& mod) {
    const uint64_t mod2 = 2 * mod;
    const uint64_t t = x + mod2 - y; // [0, 4q)
    uint64_t s = x + y;              // [0, 4q)
    modsub(s, mod2);                 // [0, 2q)
    x = s;
    y = multiply_and_reduce_shoup_lazy(t, tw, tw_shoup, mod);
}

__device__ __forceinline__ void fntt8(uint64_t* s, const uint64_t* tw,
                                      const uint64_t* tw_shoup, uint64_t tw_idx,
                                      uint64_t mod) {
    // stage 1
    ct_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
    ct_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
    ct_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
    ct_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
    // stage 2
    ct_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    ct_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    ct_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    ct_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    // stage 3
    ct_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
    ct_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
    ct_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
    ct_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);
}

__device__ __forceinline__ void fntt4(uint64_t* s, const uint64_t* tw,
                                      const uint64_t* tw_shoup, uint64_t tw_idx,
                                      uint64_t mod) {
    // stage 1
    ct_butterfly(s[0], s[2], tw[tw_idx], tw_shoup[tw_idx], mod);
    ct_butterfly(s[1], s[3], tw[tw_idx], tw_shoup[tw_idx], mod);
    // stage 2
    ct_butterfly(s[0], s[1], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    ct_butterfly(s[2], s[3], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
}

__device__ __forceinline__ void intt8(uint64_t* s, const uint64_t* tw,
                                      const uint64_t* tw_shoup, uint64_t tw_idx,
                                      uint64_t mod) {
    // stage 1
    gs_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
    gs_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
    gs_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
    gs_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);

    // stage 2
    gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    gs_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    gs_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);

    // stage 3
    gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
    gs_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
    gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
    gs_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
}

__device__ __forceinline__ void intt4(uint64_t* s, const uint64_t* tw,
                                      const uint64_t* tw_shoup, uint64_t tw_idx,
                                      uint64_t mod) {
    // stage 1
    gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    // stage 2
    gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
    gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
}

__device__ static void d_poly_inwt_radix8_phase2(Params* params,
                                                 const size_t coeff_mod_size,
                                                 const size_t start_mod_idx,
                                                 uint64_t* buffer,
                                                 uint64_t* samples, size_t i) {
    const size_t n = params->N;
    const size_t n1 = params->n1;
    const size_t n2 = params->n2;
    const uint64_t* itwiddles = params->itwiddle;
    const uint64_t* itwiddles_shoup = params->itwiddle_shoup;
    const uint64_t* modulus_table = params->qVec;
    size_t group = n2 / 8;
    size_t set = threadIdx.x / group;
    // size of a block
    size_t t = n / 2 / n1;
    // prime idx
    size_t twr_idx = i / (n / 8) + start_mod_idx;
    // index in N/2 range
    size_t n_idx = i % (n / 8);
    // i'th block
    size_t m_idx = n_idx / group;
    size_t t_idx = n_idx % group;
    // base address
    // auto in_ptr = in + twr_idx * n;
    const uint64_t* psi = itwiddles + n * twr_idx;
    const uint64_t* psi_shoup = itwiddles_shoup + n * twr_idx;
    uint64_t modulus = modulus_table[twr_idx];

    size_t tw_idx = n1 + m_idx;
    size_t tw_idx2 = (t / 4) * tw_idx + t_idx;
    intt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
    for (size_t l = 0; l < 8; l++) {
        buffer[set * n2 + 8 * t_idx + l] = samples[l];
    }
    size_t tail = 0;
    __syncthreads();

#pragma unroll
    for (size_t j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
        size_t m_idx2 = t_idx / (k / 4);
        size_t t_idx2 = t_idx % (k / 4);
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] =
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
        }
        tw_idx2 = j * tw_idx + m_idx2;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                samples[l];
        }
        if (j == 2)
            tail = 1;
        if (j == 4)
            tail = 2;
        __syncthreads();
    }

#pragma unroll
    for (size_t j = 0; j < 8; j++) {
        samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
    }
    if (tail == 1) {
        gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx],
                     modulus);
        gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx],
                     modulus);
        gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx],
                     modulus);
        gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx],
                     modulus);
    } else if (tail == 2) {
        intt4(samples, psi, psi_shoup, tw_idx, modulus);
        intt4(samples + 1, psi, psi_shoup, tw_idx, modulus);
    }
}

__device__ static void d_poly_inplace_inwt_radix8_phase1(
    uint64_t* in, Params* params, const size_t coeff_mod_size,
    const size_t start_mod_idx, uint64_t* buffer, uint64_t* samples, size_t i) {
    const uint64_t* itwiddles = params->itwiddle;
    const uint64_t* itwiddles_shoup = params->itwiddle_shoup;
    const uint64_t* modulus_table = params->qVec;
    const uint64_t* inv_degree_modulo = params->n_inv;
    const uint64_t* inv_degree_modulo_shoup = params->n_inv_shoup;
    const size_t n = params->N;
    const size_t n1 = params->n1;
    const size_t pad = params->pad;
    // pad address
    size_t pad_tid = threadIdx.x % pad;
    size_t pad_idx = threadIdx.x / pad;

    size_t group = n1 / 8;
    // size of a block
    // uint64_t samples[8];
    size_t t = n / 2;
    // prime idx
    size_t twr_idx = i / (n / 8) + start_mod_idx;
    // index in N/2 range
    size_t n_idx = i % (n / 8);

    // base address
    uint64_t* data_ptr = in + twr_idx * n;
    const uint64_t* psi = itwiddles + n * twr_idx;
    const uint64_t* psi_shoup = itwiddles_shoup + n * twr_idx;
    uint64_t modulus_value = modulus_table[twr_idx];
    uint64_t inv_degree_mod = inv_degree_modulo[twr_idx];
    uint64_t inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx];

    size_t n_init =
        2 * t / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
    for (size_t j = 0; j < 8; j++) {
        samples[j] = *(data_ptr + n_init + t / 4 / group * j);
    }
    size_t tw_idx = 1;
    size_t tw_idx2 = group * tw_idx + pad_idx;
    intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
    for (size_t j = 0; j < 8; j++) {
        buffer[pad_tid * (n1 + pad) + 8 * pad_idx + j] = samples[j];
    }
    size_t tail = 0;
    __syncthreads();

#pragma unroll
    for (size_t j = group / 8, k = 32; j > 0; j >>= 3, k *= 8) {
        size_t m_idx2 = pad_idx / (k / 4);
        size_t t_idx2 = pad_idx % (k / 4);
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                                (k / 4) * l];
        }
        size_t tw_idx2 = j * tw_idx + m_idx2;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                   (k / 4) * l] = samples[l];
        }
        if (j == 2)
            tail = 1;
        if (j == 4)
            tail = 2;
        __syncthreads();
    }
    if (group < 8)
        tail = (group == 4) ? 2 : 1;
#pragma unroll
    for (size_t l = 0; l < 8; l++) {
        samples[l] = buffer[pad_tid * (n1 + pad) + pad_idx + group * l];
    }
    if (tail == 1) {
        gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx],
                     modulus_value);
        gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx],
                     modulus_value);
        gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx],
                     modulus_value);
        gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx],
                     modulus_value);
    } else if (tail == 2) {
        intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
        intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
    }

    for (size_t j = 0; j < 4; j++) {
        samples[j] = multiply_and_reduce_shoup_lazy(
            samples[j], inv_degree_mod, inv_degree_mod_shoup, modulus_value);
    }

    n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
    for (size_t j = 0; j < 8; j++) {
        modsub(samples[j], modulus_value);
    }
}