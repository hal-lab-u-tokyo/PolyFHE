#include <cstdint>

#include "hifive/kernel/device_context.hpp"

__forceinline__ __device__ void csub_q(uint64_t& operand,
                                       const uint64_t& modulus) {
    const uint64_t tmp = operand - modulus;
    operand = tmp + (tmp >> 63) * modulus;
}

[[nodiscard]] __inline__ __device__ uint64_t multiply_and_reduce_shoup_lazy(
    const uint64_t& operand1, const uint64_t& operand2,
    const uint64_t& operand2_shoup, const uint64_t& modulus) {
    const uint64_t hi = __umul64hi(operand1, operand2_shoup);
    return operand1 * operand2 - hi * modulus;
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
    // csub_q(x, mod2);
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
    csub_q(s, mod2);                 // [0, 2q)
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

__device__ void load_g2s_phase1(uint64_t* gmem, uint64_t* smem, size_t n,
                                size_t n1, size_t n2) {
    size_t batch_idx = blockIdx.x / n2;
    size_t tmp = n2 * threadIdx.x + blockIdx.x % n2;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n1 / 8;
        const size_t gmem_idx = batch_idx * n + tmp + n / 8 * i;
        smem[smem_idx] = gmem[gmem_idx];
    }
}

__device__ void store_s2g_phase1(uint64_t* gmem, uint64_t* smem, size_t n,
                                 size_t n1, size_t n2) {
    size_t batch_idx = blockIdx.x / n2;
    size_t tmp = n2 * threadIdx.x + blockIdx.x % n2;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n1 / 8;
        const size_t gmem_idx = batch_idx * n + tmp + n / 8 * i;
        gmem[gmem_idx] = smem[smem_idx];
    }
}

__device__ void load_g2s_phase1_blocked(uint64_t* gmem, uint64_t* smem,
                                        size_t n, size_t n1, size_t n2) {
    size_t batch_idx = threadIdx.x / (n1 / 8);
    size_t thread_idx = threadIdx.x % (n1 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = thread_idx + i * n1 / 8;
        const size_t gmem_idx = blockIdx.x + smem_idx * n2;
        smem[smem_idx + batch_idx * n1] = gmem[gmem_idx + batch_idx * n];
    }
}

__device__ void store_s2g_phase1_blocked(uint64_t* gmem, uint64_t* smem,
                                         size_t n, size_t n1, size_t n2) {
    size_t batch_idx = threadIdx.x / (n1 / 8);
    size_t thread_idx = threadIdx.x % (n1 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = thread_idx + i * n1 / 8;
        const size_t gmem_idx = blockIdx.x + smem_idx * n2;
        gmem[gmem_idx + batch_idx * n] = smem[smem_idx + batch_idx * n1];
    }
}

__device__ void load_g2s_phase2(uint64_t* gmem, uint64_t* smem, size_t n,
                                size_t n1, size_t n2) {
    const int batch_idx = blockIdx.x / n1;
    const int block_idx = blockIdx.x % n1;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n2 / 8;
        smem[smem_idx] = gmem[block_idx * n2 + smem_idx + batch_idx * n];
    }
}

__device__ void store_s2g_phase2(uint64_t* gmem, uint64_t* smem, size_t n,
                                 size_t n1, size_t n2) {
    const int batch_idx = blockIdx.x / n1;
    const int block_idx = blockIdx.x % n1;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n2 / 8;
        gmem[block_idx * n2 + smem_idx + batch_idx * n] = smem[smem_idx];
    }
}

__device__ void load_g2s_phase2_blocked(uint64_t* gmem, uint64_t* smem,
                                        size_t n, size_t n1, size_t n2) {
    size_t batch_idx = threadIdx.x / (n2 / 8);
    size_t thread_idx = threadIdx.x % (n2 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_offset = thread_idx + i * n2 / 8;
        const size_t gmem_offset = (blockIdx.x * n2) + smem_offset;
        smem[smem_offset + batch_idx * n2] = gmem[gmem_offset + batch_idx * n];
    }
}

__device__ void store_s2g_phase2_blocked(uint64_t* gmem, uint64_t* smem,
                                         size_t n, size_t n1, size_t n2) {
    size_t batch_idx = threadIdx.x / (n2 / 8);
    size_t thread_idx = threadIdx.x % (n2 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_offset = thread_idx + i * n2 / 8;
        const size_t gmem_offset = (blockIdx.x * n2) + smem_offset;
        gmem[gmem_offset + batch_idx * n] = smem[smem_offset + batch_idx * n2];
    }
}

__device__ void NTTPhase1(uint64_t* buffer, NTTParams* params,
                          const size_t batch_idx, const size_t thread_idx) {
    size_t group = params->n1 / 8;
    uint64_t samples[8];

    // base address
    const uint64_t* psi = params->roots_pow[batch_idx];
    const uint64_t* psi_shoup = params->roots_pow_shoup[batch_idx];
    uint64_t modulus = params->q[batch_idx];

    for (size_t j = 0; j < 8; j++) {
        samples[j] = buffer[thread_idx + group * j];
    }
    size_t tw_idx = 1;
    fntt8(samples, psi, psi_shoup, tw_idx, modulus);
    for (size_t j = 0; j < 8; j++) {
        buffer[thread_idx + group * j] = samples[j];
    }

    size_t remain_iters = 0;
    __syncthreads();
    for (size_t j = 8, k = group / 2; j < group + 1; j *= 8, k >>= 3) {
        size_t m_idx2 = thread_idx / (k / 4);
        size_t t_idx2 = thread_idx % (k / 4);
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[2 * m_idx2 * k + t_idx2 + (k / 4) * l];
        }
        size_t tw_idx2 = j * tw_idx + m_idx2;
        fntt8(samples, psi, psi_shoup, tw_idx2, modulus);
        for (size_t l = 0; l < 8; l++) {
            buffer[2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
        }
        if (j == group / 2)
            remain_iters = 1;
        if (j == group / 4)
            remain_iters = 2;
        __syncthreads();
    }

    if (group < 8)
        remain_iters = (group == 4) ? 2 : 1;
    for (size_t l = 0; l < 8; l++) {
        samples[l] = buffer[8 * thread_idx + l];
    }
    if (remain_iters == 1) {
        size_t tw_idx2 = 4 * group * tw_idx + 4 * thread_idx;
        ct_butterfly(samples[0], samples[1], psi[tw_idx2], psi_shoup[tw_idx2],
                     modulus);
        ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1],
                     psi_shoup[tw_idx2 + 1], modulus);
        ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2],
                     psi_shoup[tw_idx2 + 2], modulus);
        ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3],
                     psi_shoup[tw_idx2 + 3], modulus);
    } else if (remain_iters == 2) {
        size_t tw_idx2 = 2 * group * tw_idx + 2 * thread_idx;
        fntt4(samples, psi, psi_shoup, tw_idx2, modulus);
        fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus);
    }
    for (size_t l = 0; l < 8; l++) {
        buffer[8 * thread_idx + l] = samples[l];
    }

    __syncthreads();
}

__device__ void NTTPhase2(uint64_t* buffer, NTTParams* params, size_t batch_idx,
                          size_t thread_idx, size_t n_idx) {
    size_t n2 = params->n2;
    size_t group = params->n2 / 8;
    size_t set = thread_idx / group;
    // size of a block
    uint64_t samples[8];
    size_t t = n2 / 2;

    // tid'th block
    size_t m_idx = n_idx / (t / 4);
    size_t t_idx = n_idx % (t / 4);

    uint64_t modulus = params->q[batch_idx];
    const uint64_t* psi = params->roots_pow[batch_idx];
    const uint64_t* psi_shoup = params->roots_pow_shoup[batch_idx];
    for (size_t j = 0; j < 8; j++) {
        samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
    }
    size_t tw_idx = params->n1 + m_idx;
    fntt8(samples, psi, psi_shoup, tw_idx, modulus);
    for (size_t j = 0; j < 8; j++) {
        buffer[set * n2 + t_idx + t / 4 * j] = samples[j];
    }

    size_t tail = 0;
    __syncthreads();

    for (size_t j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
        size_t m_idx2 = t_idx / (k / 4);
        size_t t_idx2 = t_idx % (k / 4);
        for (size_t l = 0; l < 8; l++) {
            samples[l] =
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
        }
        size_t tw_idx2 = j * tw_idx + m_idx2;
        fntt8(samples, psi, psi_shoup, tw_idx2, modulus);
        for (size_t l = 0; l < 8; l++) {
            buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                samples[l];
        }
        if (j == t / 8)
            tail = 1;
        if (j == t / 16)
            tail = 2;
        __syncthreads();
    }

    for (size_t l = 0; l < 8; l++) {
        samples[l] = buffer[set * n2 + 8 * t_idx + l];
    }
    if (tail == 1) {
        size_t tw_idx2 = t * tw_idx + 4 * t_idx;
        ct_butterfly(samples[0], samples[1], psi[tw_idx2], psi_shoup[tw_idx2],
                     modulus);
        ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1],
                     psi_shoup[tw_idx2 + 1], modulus);
        ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2],
                     psi_shoup[tw_idx2 + 2], modulus);
        ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3],
                     psi_shoup[tw_idx2 + 3], modulus);
    } else if (tail == 2) {
        size_t tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
        fntt4(samples, psi, psi_shoup, tw_idx2, modulus);
        fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus);
    }
    for (size_t l = 0; l < 8; l++) {
        buffer[set * n2 + 8 * t_idx + l] = samples[l];
    }
    __syncthreads();

    uint64_t modulus2 = modulus << 1;
    // final reduction
    for (size_t j = 0; j < 8; j++) {
        samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
        csub_q(samples[j], modulus2);
        csub_q(samples[j], modulus);
    }
    for (size_t j = 0; j < 8; j++) {
        buffer[set * n2 + t_idx + t / 4 * j] = samples[j];
    }
}

__global__ void ntt_phase1_batched(uint64_t* inout, NTTParams* params) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < params->n2 * params->batch) {
        load_g2s_phase1(inout, buffer, params->N, params->n1, params->n2);

        NTTPhase1(buffer, params, blockIdx.x / params->n2, threadIdx.x);

        store_s2g_phase1(inout, buffer, params->N, params->n1, params->n2);
    }
}

__global__ void ntt_phase2_batched(uint64_t* inout, NTTParams* params) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < params->n1 * params->batch) {
        load_g2s_phase2(inout, buffer, params->N, params->n1, params->n2);

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        NTTPhase2(buffer, params, blockIdx.x / params->n1, threadIdx.x,
                  tid % (params->N / 8));

        store_s2g_phase2(inout, buffer, params->N, params->n1, params->n2);
    }
}

__global__ void ntt_phase1_batched_blocked(uint64_t* inout, NTTParams* params) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < params->n2 * params->batch) {
        size_t batch_idx = threadIdx.x / (params->n1 / 8);

        load_g2s_phase1_blocked(inout, buffer, params->N, params->n1,
                                params->n2);

        NTTPhase1(buffer + batch_idx * params->n1, params, batch_idx,
                  threadIdx.x % (params->n1 / 8));

        store_s2g_phase1_blocked(inout, buffer, params->N, params->n1,
                                 params->n2);
    }
}

__global__ void ntt_phase2_batched_blocked(uint64_t* inout, NTTParams* params) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < params->n1 * params->batch) {
        load_g2s_phase2_blocked(inout, buffer, params->N, params->n1,
                                params->n2);

        size_t batch_idx = threadIdx.x / (params->n2 / 8);
        size_t thread_idx = threadIdx.x % (params->n2 / 8);
        size_t n_idx = blockIdx.x * params->n2 / 8 + thread_idx;

        NTTPhase2(buffer + batch_idx * params->n2, params, batch_idx,
                  thread_idx, n_idx);

        store_s2g_phase2_blocked(inout, buffer, params->N, params->n1,
                                 params->n2);
    }
}