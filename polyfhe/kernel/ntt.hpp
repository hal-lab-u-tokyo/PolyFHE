#pragma once

#include <cstdint>

#include "polyfhe/kernel/device_context.hpp"

// GPU
extern "C" {

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
    const uint64_t hi = __umul64hi(y, tw_shoup);
    const uint64_t tw_y = y * tw - hi * mod;
    const uint64_t mod2 = 2 * mod;
    const uint64_t tmp = x - mod2;
    x = tmp + (tmp >> 63) * mod2;
    y = x + mod2 - tw_y;
    x += tw_y;
    /*
     uint64_t x_copy = x;
     uint64_t y_copy = y;
     y_copy = (tw * y_copy) % mod;
     x = (x_copy + y_copy) % mod;
     y = (x_copy + mod - y_copy) % mod;
     */
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

__device__ __forceinline__ void load_g2s_phase1(uint64_t* gmem, uint64_t* smem,
                                                size_t n, size_t n1,
                                                size_t n2) {
    size_t batch_idx = blockIdx.x / n2;
    size_t tmp = n2 * threadIdx.x + blockIdx.x % n2;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n1 / 8;
        const size_t gmem_idx = batch_idx * n + tmp + n / 8 * i;
        smem[smem_idx] = gmem[gmem_idx];
    }
}

__device__ __forceinline__ void store_s2g_phase1(uint64_t* gmem, uint64_t* smem,
                                                 size_t n, size_t n1,
                                                 size_t n2) {
    size_t batch_idx = blockIdx.x / n2;
    size_t tmp = n2 * threadIdx.x + blockIdx.x % n2;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n1 / 8;
        const size_t gmem_idx = batch_idx * n + tmp + n / 8 * i;
        gmem[gmem_idx] = smem[smem_idx];
    }
}

__device__ __forceinline__ void load_g2s_phase1_blocked(uint64_t* gmem,
                                                        uint64_t* smem,
                                                        size_t n, size_t n1,
                                                        size_t n2) {
    size_t batch_idx = threadIdx.x / (n1 / 8);
    size_t thread_idx = threadIdx.x % (n1 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = thread_idx + i * n1 / 8;
        const size_t gmem_idx = blockIdx.x + smem_idx * n2;
        smem[smem_idx + batch_idx * n1] = gmem[gmem_idx + batch_idx * n];
    }
}

__device__ __forceinline__ void store_s2g_phase1_blocked(uint64_t* gmem,
                                                         uint64_t* smem,
                                                         size_t n, size_t n1,
                                                         size_t n2) {
    size_t batch_idx = threadIdx.x / (n1 / 8);
    size_t thread_idx = threadIdx.x % (n1 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = thread_idx + i * n1 / 8;
        const size_t gmem_idx = blockIdx.x + smem_idx * n2;
        gmem[gmem_idx + batch_idx * n] = smem[smem_idx + batch_idx * n1];
    }
}

__device__ __forceinline__ void load_g2s_phase2(uint64_t* gmem, uint64_t* smem,
                                                size_t n, size_t n1,
                                                size_t n2) {
    const int batch_idx = blockIdx.x / n1;
    const int block_idx = blockIdx.x % n1;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n2 / 8;
        smem[smem_idx] = gmem[block_idx * n2 + smem_idx + batch_idx * n];
    }
}

__device__ __forceinline__ void store_s2g_phase2(uint64_t* gmem, uint64_t* smem,
                                                 size_t n, size_t n1,
                                                 size_t n2) {
    const int batch_idx = blockIdx.x / n1;
    const int block_idx = blockIdx.x % n1;
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_idx = threadIdx.x + i * n2 / 8;
        gmem[block_idx * n2 + smem_idx + batch_idx * n] = smem[smem_idx];
    }
}

__device__ __forceinline__ void load_g2s_phase2_blocked(uint64_t* gmem,
                                                        uint64_t* smem,
                                                        size_t n, size_t n1,
                                                        size_t n2) {
    size_t batch_idx = threadIdx.x / (n2 / 8);
    size_t thread_idx = threadIdx.x % (n2 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_offset = thread_idx + i * n2 / 8;
        const size_t gmem_offset = (blockIdx.x * n2) + smem_offset;
        smem[smem_offset + batch_idx * n2] = gmem[gmem_offset + batch_idx * n];
    }
}

__device__ __forceinline__ void store_s2g_phase2_blocked(uint64_t* gmem,
                                                         uint64_t* smem,
                                                         size_t n, size_t n1,
                                                         size_t n2) {
    size_t batch_idx = threadIdx.x / (n2 / 8);
    size_t thread_idx = threadIdx.x % (n2 / 8);
#pragma unroll
    for (size_t i = 0; i < 8; i++) {
        const size_t smem_offset = thread_idx + i * n2 / 8;
        const size_t gmem_offset = (blockIdx.x * n2) + smem_offset;
        gmem[gmem_offset + batch_idx * n] = smem[smem_offset + batch_idx * n2];
    }
}

__device__ __forceinline__ void NTTPhase1Op(uint64_t* buffer, NTTParams* params,
                                            size_t batch_idx) {
    bool debug = false;
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n1;
    for (int m = 1; m < params->n1; m *= 2) {
        t = t / 2;
        int j = threadIdx.x & (m - 1);
        int k = 2 * m * (threadIdx.x / m);
        const int rootidx = t * j * params->n2;
        uint64_t S = params->roots_pow[batch_idx][rootidx];
        __syncthreads();
        uint64_t U = buffer[k + j];
        uint64_t V_ = buffer[k + j + m];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        if (debug) {
            printf(
                "m:%d, block:%d, (a[%d],a[%d]) = (%ld,%ld), "
                "U=%ld,V=%ld,S=%ld,rootidx=%d\n",
                m, blockIdx.x, blockIdx.x * params->n1 + (k + j),
                blockIdx.x * params->n1 + (k + j + m), buffer[k + j],
                buffer[k + j + m], U, V_, S, rootidx);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void NTTPhase2Op(uint64_t* buffer, NTTParams* params,
                                            size_t batch_idx,
                                            size_t block_idx) {
    bool debug = false;
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n2;
    for (int m = 1; m < params->n2; m *= 2) {
        t = t / 2;
        int j = threadIdx.x & (m - 1);
        int k = 2 * m * (threadIdx.x / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow[batch_idx][rootidx];
        __syncthreads();
        uint64_t U = buffer[k + j];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t V_ = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        if (debug) {
            printf(
                "m:%d, block:%ld, (a[%ld],a[%ld]) = (%ld,%ld), "
                "U=%ld,V=%ld,S=%ld,rootidx=%d\n",
                m, block_idx, block_idx * params->n2 + (k + j),
                block_idx * params->n2 + (k + j + m), buffer[k + j],
                buffer[k + j + m], U, V_, S, rootidx);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void iNTTPhase2Op(uint64_t* buffer,
                                             NTTParams* params,
                                             const size_t batch_idx,
                                             size_t block_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    for (int m = params->n2 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n2 / step;
        int j = threadIdx.x & (m - 1);
        int k = 2 * m * (threadIdx.x / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        __syncthreads();
    }
}

__device__ __forceinline__ void iNTTPhase1Op(uint64_t* buffer,
                                             NTTParams* params,
                                             const size_t batch_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    for (int m = params->n1 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n1 / step;
        int j = threadIdx.x & (m - 1);
        int k = 2 * m * (threadIdx.x / m);
        const int rootidx = t * j * params->n2;

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        __syncthreads();
    }
    const uint64_t Ninv = params->N_inv[batch_idx];
    buffer[threadIdx.x] = (buffer[threadIdx.x] * Ninv) % q;
    buffer[threadIdx.x + blockDim.x] =
        (buffer[threadIdx.x + blockDim.x] * Ninv) % q;
}

} // extern "C"