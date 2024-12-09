#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

struct NTTParams {
    int N;
    int logN;
    int N_inv;
    int batch;
    uint64_t q;
    uint64_t mu;
    uint64_t m;
    uint64_t root;
    uint64_t root_inv;
    uint64_t *roots_pow;
    uint64_t *roots_pow_shoup;
    uint64_t *roots_pow_inv;
    uint64_t *roots_pow_inv_shoup;
    int n1; // N = n1 * n2
    int n2;
};

extern "C" {
__device__ void NTTPhase1(DeviceContext *dc, const int batch, uint64_t *buffer,
                          size_t thread_idx);
__device__ void NTTPhase2(DeviceContext *dc, const int batch, uint64_t *buffer,
                          size_t thread_idx, size_t n_idx);

__device__ void load_g2s_phase1(uint64_t *gmem, uint64_t *smem, size_t n,
                                size_t n1, size_t n2);
__device__ void store_s2g_phase1(uint64_t *gmem, uint64_t *smem, size_t n,
                                 size_t n1, size_t n2);
__device__ void load_g2s_phase1_blocked(uint64_t *gmem, uint64_t *smem,
                                        size_t n, size_t n1, size_t n2);
__device__ void store_s2g_phase1_blocked(uint64_t *gmem, uint64_t *smem,
                                         size_t n, size_t n1, size_t n2);
__device__ void load_g2s_phase2(uint64_t *gmem, uint64_t *smem, size_t n,
                                size_t n1, size_t n2);
__device__ void store_s2g_phase2(uint64_t *gmem, uint64_t *smem, size_t n,
                                 size_t n1, size_t n2);
__device__ void load_g2s_phase2_blocked(uint64_t *gmem, uint64_t *smem,
                                        size_t n, size_t n1, size_t n2);
__device__ void store_s2g_phase2_blocked(uint64_t *gmem, uint64_t *smem,
                                         size_t n, size_t n1, size_t n2);
}