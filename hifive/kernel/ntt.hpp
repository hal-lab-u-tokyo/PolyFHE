#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

// GPU
extern "C" {
__device__ void NTTPhase1Op(uint64_t *buffer, NTTParams *params,
                            const size_t batch_idx, const size_t thread_idx);
__device__ void NTTPhase2Op(uint64_t *buffer, NTTParams *params,
                            size_t batch_idx, size_t thread_idx, size_t n_idx);

__device__ void load_g2s_phase1(uint64_t *gmem, uint64_t *smem, size_t n,
                                size_t n1, size_t n2);
__device__ void store_s2g_phase1(uint64_t *gmem, uint64_t *smem, size_t n,
                                 size_t n1, size_t n2);
__device__ void load_g2s_phase2(uint64_t *gmem, uint64_t *smem, size_t n,
                                size_t n1, size_t n2);
__device__ void store_s2g_phase2(uint64_t *gmem, uint64_t *smem, size_t n,
                                 size_t n1, size_t n2);

__device__ void load_g2s_phase1_blocked(uint64_t *gmem, uint64_t *smem,
                                        size_t n, size_t n1, size_t n2);
__device__ void store_s2g_phase1_blocked(uint64_t *gmem, uint64_t *smem,
                                         size_t n, size_t n1, size_t n2);
__device__ void load_g2s_phase2_blocked(uint64_t *gmem, uint64_t *smem,
                                        size_t n, size_t n1, size_t n2);
__device__ void store_s2g_phase2_blocked(uint64_t *gmem, uint64_t *smem,
                                         size_t n, size_t n1, size_t n2);
}