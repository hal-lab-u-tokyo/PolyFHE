#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

// GPU
__global__ void ntt_phase1_batched(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase2_batched(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase1_batched_blocked(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase2_batched_blocked(uint64_t *inout, NTTParams *params);

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