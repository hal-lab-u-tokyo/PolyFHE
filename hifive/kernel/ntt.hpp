#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

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