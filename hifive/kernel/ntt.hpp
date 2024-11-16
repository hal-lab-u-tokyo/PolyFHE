#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

struct NTTParams {
    uint64_t N;
    uint64_t N_inv;
    uint64_t batch;
    uint64_t q;
    uint64_t root;
    uint64_t root_inv;
    uint64_t *roots_pow;
    uint64_t *roots_pow_shoup;
    uint64_t *roots_pow_inv;
    uint64_t *roots_pow_inv_shoup;
    int n1; // N = n1 * n2
    int n2;
    int n1_3;
    int n2_3;
    int n3_3;
};

extern "C" {
__device__ void NTTPhase1Batched(DeviceContext *dc, uint64_t *buff,
                                 NTTParams *params);
__device__ void NTTPhase2Batched(DeviceContext *dc, uint64_t *buff,
                                 NTTParams *params);
}