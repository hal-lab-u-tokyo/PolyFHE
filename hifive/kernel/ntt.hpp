#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

extern "C" {
__device__ void NTTPhase1Batched(DeviceContext *dc, const int batch,
                                 uint64_t *smem, uint64_t *gmem);
__device__ void NTTPhase2Batched(DeviceContext *dc, const int batch,
                                 uint64_t *smem, uint64_t *gmem);
}