#pragma once

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

// GPU
__global__ void ntt_phase1_batched(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase2_batched(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase1_batched_blocked(uint64_t *inout, NTTParams *params);
__global__ void ntt_phase2_batched_blocked(uint64_t *inout, NTTParams *params);
