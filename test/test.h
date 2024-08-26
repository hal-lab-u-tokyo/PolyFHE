#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// HiFive
#include "ciphertext.h"
#include "evaluate.h"
#include "gpu_utils.h"
#include "ntt.h"
#include "polynomial.h"

// SEAL
#include "seal/seal.h"

__global__ void check_can_access(uint64_t *data, uint64_t idx);