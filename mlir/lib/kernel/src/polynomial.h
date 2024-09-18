#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "phantom.h"

namespace hifive {

__global__ void poly_add_mod(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                             uint64_t *d_modulus, int limb);

__global__ void poly_mult_mod(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                              uint64_t *d_modulus, int limb);

} // namespace hifive