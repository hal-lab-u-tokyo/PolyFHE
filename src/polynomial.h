#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "phantom.h"

namespace hifive {

__global__ void poly_add_mod(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                             uint64_t *d_modulus, int limb);

__global__ void poly_add(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                         DModulus *modulus, int limb);

__global__ void poly_mult(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                          DModulus *modulus, int limb);
__global__ void poly_mult_accum(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                                DModulus *modulus, int limb);

} // namespace hifive