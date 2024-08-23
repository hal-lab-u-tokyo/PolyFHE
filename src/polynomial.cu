#include "polynomial.h"
#include "uintmodmath.cuh"

namespace hifive {

__global__ void poly_add(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                         DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] = (v0 + v1) > mod ? (v0 + v1 - mod) : (v0 + v1);
}

__global__ void poly_mult(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                          DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    DModulus mod = modulus[limb];
    d_out[idx] = phantom::arith::multiply_and_barrett_reduce_uint64(
        v0, v1, mod.value(), mod.const_ratio());
}

__global__ void poly_mult_accum(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                                DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    DModulus mod = modulus[limb];
    d_out[idx] += phantom::arith::multiply_and_barrett_reduce_uint64(
        v0, v1, mod.value(), mod.const_ratio());
    if (d_out[idx] > mod.value()) {
        d_out[idx] -= mod.value();
    }
}

} // namespace hifive