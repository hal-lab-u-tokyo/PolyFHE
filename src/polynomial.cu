#include "polynomial.h"
#include "uintmodmath.cuh"

namespace hifive {

__global__ void poly_add_mod(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                             uint64_t *d_mod, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = d_mod[limb];
    d_out[idx] = (v0 + v1) >= mod ? (v0 + v1 - mod) : (v0 + v1);
}

__global__ void poly_mult_mod(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                              uint64_t *d_mod, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = d_mod[limb];
    d_out[idx] = (v0 * v1) % mod;
}

} // namespace hifive