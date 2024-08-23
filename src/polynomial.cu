#include "polynomial.h"

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
    uint64_t mod = modulus[limb].value();
    d_out[idx] = (v0 * v1) % mod;
    if (idx == 0 && limb == 0) {
        printf("v0: %lu, v1: %lu, mod: %lu, d_out: %lu\n", v0, v1, mod,
               d_out[idx]);
    }
}

__global__ void poly_mult_accum(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                                DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] += ((v0 * v1) % mod);
    d_out[idx] %= mod;
    if (idx == 0 && limb == 0) {
        printf("v0: %lu, v1: %lu, mod: %lu, d_out: %lu\n", v0, v1, mod,
               d_out[idx]);
    }
}