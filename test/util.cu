#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <random>

#include "test/test.hpp"

std::pair<uint64_t *, uint64_t *> create_random_polynomial(
    const int N, const int L, const uint64_t *moduli) {
    // generate 61-bit random numbers
    std::random_device rd;
    std::mt19937_64 gen(rd());
    uint64_t *a = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    for (int i = 0; i < L; i++) {
        const uint64_t mod = moduli[i];
        std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
        for (int j = 0; j < N; j++) {
            a[i * N + j] = dist(gen);
        }
    }

    uint64_t *d_a;
    cudaMalloc(&d_a, N * L * sizeof(uint64_t));
    cudaMemcpy(d_a, a, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    return {a, d_a};
}

std::pair<uint64_t *, uint64_t *> create_linear_polynomial(const int N,
                                                           const int L) {
    uint64_t *a = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    for (int i = 0; i < N * L; i++) {
        a[i] = i;
    }

    uint64_t *d_a;
    cudaMalloc(&d_a, N * L * sizeof(uint64_t));
    cudaMemcpy(d_a, a, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    return {a, d_a};
}

std::pair<uint64_t *, uint64_t *> create_zeros_polynomial(const int N,
                                                          const int L) {
    uint64_t *a = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    for (int i = 0; i < N * L; i++) {
        a[i] = 0;
    }

    uint64_t *d_a;
    cudaMalloc(&d_a, N * L * sizeof(uint64_t));
    cudaMemcpy(d_a, a, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    return {a, d_a};
}