#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "hifive/kernel/polynomial.hpp"

__global__ void gAdd(DeviceContext *dc, const int N, const int block_width,
                     const int block_height, uint64_t *d_c, const uint64_t *d_a,
                     const uint64_t *d_b, bool c_is_shared, bool a_is_shared,
                     bool b_is_shared) {
    uint64_t *d_ci = d_c + blockIdx.x * block_width;
    const uint64_t *d_ai = d_a + blockIdx.x * block_width;
    const uint64_t *d_bi = d_b + blockIdx.x * block_width;
    Add(dc, N, block_width, block_height, d_ci, d_ai, d_bi, false, false,
        false);
}

int test_poly_add(DeviceContext *dc) {
    uint64_t *a, *b, *c;
    uint64_t *d_a, *d_b, *d_c;
    const int N = 1 << 16;
    const int L = 24;
    a = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    b = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    c = (uint64_t *) malloc(N * L * sizeof(uint64_t));
    for (int i = 0; i < N * L; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    cudaMalloc(&d_a, N * L * sizeof(uint64_t));
    cudaMalloc(&d_b, N * L * sizeof(uint64_t));
    cudaMalloc(&d_c, N * L * sizeof(uint64_t));
    cudaMemcpy(d_a, a, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);

    const int block_width = 1 << 8;
    const int block_height = L;
    const int block_size = block_width * block_height * sizeof(uint64_t);
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        gAdd<<<N / block_width, block_width, block_size>>>(
            dc, N, block_width, block_height, d_c, d_a, d_b, false, false,
            false);
        CudaCheckError();
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(c, d_c, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N * L; i++) {
            if (c[i] != a[i] + b[i]) {
                std::cout << "Error at index " << i << ": " << c[i]
                          << " != " << a[i] << " + " << b[i] << std::endl;
                return 1;
            }
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Time: " << elapsed.count() << "us" << std::endl;
    }
    return 0;
}