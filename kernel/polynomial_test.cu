#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "polynomial.h"

// Goal of code generation:
__global__ void fused_add(uint64_t *dst, const uint64_t *a, const uint64_t *b,
                          const int width, const int height, const int N) {
    extern __shared__ uint64_t shared[];
    poly_add(shared, a, b, true, false, false, width, height, N);
    poly_add(dst, shared, b, false, true, false, width, height, N);
}

int test_add() {
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

    const int block_width = 256;
    const int block_height = L;
    const int block_size = block_width * block_height * sizeof(uint64_t);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N / block_width; i++) {
        fused_add<<<1, block_width, block_size>>>(
            d_c + i * block_width, d_a + i * block_width, d_b + i * block_width,
            block_width, block_height, N);
    }
    CudaCheckError();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << "us" << std::endl;
    cudaMemcpy(c, d_c, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * L; i++) {
        if (c[i] != a[i] + b[i] * 2) {
            std::cout << "Error at index " << i << ": " << c[i]
                      << " != " << a[i] << " + " << b[i] * 2 << std::endl;
            return 1;
        }
    }
    return 0;
}

int main() {
    int ret;
    ret = test_add();
    if (ret != 0) {
        std::cout << "Test Add failed:  " << ret << std::endl;
        return ret;
    }

    std::cout << "All tests passed" << std::endl;
    return 0;
}