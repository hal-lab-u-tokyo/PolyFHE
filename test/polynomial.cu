#include "test.h"

TEST(Polynomial, Dummy) { cudaFree(0); }

TEST(Polynomial, Add) {
    const int n = 16;
    const int l = 10;
    uint64_t *h_a = (uint64_t *) malloc(n * sizeof(uint64_t));
    uint64_t *h_mod = (uint64_t *) malloc(l * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *) malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
    }
    for (int i = 0; i < l; i++) {
        h_mod[i] = n + i;
    }

    uint64_t *d_a, *d_b, *d_mod, *d_out;
    checkCudaErrors(cudaMalloc(&d_a, n * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_b, n * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_mod, l * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_out, n * sizeof(uint64_t)));
    checkCudaErrors(
        cudaMemcpy(d_a, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_b, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_mod, h_mod, l * sizeof(uint64_t), cudaMemcpyHostToDevice));
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(n, 1, 1);
    hifive::poly_add_mod<<<gridSize, blockSize>>>(d_out, d_a, d_b, d_mod, 3);
    cudaDeviceSynchronize();
    checkCudaErrors(
        cudaMemcpy(h_out, d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(h_out[i], (i + i) % (n + 3));
    }
}

TEST(Polynomial, Mult) {
    const int n = 16;
    const int l = 10;
    uint64_t *h_a = (uint64_t *) malloc(n * sizeof(uint64_t));
    uint64_t *h_mod = (uint64_t *) malloc(l * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *) malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
    }
    for (int i = 0; i < l; i++) {
        h_mod[i] = n + i;
    }

    uint64_t *d_a, *d_b, *d_mod, *d_out;
    checkCudaErrors(cudaMalloc(&d_a, n * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_b, n * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_mod, l * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_out, n * sizeof(uint64_t)));
    checkCudaErrors(
        cudaMemcpy(d_a, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_b, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_mod, h_mod, l * sizeof(uint64_t), cudaMemcpyHostToDevice));
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(n, 1, 1);
    hifive::poly_mult_mod<<<gridSize, blockSize>>>(d_out, d_a, d_b, d_mod, 3);
    cudaDeviceSynchronize();
    checkCudaErrors(
        cudaMemcpy(h_out, d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(h_out[i], (i * i) % (n + 3));
    }
}