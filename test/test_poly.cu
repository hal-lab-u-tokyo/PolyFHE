#include <cstdint>
#include <iostream>

#include "test/test.hpp"

__global__ void gAdd(DeviceContext *dc, const int N, const int block_x,
                     const int block_y, uint64_t *d_c, const uint64_t *d_a,
                     const uint64_t *d_b, bool c_is_shared, bool a_is_shared,
                     bool b_is_shared) {
    uint64_t *d_ci = d_c + blockIdx.x * block_x;
    const uint64_t *d_ai = d_a + blockIdx.x * block_x;
    const uint64_t *d_bi = d_b + blockIdx.x * block_x;
    Add(dc, N, block_x, block_y, d_ci, d_ai, d_bi, false, false, false);
}

void test_poly_add(FHEContext &context, const int N, const int L,
                   const int block_x, const int block_y) {
    std::cout << "test_poly_add: N=" << N << ", L=" << L
              << ", block_x=" << block_x << ", block_y=" << block_y
              << std::endl;

    auto [a, d_a] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
    auto [b, d_b] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
    auto [c, d_c] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);

    const int block_size = block_x * block_y * sizeof(uint64_t);
    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        gAdd<<<N / block_x, block_x, block_size>>>(
            context.get_device_context(), N, block_x, block_y, d_c, d_a, d_b,
            false, false, false);
        CudaCheckError();
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(c, d_c, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                const int idx = i * N + j;
                if (c[idx] !=
                    (a[idx] + b[idx]) % context.get_host_context()->qVec[i]) {
                    std::cout << "Error at index[" << i << "][" << j
                              << "]: " << c[idx] << " != (" << a[idx] << " + "
                              << b[idx] << ") % "
                              << context.get_host_context()->qVec[i]
                              << std::endl;
                    return;
                }
            }
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Time: " << elapsed.count() << "us" << std::endl;
    }
}