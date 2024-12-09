#include <cstdint>
#include <iostream>

#include "hifive/kernel/ntt.hpp"
#include "test/test.hpp"

void test_ntt(FHEContext& context, const uint64_t logN, const uint64_t L) {
    std::cout << "=============== Test NTT ==================" << std::endl;
    const uint64_t N = 1 << logN;

    const int n_iters = 5;
    int total = 0;
    for (int iter = 0; iter < n_iters; iter++) {
        auto [inout, d_inout] = create_linear_polynomial(N, L);

        auto start = std::chrono::high_resolution_clock::now();

        cudaDeviceSynchronize();
        CudaCheckError();

        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(inout, d_inout, N * L * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                const int idx = i * N + j;
                /*
                if (out[idx] != in[idx]) {
                    std::cout << "Error at index[" << i << "][" << j
                              << "]: " << out[idx] << " != " << in[idx]
                              << std::endl;
                    return;
                }
                */
            }
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Time: " << elapsed.count() << "us" << std::endl;
        if (iter > 0) {
            total += elapsed.count();
        }
    }
    std::cout << "Average time: " << total / (n_iters - 1) << "us" << std::endl;
}