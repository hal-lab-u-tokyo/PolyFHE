#include <cstdint>
#include <iostream>

#include "hifive/kernel/ntt.hpp"
#include "test/test.hpp"

void test_ntt(FHEContext& context, const uint64_t logN, const uint64_t L) {
    std::cout << "=============== Test NTT ==================" << std::endl;
    const uint64_t N = 1 << logN;
    std::shared_ptr<Params> params = context.get_h_params();
    NTTParams* ntt_params = params->ntt_params;

    const int n_iters = 5;
    double total = 0;

    dim3 gridPhase1(ntt_params->n2 * ntt_params->batch);
    dim3 blockPhase1(ntt_params->n1 / 8);
    dim3 gridPhase2(ntt_params->n1 * ntt_params->batch);
    dim3 blockPhase2(ntt_params->n2 / 8);
    const int shared_mem_size_phase1 = ntt_params->n1 * sizeof(uint64_t);
    const int shared_mem_size_phase2 = ntt_params->n2 * sizeof(uint64_t);

    for (int iter = 0; iter < n_iters; iter++) {
        auto [inout, d_inout] = create_linear_polynomial(N, L);

        auto start = std::chrono::high_resolution_clock::now();

        ntt_phase1_batched<<<gridPhase1, blockPhase1, shared_mem_size_phase1>>>(
            d_inout, context.get_d_ntt_params());
        ntt_phase2_batched<<<gridPhase2, blockPhase2, shared_mem_size_phase2>>>(
            d_inout, context.get_d_ntt_params());

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
        double exectime =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        std::cout << "Time: " << exectime << "us" << std::endl;
        if (iter > 0) {
            total += exectime;
        }
    }
    std::cout << "Average time: " << total / (n_iters - 1) << "us" << std::endl;
}