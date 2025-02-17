#include <cstdint>
#include <iostream>

#include "polyfhe/kernel/ntt.hpp"
#include "test/test.hpp"

void ntt_radix2_cpu(uint64_t *a, NTTParams &params, const int batch_idx) {
    const uint64_t q = params.q[batch_idx];
    uint64_t t = params.N;
    uint64_t j1, j2;
    for (int m = 1; m < params.N; m *= 2) {
        t = t / 2;
        for (int i = 0; i < m; i++) {
            j1 = 2 * i * t;
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                uint64_t u = a[j];
                uint64_t v =
                    (a[j + t] * params.roots_pow[batch_idx][m + i]) % q;
                a[j] = (u + v) % q;
                a[j + t] = (u - v + q) % q;
            }
        }
    }
}

void intt_radix2_cpu(uint64_t *a, NTTParams &params, const int batch_idx) {
    const uint64_t q = params.q[batch_idx];
    uint64_t t = 1;
    uint64_t j1, j2, h;
    for (int m = params.N; m > 1; m /= 2) {
        j1 = 0;
        h = m / 2;
        for (int i = 0; i < h; i++) {
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                uint64_t u = a[j];
                uint64_t v = a[j + t];
                a[j] = (u + v) % q;
                a[j + t] = (((u - v + q) % q) *
                            params.roots_pow_inv[batch_idx][h + i]) %
                           q;
            }
            j1 += 2 * t;
        }
        t *= 2;
    }
    const uint64_t n_inv = params.N_inv[batch_idx];
    for (int i = 0; i < params.N; i++) {
        a[i] = (a[i] * n_inv) % q;
    }
}

void test_ntt(FHEContext &context, const uint64_t logN, const uint64_t L) {
    std::cout << "=============== Test NTT ==================" << std::endl;
    const uint64_t N = 1 << logN;
    std::shared_ptr<Params> params = context.get_h_params();
    NTTParams *ntt_params = params->ntt_params;

    std::cout << "N: " << (1 << logN) << ", L: " << L << std::endl;
    std::cout << "N1: " << ntt_params->n1 << ", N2: " << ntt_params->n2
              << std::endl;
    for (int i = 0; i < L; i++) {
        std::cout << "q[" << i << "]: " << ntt_params->q[i] << std::endl;
    }
    for (int i = 0; i < L; i++) {
        std::cout << "root[" << i << "]: " << ntt_params->root[i] << std::endl;
    }

    const int n_iters = 5;
    double total = 0;

    dim3 gridPhase1(ntt_params->n2 * ntt_params->batch);
    dim3 blockPhase1(ntt_params->n1 / 8);
    dim3 gridPhase2(ntt_params->n1 * ntt_params->batch);
    dim3 blockPhase2(ntt_params->n2 / 8);
    const int shared_mem_size_phase1 = ntt_params->n1 * sizeof(uint64_t);
    const int shared_mem_size_phase2 = ntt_params->n2 * sizeof(uint64_t);

    // ref
    auto [a_ref, d_a_ref] = create_linear_polynomial(N, L);
    for (int i = 0; i < L; i++) {
        ntt_radix2_cpu(a_ref + i * N, *ntt_params, i);
    }

    for (int iter = 0; iter < n_iters; iter++) {
        auto [a, d_a] = create_linear_polynomial(N, L);
        auto start = std::chrono::high_resolution_clock::now();

        ntt_phase1_batched<<<gridPhase1, blockPhase1, shared_mem_size_phase1>>>(
            d_a, context.get_d_ntt_params());
        ntt_phase2_batched<<<gridPhase2, blockPhase2, shared_mem_size_phase2>>>(
            d_a, context.get_d_ntt_params());

        cudaDeviceSynchronize();
        CudaCheckError();

        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(a, d_a, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        bool passed = true;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                const int idx = i * N + j;
                if (a[idx] != a_ref[idx]) {
                    std::cout << "Error at index[" << i << "][" << j
                              << "]: " << a[idx] << " != " << a_ref[idx]
                              << std::endl;
                    passed = false;
                }
            }
        }
        double exectime =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        std::cout << "Time: " << exectime << "us" << std::endl;
        if (iter > 0) {
            total += exectime;
        }
        if (passed) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }
    std::cout << "Average time: " << total / (n_iters - 1) << "us" << std::endl;
}