#include <cstdint>
#include <iostream>

#include "hifive/kernel/ntt.hpp"
#include "test/test.hpp"

__global__ void ntt_phase1(uint64_t* inout, DeviceContext* dc,
                           const int batch) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < dc->N2 * batch) {
        load_g2s_phase1(inout, buffer, dc->N, dc->N1, dc->N2);

        NTTPhase1(dc, batch, buffer, threadIdx.x);

        store_s2g_phase1(inout, buffer, dc->N, dc->N1, dc->N2);
    }
}

__global__ void ntt_phase2(uint64_t* inout, DeviceContext* dc,
                           const int batch) {
    extern __shared__ uint64_t buffer[];
    if (blockIdx.x < dc->N1 * batch) {
        load_g2s_phase2(inout, buffer, dc->N, dc->N1, dc->N2);

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        NTTPhase2(dc, batch, buffer, threadIdx.x, tid % (dc->N / 8));

        store_s2g_phase2(inout, buffer, dc->N, dc->N1, dc->N2);
    }
}

void test_ntt(FHEContext& context, const int N, const int N1, const int N2,
              const int L) {
    std::cout << "test_poly_ntt: N=" << N << ", L=" << L << std::endl;
    std::cout << "N1=" << N1 << ", N2=" << N2 << std::endl;

    /*
    auto [in, d_in] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
    auto [out, d_out] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
        */

    for (int i = 0; i < 5; i++) {
        auto [inout, d_inout] = create_linear_polynomial(N, L);

        dim3 gridPhase1(N2 * L);
        dim3 blockPhase1(N1 / 8);
        dim3 gridPhase2(N1 * L);
        dim3 blockPhase2(N2 / 8);
        const int shared_mem_size_phase1 = N1 * sizeof(uint64_t);
        const int shared_mem_size_phase2 = N2 * sizeof(uint64_t);

        auto start = std::chrono::high_resolution_clock::now();

        ntt_phase1<<<gridPhase1, blockPhase1, shared_mem_size_phase1>>>(
            d_inout, context.get_device_context(), L);
        // ntt_phase2<<<gridPhase2, blockPhase2, shared_mem_size_phase2>>>(
        //     d_inout, context.get_device_context(), L);
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
    }
}