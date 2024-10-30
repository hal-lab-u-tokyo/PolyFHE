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

__global__ void gMult(DeviceContext *dc, const int N, const int block_x,
                      const int block_y, uint64_t *d_c, const uint64_t *d_a,
                      const uint64_t *d_b, bool c_is_shared, bool a_is_shared,
                      bool b_is_shared) {
    uint64_t *d_ci = d_c + blockIdx.x * block_x;
    const uint64_t *d_ai = d_a + blockIdx.x * block_x;
    const uint64_t *d_bi = d_b + blockIdx.x * block_x;
    Mult(dc, N, block_x, block_y, d_ci, d_ai, d_bi, false, false, false);
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
        cudaDeviceSynchronize();
        CudaCheckError();
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

void test_poly_mult(FHEContext &context, const int N, const int L,
                    const int block_x, const int block_y) {
    std::cout << "test_poly_mult: N=" << N << ", L=" << L
              << ", block_x=" << block_x << ", block_y=" << block_y
              << std::endl;
    /*
        // TODO: Overflow issue
        auto [a, d_a] =
            create_random_polynomial(N, L, context.get_host_context()->qVec);
        auto [b, d_b] =
            create_random_polynomial(N, L, context.get_host_context()->qVec);
        auto [c, d_c] =
            create_random_polynomial(N, L, context.get_host_context()->qVec);
    */
    auto [a, d_a] = create_linear_polynomial(N, L);
    auto [b, d_b] = create_linear_polynomial(N, L);
    auto [c, d_c] = create_linear_polynomial(N, L);
    const int block_size = block_x * block_y * sizeof(uint64_t);
    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        gMult<<<N / block_x, block_x, block_size>>>(
            context.get_device_context(), N, block_x, block_y, d_c, d_a, d_b,
            false, false, false);
        cudaDeviceSynchronize();
        CudaCheckError();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(c, d_c, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                const int idx = i * N + j;
                if (c[idx] !=
                    (a[idx] * b[idx]) % context.get_host_context()->qVec[i]) {
                    std::cout << "Error at index[" << i << "][" << j
                              << "]: " << c[idx] << " != (" << a[idx] << " * "
                              << b[idx] << ") % "
                              << context.get_host_context()->qVec[i] << " = "
                              << (a[idx] * b[idx]) %
                                     context.get_host_context()->qVec[i]
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

void test_poly_ntt(FHEContext &context, const int N, const int L,
                   const int block_x, const int block_y) {
    std::cout << "test_poly_ntt: N=" << N << ", L=" << L
              << ", block_x=" << block_x << ", block_y=" << block_y
              << std::endl;

    /*
    auto [in, d_in] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
    auto [out, d_out] =
        create_random_polynomial(N, L, context.get_host_context()->qVec);
        */
    auto [in, d_in] = create_linear_polynomial(N, L);
    auto [out, d_out] = create_linear_polynomial(N, L);

    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 gridDim(2048);
        dim3 blockDim(256);
        const int per_thread_ntt_size = 8;
        const int first_stage_radix_size = 256;
        const int second_radix_size = N / first_stage_radix_size;
        const int per_thread_storage =
            blockDim.x * per_thread_ntt_size * sizeof(uint64_t);
        const int pad = 4;
        Ntt8PointPerThreadPhase1<<<gridDim, (first_stage_radix_size / 8) * pad,
                                   (first_stage_radix_size + pad + 1) * pad *
                                       sizeof(uint64_t)>>>(
            context.get_device_context(), d_in, L, N, 0,
            first_stage_radix_size / per_thread_ntt_size);
        Ntt8PointPerThreadPhase2<<<gridDim, blockDim.x, per_thread_storage>>>(
            context.get_device_context(), d_in, first_stage_radix_size, L, N, 0,
            second_radix_size / per_thread_ntt_size);

        cudaDeviceSynchronize();
        CudaCheckError();

        Intt8PointPerThreadPhase2OoP<<<gridDim, blockDim, per_thread_storage>>>(
            context.get_device_context(), d_in, d_in, first_stage_radix_size, L,
            N, 0, second_radix_size / per_thread_ntt_size);
        Intt8PointPerThreadPhase1OoP<<<
            gridDim, (first_stage_radix_size / 8) * pad,
            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t)>>>(
            context.get_device_context(), d_in, d_in, 1, L, N, 0, pad,
            first_stage_radix_size / 8);
        cudaDeviceSynchronize();
        CudaCheckError();

        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(out, d_in, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                const int idx = i * N + j;
                if (out[idx] != in[idx]) {
                    std::cout << "Error at index[" << i << "][" << j
                              << "]: " << out[idx] << " != " << in[idx]
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