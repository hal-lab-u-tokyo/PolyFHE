#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hifive/kernel/device_context.hpp"

#define CudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess) {                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(0);                                                 \
        }                                                            \
    }

extern "C" {
__device__ void Add(DeviceContext *dc, const int l, uint64_t *dst,
                    const uint64_t *a, const uint64_t *b, const int n_dst,
                    const int n_a, const int n_b);

__device__ void Mult(DeviceContext *dc, const int l, uint64_t *dst,
                     const uint64_t *a, const uint64_t *b, const int n_dst,
                     const int n_a, const int n_b);

__global__ void Ntt8PointPerThreadPhase1(DeviceContext *dc, uint64_t *op,
                                         const int num_prime, const int N,
                                         const int start_prime_idx,
                                         const int radix);

__global__ void Ntt8PointPerThreadPhase2(DeviceContext *dc, uint64_t *op,
                                         const int m, const int num_prime,
                                         const int N, const int start_prime_idx,
                                         const int radix);

__global__ void Intt8PointPerThreadPhase2OoP(DeviceContext *dc,
                                             const uint64_t *in, uint64_t *out,
                                             const int m, const int num_prime,
                                             const int N,
                                             const int start_prime_idx,
                                             const int radix);

__global__ void Intt8PointPerThreadPhase1OoP(DeviceContext *dc,
                                             const uint64_t *in, uint64_t *out,
                                             const int m, const int num_prime,
                                             const int N,
                                             const int start_prime_idx, int pad,
                                             int radix);
}