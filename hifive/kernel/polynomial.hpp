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
__device__ void Add(Params *params, uint64_t *dst, const uint64_t *a,
                    const uint64_t *b, const int n, const int n_dst,
                    const int n_a, const int n_b);

__device__ void Mult(Params *p, const int n, const int l, uint64_t *dst,
                     const uint64_t *a, const uint64_t *b, const int n_dst,
                     const int n_a, const int n_b);

void Add_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b);
}