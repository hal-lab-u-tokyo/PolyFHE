#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#define CudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess) {                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(0);                                                 \
        }                                                            \
    }

__device__ void poly_add(uint64_t *dst, const uint64_t *a, const uint64_t *b,
                         const bool dst_shared, const bool a_shared,
                         const bool b_shared, const int width, const int height,
                         const int N);
__device__ void poly_add_equal(uint64_t *dst, const uint64_t *a,
                               const uint64_t *b, const bool dst_shared,
                               const bool a_shared, const bool b_shared,
                               const int width, const int height, const int N);