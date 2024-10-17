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
__device__ void Add(DeviceContext *dc, const int N, const int block_x,
                    const int block_y, uint64_t *dst, const uint64_t *a,
                    const uint64_t *b, const bool if_dst_shared,
                    const bool if_a_shared, const bool if_b_shared);

__device__ void Mult(DeviceContext *dc, const int N, const int block_x,
                     const int block_y, uint64_t *dst, const uint64_t *a,
                     const uint64_t *b, const bool if_dst_shared,
                     const bool if_a_shared, const bool if_b_shared);
}