#include <stdio.h>

#include "polynomial.h"

__device__ void poly_add(uint64_t *dst, const uint64_t *a, const uint64_t *b,
                         const bool dst_shared, const bool a_shared,
                         const bool b_shared, const int width, const int height,
                         const int N) {
    int idx = threadIdx.x;
    if (idx < width) {
#pragma unroll
        for (int i = 0; i < height; i++) {
            const int dst_idx = dst_shared ? i * width + idx : i * N + idx;
            const int a_idx = a_shared ? i * width + idx : i * N + idx;
            const int b_idx = b_shared ? i * width + idx : i * N + idx;
            dst[dst_idx] = a[a_idx] + b[b_idx];
        }
    }
}

__device__ void poly_add_equal(uint64_t *dst, const uint64_t *a,
                               const uint64_t *b, const bool dst_shared,
                               const bool a_shared, const bool b_shared,
                               const int width, const int height, const int N) {
    int idx = threadIdx.x;
    if (idx < width) {
#pragma unroll
        for (int i = 0; i < height; i++) {
            const int dst_idx = dst_shared ? i * width + idx : i * N + idx;
            const int a_idx = a_shared ? i * width + idx : i * N + idx;
            const int b_idx = b_shared ? i * width + idx : i * N + idx;
            dst[dst_idx] += a[a_idx] + b[b_idx];
        }
    }
}