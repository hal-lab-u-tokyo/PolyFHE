#include <stdio.h>

#include "hifive/kernel/polynomial.hpp"

__device__ void Add(DeviceContext *dc, const int N, const int block_x,
                    const int block_y, uint64_t *dst, const uint64_t *a,
                    const uint64_t *b, const bool if_dst_shared,
                    const bool if_a_shared, const bool if_b_shared) {
    const int idx = threadIdx.x;
    if (idx < block_x) {
        for (int i = 0; i < block_y; i++) {
            const int dst_idx = if_dst_shared ? i * block_x + idx : i * N + idx;
            const int a_idx = if_a_shared ? i * block_x + idx : i * N + idx;
            const int b_idx = if_b_shared ? i * block_x + idx : i * N + idx;
            dst[dst_idx] = a[a_idx] + b[b_idx];
        }
    }
}