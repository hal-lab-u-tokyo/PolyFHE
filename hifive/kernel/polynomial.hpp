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
enum class ElemWiseOp { Add, Sub, Mult };

__forceinline__ __device__ uint64_t calc_elemwise(ElemWiseOp op,
                                                  const uint64_t a,
                                                  const uint64_t b,
                                                  const uint64_t q) {
    uint64_t res;
    if (op == ElemWiseOp::Add) {
        res = a + b;
        if (res >= q) {
            res -= q;
        }
    } else if (op == ElemWiseOp::Sub) {
        res = a + q - b;
        if (res >= q) {
            res -= q;
        }
    } else if (op == ElemWiseOp::Mult) {
        // TODO: Faster modmul
        res = (a * b) % q;
    }
    return res;
}

__forceinline__ __device__ void ElemWiseOp_Elem(
    ElemWiseOp op, Params *params, uint64_t *dst, const uint64_t *a,
    const uint64_t *b, const int dst_global, const int a_global,
    const int b_global, const int sPoly_x, const int l_idx, const int n_gidx,
    const int n_sidx) {
    const uint64_t qi = params->qVec[l_idx];
    const int dst_idx =
        dst_global * (l_idx * params->N + n_gidx) + (1 - dst_global) * n_sidx;
    const int a_idx =
        a_global * (l_idx * params->N + n_gidx) + (1 - a_global) * n_sidx;
    const int b_idx =
        b_global * (l_idx * params->N + n_gidx) + (1 - b_global) * n_sidx;
    dst[dst_idx] = calc_elemwise(op, a[a_idx], b[b_idx], qi);
}

void Add_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
           const int start_limb, const int end_limb);
void Sub_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
           const int start_limb, const int end_limb);
void Mult_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
            const int start_limb, const int end_limb);
void ModUp_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
             const int end_limb);
void NTT_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
           const int end_limb);
void iNTT_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
            const int end_limb);
}