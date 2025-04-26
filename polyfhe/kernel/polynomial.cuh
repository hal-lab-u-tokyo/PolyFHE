#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "phantom-fhe/include/uintmodmath.cuh"
#include "polyfhe/kernel/device_context.hpp"

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

__forceinline__ __device__ void csub_q(uint64_t &operand,
                                       const uint64_t &modulus) {
    const uint64_t tmp = operand - modulus;
    operand = tmp + (tmp >> 63) * modulus;
}

/** uint64 modular multiplication, result = operand1 * operand2 % mod
 * @param[in] operand1 The first operand (64 bits).
 * @param[in] operand2 The second operand (64 bits).
 * @param[in] modulus The modulus value (64 bits).
 * @param[in] barrett_mu 2^128/mod, (128 bits).
 *  res (64 bits).
 */
__forceinline__ __device__ uint64_t multiply_and_barrett_reduce_uint64(
    const uint64_t &operand1, const uint64_t &operand2, const uint64_t &modulus,
    const uint64_t *barrett_mu) {
    uint64_t result;
    uint64_t q = modulus;
    uint64_t ratio0 = barrett_mu[0];
    uint64_t ratio1 = barrett_mu[1];
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " .reg .u64 lo, hi;\n\t"
        // 128-bit multiply
        " mul.lo.u64 lo, %1, %2;\n\t"
        " mul.hi.u64 hi, %1, %2;\n\t"
        // Multiply input and const_ratio
        // Round 1
        " mul.hi.u64 tmp, lo, %3;\n\t"
        " mad.lo.cc.u64 tmp, lo, %4, tmp;\n\t"
        " madc.hi.u64 %0, lo, %4, 0;\n\t"
        // Round 2
        " mad.lo.cc.u64 tmp, hi, %3, tmp;\n\t"
        " madc.hi.u64 %0, hi, %3, %0;\n\t"
        // This is all we care about
        " mad.lo.u64 %0, hi, %4, %0;\n\t"
        // Barrett subtraction
        " mul.lo.u64 %0, %0, %5;\n\t"
        " sub.u64 %0, lo, %0;\n\t"
        "}"
        : "=l"(result)
        : "l"(operand1), "l"(operand2), "l"(ratio0), "l"(ratio1), "l"(q));
    csub_q(result, q);
    return result;
}

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

__forceinline__ __device__ void ElemWiseOp_Elem_v2(
    ElemWiseOp op, Params *params, uint64_t *dst, const uint64_t *a,
    const uint64_t *b, const int dst_global, const int a_global,
    const int b_global, const int sPoly_x, const int l_idx, const int n_gidx,
    const int n_sidx) {
    const uint64_t qi = params->qVec[l_idx];
    const int l_n_gidx = l_idx * params->N + n_gidx;
    const int l_n_sidx = l_idx * sPoly_x + n_sidx;
    const int dst_idx = dst_global * l_n_gidx + (1 - dst_global) * l_n_sidx;
    const int a_idx = a_global * l_n_gidx + (1 - a_global) * l_n_sidx;
    const int b_idx = b_global * l_n_gidx + (1 - b_global) * l_n_sidx;
    dst[dst_idx] = calc_elemwise(op, a[a_idx], b[b_idx], qi);
}

__forceinline__ __device__ void ElemWiseOp_ElemSlot(
    ElemWiseOp op, Params *params, uint64_t *dst, const uint64_t *a,
    const uint64_t *b, const int dst_global, const int a_global,
    const int b_global, const int sPoly_x, const int start_limb,
    const int end_limb, const int n_gidx, const int n_sidx) {
    for (int l_idx = start_limb; l_idx < end_limb; l_idx++) {
        const int l_n_gidx = l_idx * params->N + n_gidx;
        const int l_n_sidx = l_idx * sPoly_x + n_sidx;
        const uint64_t qi = params->qVec[l_idx];
        const int dst_idx = dst_global * l_n_gidx + (1 - dst_global) * l_n_sidx;
        const int a_idx = a_global * l_n_gidx + (1 - a_global) * l_n_sidx;
        const int b_idx = b_global * l_n_gidx + (1 - b_global) * l_n_sidx;
        dst[dst_idx] = calc_elemwise(op, a[a_idx], b[b_idx], qi);
    }
}

__forceinline__ __device__ void ModUpOp(
    Params *params, uint64_t *dst, const uint64_t *src, const int dst_global,
    const int src_global, const int sPoly_x, const int n_gidx, const int n_sidx,
    const int start_limb, const int end_limb) {
    for (int k = 0; k < params->K; k++) {
        const int dst_idx =
            dst_global * ((params->limb + k) * params->N + n_gidx) +
            (1 - dst_global) * ((params->limb + k) * sPoly_x + n_sidx);
        uint64_t sum = 0;
        for (int l = start_limb; l < end_limb; l++) {
            const int src_idx = src_global * (l * params->N + n_gidx) +
                                (1 - src_global) * (l * sPoly_x + n_sidx);
            sum += src[src_idx];
        }
        dst[dst_idx] = sum;
    }
    for (int l = 0; l < params->limb; l++) {
        const int dst_idx = dst_global * (l * params->N + n_gidx) +
                            (1 - dst_global) * (l * sPoly_x + n_sidx);
        dst[dst_idx] = 0;
    }
}

__forceinline__ __device__ void BConvOp(
    Params *params, uint64_t *res1, uint64_t *res2, const uint64_t *in_reg,
    uint64_t *s_qiHat_mod_pj, const size_t degree_idx,
    const size_t out_prime_idx, const DModulus *ibase, uint64_t ibase_size,
    const DModulus *obase, uint64_t obase_size, size_t startPartIdx,
    size_t size_PartQl) {
    phantom::arith::uint128_t2 accum =
        base_convert_acc_unroll2_reg(in_reg, s_qiHat_mod_pj, out_prime_idx,
                                     params->N, ibase_size, degree_idx);

    uint64_t obase_value = obase[out_prime_idx].value();
    uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0],
                               obase[out_prime_idx].const_ratio()[1]};

    *res1 = phantom::arith::barrett_reduce_uint128_uint64(accum.x, obase_value,
                                                          obase_ratio);
    *res2 = phantom::arith::barrett_reduce_uint128_uint64(accum.y, obase_value,
                                                          obase_ratio);
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
void NTTPhase1_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb);
void NTTPhase2_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb);
void iNTTPhase1_h(Params *params, uint64_t *dst, uint64_t *src,
                  const int start_limb, const int end_limb);
void iNTTPhase2_h(Params *params, uint64_t *dst, uint64_t *src,
                  const int start_limb, const int end_limb);
}