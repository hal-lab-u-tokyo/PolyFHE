#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

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

/** Modular multiplication, result = operand1 * operand2 % mod
 * @param[in] operand1 The first operand (64 bits).
 * @param[in] operand2 Second operand (64-bit operand).
 * @param[in] operand2_shoup Second operand ( 64-bit quotient).
 * @param[in] modulus The modulus value (64 bits).
 *  res (64 bits).
 */
[[nodiscard]] __inline__ __device__ uint64_t xxx_multiply_and_reduce_shoup(
    const uint64_t &operand1, const uint64_t &operand2,
    const uint64_t &operand2_shoup, const uint64_t &modulus) {
    const uint64_t hi = __umul64hi(operand1, operand2_shoup);
    uint64_t res = operand1 * operand2 - hi * modulus;
    csub_q(res, modulus);
    return res;
}

/** uint64 modular multiplication, result = operand1 * operand2 % mod
 * @param[in] operand1 The first operand (64 bits).
 * @param[in] operand2 The second operand (64 bits).
 * @param[in] modulus The modulus value (64 bits).
 * @param[in] barrett_mu 2^128/mod, (128 bits).
 *  res (64 bits).
 */
__forceinline__ __device__ uint64_t xxx_multiply_and_barrett_reduce_uint64(
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

typedef struct xxx_uint128_t {
    uint64_t hi;
    uint64_t lo;
    // TODO: implement uint128_t basic operations
    //    __device__ uint128_t &operator+=(const uint128_t &op);
} xxx_uint128_t;

struct xxx_uint128_t2 {
    xxx_uint128_t x;
    xxx_uint128_t y;
};

/** unsigned 128-bit integer addition.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[out] result The result
 * return carry bit
 */
__forceinline__ __device__ void xxx_add_uint128_uint128(
    const xxx_uint128_t &operand1, const xxx_uint128_t &operand2,
    xxx_uint128_t &result) {
    asm volatile(
        "{\n\t"
        "add.cc.u64     %0, %2, %4;\n\t"
        "addc.u64    %1, %3, %5;\n\t"
        "}"
        : "=l"(result.lo), "=l"(result.hi)
        : "l"(operand1.lo), "l"(operand1.hi), "l"(operand2.lo),
          "l"(operand2.hi));
}

/**  a * b, return product is 128 bits.
 * @param[in] operand1 The multiplier
 * @param[in] operand2 The multiplicand
 * return operand1 * operand2 in 128bits
 */
__forceinline__ __device__ xxx_uint128_t
xxx_multiply_uint64_uint64(const uint64_t &operand1, const uint64_t &operand2) {
    xxx_uint128_t result_;
    result_.lo = operand1 * operand2;
    result_.hi = __umul64hi(operand1, operand2);
    return result_;
}

/** Reduce an 128-bit product into 64-bit modulus field using Barrett reduction
 * @param[in] product The input 128-bit product.
 * @param[in] modulus The modulus.
 * @param[in] barrett_mu The pre-computed value for mod, (2^128/modulus) in 128
 * bits. Return prod % mod
 */
__forceinline__ __device__ uint64_t xxx_barrett_reduce_uint128_uint64(
    const xxx_uint128_t &product, const uint64_t &modulus,
    const uint64_t *barrett_mu) {
    uint64_t result;
    uint64_t q = modulus;

    uint64_t lo = product.lo;
    uint64_t hi = product.hi;
    uint64_t ratio0 = barrett_mu[0];
    uint64_t ratio1 = barrett_mu[1];

    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        // Multiply input and const_ratio
        // Round 1
        " mul.hi.u64 tmp, %1, %3;\n\t"
        " mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
        " madc.hi.u64 %0, %1, %4, 0;\n\t"
        // Round 2
        " mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
        " madc.hi.u64 %0, %2, %3, %0;\n\t"
        // This is all we care about
        " mad.lo.u64 %0, %2, %4, %0;\n\t"
        // Barrett subtraction
        " mul.lo.u64 %0, %0, %5;\n\t"
        " sub.u64 %0, %1, %0;\n\t"
        "}"
        : "=l"(result)
        : "l"(lo), "l"(hi), "l"(ratio0), "l"(ratio1), "l"(q));
    csub_q(result, q);
    return result;
}

__device__ inline uint64_t load_global_nc_u64(const uint64_t *ptr) {
    uint64_t val;
    asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}

__forceinline__ __device__ void MulKeyAccumOp(Params *params, uint64_t *dst_ax,
                                              uint64_t *dst_bx, uint64_t **in,
                                              uint64_t **key, int beta,
                                              size_t tid, int twr) {
    const int size_QP_n = params->N * params->KL;
    xxx_uint128_t prod0, prod1;
    xxx_uint128_t acc0, acc1;
    // acc0 = xxx_multiply_uint64_uint64(in[0][tid], key[0][tid]);
    // acc1 = xxx_multiply_uint64_uint64(in[0][tid], key[0][tid + size_QP_n]);

    acc0 = xxx_multiply_uint64_uint64(in[0][tid],
                                      load_global_nc_u64(&key[0][tid]));
    acc1 = xxx_multiply_uint64_uint64(
        in[0][tid], load_global_nc_u64(&key[0][tid + size_QP_n]));

    for (int i = 1; i < beta; i++) {
        // prod0 = xxx_multiply_uint64_uint64(in[i][tid], key[i][tid]);
        // prod1 = xxx_multiply_uint64_uint64(in[i][tid], key[i][tid +
        // size_QP_n]);
        prod0 = xxx_multiply_uint64_uint64(in[i][tid],
                                           load_global_nc_u64(&key[i][tid]));
        prod1 = xxx_multiply_uint64_uint64(
            in[i][tid], load_global_nc_u64(&key[i][tid + size_QP_n]));

        xxx_add_uint128_uint128(prod0, acc0, acc0);
        xxx_add_uint128_uint128(prod1, acc1, acc1);
    }

    uint64_t res0 = xxx_barrett_reduce_uint128_uint64(
        acc0, params->qVec[twr], params->modulus_const_ratio + 2 * twr);
    uint64_t res1 = xxx_barrett_reduce_uint128_uint64(
        acc1, params->qVec[twr], params->modulus_const_ratio + 2 * twr);
    dst_ax[tid] = res0;
    dst_bx[tid] = res1;
}

__forceinline__ __device__ void MulKeyAccumOp_opt(
    Params *params, uint64_t *dst_ax, uint64_t *dst_bx, uint64_t **in,
    uint64_t **key, int beta, size_t tid, int twr, uint64_t *shared,
    uint64_t *reg, int idx_j) {
    const int size_QP_n = params->N * params->KL;
    xxx_uint128_t prod0, prod1;
    xxx_uint128_t acc0, acc1;
    // acc0 = xxx_multiply_uint64_uint64(in[0][tid], key[0][tid]);
    // acc1 = xxx_multiply_uint64_uint64(in[0][tid], key[0][tid + size_QP_n]);
    // acc0 = xxx_multiply_uint64_uint64(reg[idx_j],
    //                                   load_global_nc_u64(&key[beta -
    //                                   1][tid]));
    // acc1 = xxx_multiply_uint64_uint64(
    //     reg[idx_j], load_global_nc_u64(&key[beta - 1][tid + size_QP_n]));
    acc0 = xxx_multiply_uint64_uint64(in[beta - 1][tid],
                                      load_global_nc_u64(&key[beta - 1][tid]));
    acc1 = xxx_multiply_uint64_uint64(
        in[beta - 1][tid], load_global_nc_u64(&key[beta - 1][tid + size_QP_n]));

    for (int i = 0; i < beta - 1; i++) {
        // prod0 = xxx_multiply_uint64_uint64(in[i][tid], key[i][tid]);
        // prod1 = xxx_multiply_uint64_uint64(in[i][tid], key[i][tid +
        // size_QP_n]);
        prod0 = xxx_multiply_uint64_uint64(in[i][tid],
                                           load_global_nc_u64(&key[i][tid]));
        prod1 = xxx_multiply_uint64_uint64(
            in[i][tid], load_global_nc_u64(&key[i][tid + size_QP_n]));
        xxx_add_uint128_uint128(prod0, acc0, acc0);
        xxx_add_uint128_uint128(prod1, acc1, acc1);
    }

    uint64_t res0 = xxx_barrett_reduce_uint128_uint64(
        acc0, params->qVec[twr], params->modulus_const_ratio + 2 * twr);
    uint64_t res1 = xxx_barrett_reduce_uint128_uint64(
        acc1, params->qVec[twr], params->modulus_const_ratio + 2 * twr);
    dst_ax[tid] = res0;
    dst_bx[tid] = res1;
}

__forceinline__ __device__ auto xxx_base_convert_acc_unroll2_reg(
    const uint64_t *reg, const uint64_t *QHatModp, size_t out_prime_idx,
    size_t degree, size_t ibase_size, size_t degree_idx) {
    xxx_uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        xxx_uint128_t2 out{};

        uint64_t op1_x = reg[2 * i];
        uint64_t op1_y = reg[2 * i + 1];
        out.x = xxx_multiply_uint64_uint64(op1_x, op2);
        xxx_add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = xxx_multiply_uint64_uint64(op1_y, op2);
        xxx_add_uint128_uint128(out.y, accum.y, accum.y);
    }
    return accum;
}

__forceinline__ __device__ auto xxx_base_convert_acc_unroll2(
    const uint64_t *ptr, const uint64_t *QHatModp, size_t out_prime_idx,
    size_t degree, size_t ibase_size, size_t degree_idx) {
    xxx_uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        xxx_uint128_t2 out{};

        uint64_t op1_x = ptr[i * degree + degree_idx];
        uint64_t op1_y = ptr[i * degree + degree_idx + 1];
        out.x = xxx_multiply_uint64_uint64(op1_x, op2);
        xxx_add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = xxx_multiply_uint64_uint64(op1_y, op2);
        xxx_add_uint128_uint128(out.y, accum.y, accum.y);
    }
    return accum;
}

__forceinline__ __device__ void BConvOpNoReg(
    Params *params, uint64_t *res1, uint64_t *res2, const uint64_t *in,
    uint64_t *s_qiHat_mod_pj, const size_t degree_idx,
    const size_t out_prime_idx, const DModulus *ibase, uint64_t ibase_size,
    const DModulus *obase, uint64_t obase_size, size_t startPartIdx,
    size_t size_PartQl) {
    xxx_uint128_t2 accum = xxx_base_convert_acc_unroll2(
        in, s_qiHat_mod_pj, out_prime_idx, params->N, ibase_size, degree_idx);

    uint64_t obase_value = obase[out_prime_idx].value();
    uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0],
                               obase[out_prime_idx].const_ratio()[1]};

    *res1 =
        xxx_barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
    *res2 =
        xxx_barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
}

__forceinline__ __device__ void BConvOpNoReg_debug(
    Params *params, uint64_t *res1, uint64_t *res2, const uint64_t *in,
    const uint64_t *s_qiHat_mod_pj, const size_t degree_idx,
    const size_t out_prime_idx, const size_t out_prime_idx2,
    uint64_t ibase_size, size_t startPartIdx, size_t size_PartQl,
    const uint64_t *twiddles, const uint64_t *twiddles_shoup,
    const DModulus *modulus) {
    xxx_uint128_t2 accum = xxx_base_convert_acc_unroll2(
        in, s_qiHat_mod_pj, out_prime_idx, params->N, ibase_size, degree_idx);

    uint64_t obase_value = modulus[out_prime_idx2].value();
    uint64_t obase_ratio[2] = {modulus[out_prime_idx2].const_ratio()[0],
                               modulus[out_prime_idx2].const_ratio()[1]};

    *res1 =
        xxx_barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
    *res2 =
        xxx_barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
}

__forceinline__ __device__ void BConvOp(
    Params *params, uint64_t *res1, uint64_t *res2, const uint64_t *in_reg,
    uint64_t *s_qiHat_mod_pj, const size_t degree_idx,
    const size_t out_prime_idx, const DModulus *ibase, uint64_t ibase_size,
    const DModulus *obase, uint64_t obase_size, size_t startPartIdx,
    size_t size_PartQl) {
    xxx_uint128_t2 accum =
        xxx_base_convert_acc_unroll2_reg(in_reg, s_qiHat_mod_pj, out_prime_idx,
                                         params->N, ibase_size, degree_idx);

    uint64_t obase_value = obase[out_prime_idx].value();
    uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0],
                               obase[out_prime_idx].const_ratio()[1]};

    *res1 =
        xxx_barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
    *res2 =
        xxx_barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
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