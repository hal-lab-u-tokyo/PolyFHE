#include "hifive/kernel/polynomial.hpp"

struct uint128_t {
    uint64_t hi = 0;
    uint64_t lo = 0;
    __device__ uint128_t &operator+=(const uint128_t &op);
    __device__ uint128_t &operator-=(const uint128_t &op);
};

__inline__ __device__ uint128_t &uint128_t::operator+=(const uint128_t &a) {
    uint128_t res;
    res.lo = this->lo + a.lo;
    res.hi = this->hi + a.hi + (res.lo < a.lo);
    this->lo = res.lo;
    this->hi = res.hi;
    return *this;
}

__inline__ __device__ uint128_t &uint128_t::operator-=(const uint128_t &a) {
    uint128_t res;
    res.lo = this->lo - a.lo;
    res.hi = this->hi - a.hi - (res.lo > this->lo);
    this->lo = res.lo;
    this->hi = res.hi;
    return *this;
}

// k = ceil(logq) is at most 61
__device__ uint64_t modBarrett(const uint128_t a, const uint64_t q,
                               const uint64_t mu, const uint64_t twok) {
    const uint64_t k = twok / 2;

    // x1 = t >> (k - 2)
    // t is 2K bits, so x1 is K + 2 bits
    const uint64_t x1 = (a.lo >> (k - 2)) | (a.hi << (64 - k + 2));

    // x2 = x1 * mu
    // mu is K bits, so x2 is 2K + 2 bits
    uint128_t x2;
    x2.hi = __umul64hi(x1, mu);
    x2.lo = x1 * mu;

    // s = x2 >> (k + 2)
    // s is K bits
    const uint64_t s = (x2.lo >> (k + 2)) | (x2.hi << (64 - k - 2));

    // r = s * q
    // r is 2K bits
    uint128_t r;
    r.hi = __umul64hi(s, q);
    r.lo = s * q;

    // c = t - r
    uint64_t c_lo = a.lo - r.lo;

    // if c >= q, c = c - q
    if (c_lo >= q) {
        c_lo -= q;
    }

    return c_lo;
}

__device__ uint64_t modmul(uint64_t a, uint64_t b, uint64_t q, uint64_t mr,
                           uint64_t twok) {
    // a * b
    uint128_t ab;
    ab.hi = __umul64hi(a, b);
    ab.lo = a * b;

    return modBarrett(ab, q, mr, twok);
}

__device__ void Add(DeviceContext *dc, const int N, const int block_x,
                    const int block_y, uint64_t *dst, const uint64_t *a,
                    const uint64_t *b, const bool if_dst_shared,
                    const bool if_a_shared, const bool if_b_shared) {
    const int idx = threadIdx.x;
    if (idx < block_x) {
        for (int i = 0; i < block_y; i++) {
            const uint64_t qi = dc->qVec[i];
            const int dst_idx = if_dst_shared ? i * block_x + idx : i * N + idx;
            const int a_idx = if_a_shared ? i * block_x + idx : i * N + idx;
            const int b_idx = if_b_shared ? i * block_x + idx : i * N + idx;
            uint64_t result = a[a_idx] + b[b_idx];
            dst[dst_idx] = (result >= qi) ? result - qi : result;
        }
    }
}

__device__ void Mult(DeviceContext *dc, const int N, const int block_x,
                     const int block_y, uint64_t *dst, const uint64_t *a,
                     const uint64_t *b, const bool if_dst_shared,
                     const bool if_a_shared, const bool if_b_shared) {
    const int idx = threadIdx.x;
    if (idx < block_x) {
        for (int i = 0; i < block_y; i++) {
            const int dst_idx = if_dst_shared ? i * block_x + idx : i * N + idx;
            const int a_idx = if_a_shared ? i * block_x + idx : i * N + idx;
            const int b_idx = if_b_shared ? i * block_x + idx : i * N + idx;

            // barret reduction
            const uint64_t qi = dc->qVec[i];
            const uint64_t mu = dc->qrVec[i];
            const uint64_t twok = dc->qTwok[i];
            dst[dst_idx] = modmul(a[a_idx], b[b_idx], qi, mu, twok);
        }
    }
}