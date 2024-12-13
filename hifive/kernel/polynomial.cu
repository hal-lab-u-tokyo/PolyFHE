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

__device__ void Add(Params *dc, const int n, const int l, uint64_t *dst,
                    const uint64_t *a, const uint64_t *b, const int n_dst,
                    const int n_a, const int n_b) {
    for (int i = threadIdx.x; i < n * l; i += blockDim.x) {
        const int l_idx = i / n;
        const int n_idx = i % n;
        const uint64_t qi = dc->qVec[l_idx];
        const int dst_idx = l_idx * n_dst + n_idx;
        const int a_idx = l_idx * n_a + n_idx;
        const int b_idx = l_idx * n_b + n_idx;
        uint64_t res = a[a_idx] + b[b_idx];
        if (res >= qi) {
            res -= qi;
        }
        dst[dst_idx] = res;
    }
}

__device__ void Mult(Params *dc, const int n, const int l, uint64_t *dst,
                     const uint64_t *a, const uint64_t *b, const int n_dst,
                     const int n_a, const int n_b) {
    for (int i = threadIdx.x; i < n * l; i += blockDim.x) {
        const int l_idx = i / n;
        const int n_idx = i % n;
        const uint64_t qi = dc->qVec[l_idx];
        const uint64_t mu = dc->qrVec[l_idx];
        const uint64_t twok = dc->qTwok[l_idx];
        const int dst_idx = l_idx * n_dst + n_idx;
        const int a_idx = l_idx * n_a + n_idx;
        const int b_idx = l_idx * n_b + n_idx;
        dst[dst_idx] = modmul(a[a_idx], b[b_idx], qi, mu, twok);
    }
}

__device__ void MultOutputTwo(Params *dc, const int n, const int l,
                              uint64_t *dst0, uint64_t *dst1, const uint64_t *a,
                              const uint64_t *b, const int n_dst0,
                              const int n_dst1, const int n_a, const int n_b) {
    for (int i = threadIdx.x; i < n * l; i += blockDim.x) {
        const int l_idx = i / n;
        const int n_idx = i % n;
        const uint64_t qi = dc->qVec[l_idx];
        const uint64_t mu = dc->qrVec[l_idx];
        const uint64_t twok = dc->qTwok[l_idx];
        const int dst0_idx = l_idx * n_dst0 + n_idx;
        const int dst1_idx = l_idx * n_dst1 + n_idx;
        const int a_idx = l_idx * n_a + n_idx;
        const int b_idx = l_idx * n_b + n_idx;
        dst0[dst0_idx] = modmul(a[a_idx], b[b_idx], qi, mu, twok);
        dst1[dst1_idx] = modmul(a[a_idx], b[b_idx], qi, mu, twok);
    }
}