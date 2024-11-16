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

__device__ void Add(DeviceContext *dc, const int l, uint64_t *dst,
                    const uint64_t *a, const uint64_t *b, const int n_dst,
                    const int n_a, const int n_b) {
    for (int i = 0; i < l; i++) {
        const uint64_t qi = dc->qVec[i];
        const int dst_idx = i * n_dst + threadIdx.x;
        const int a_idx = i * n_a + threadIdx.x;
        const int b_idx = i * n_b + threadIdx.x;
        uint64_t res = a[a_idx] + b[b_idx];
        if (res >= qi) {
            res -= qi;
        }
        dst[dst_idx] = res;
    }
}

__device__ void Mult(DeviceContext *dc, const int l, uint64_t *dst,
                     const uint64_t *a, const uint64_t *b, const int n_dst,
                     const int n_a, const int n_b) {
    for (int i = 0; i < l; i++) {
        const uint64_t qi = dc->qVec[i];
        const uint64_t mu = dc->qrVec[i];
        const uint64_t twok = dc->qTwok[i];
        const int dst_idx = i * n_dst + threadIdx.x;
        const int a_idx = i * n_a + threadIdx.x;
        const int b_idx = i * n_b + threadIdx.x;
        dst[dst_idx] = modmul(a[a_idx], b[b_idx], qi, mu, twok);
    }
}

__device__ void MultOutputTwo(DeviceContext *dc, const int l, uint64_t *dst0,
                              uint64_t *dst1, const uint64_t *a,
                              const uint64_t *b, const int n_dst0,
                              const int n_dst1, const int n_a, const int n_b) {
    for (int i = 0; i < l; i++) {
        const uint64_t qi = dc->qVec[i];
        const uint64_t mu = dc->qrVec[i];
        const uint64_t twok = dc->qTwok[i];
        const int dst0_idx = i * n_dst0 + threadIdx.x;
        const int dst1_idx = i * n_dst1 + threadIdx.x;
        const int a_idx = i * n_a + threadIdx.x;
        const int b_idx = i * n_b + threadIdx.x;
        const uint64_t res = modmul(a[a_idx], b[b_idx], qi, mu, twok);
        dst0[dst0_idx] = res;
        dst1[dst1_idx] = res;
    }
}

__inline__ __device__ uint64_t mul_and_reduce_shoup(const uint64_t op1,
                                                    const uint64_t op2,
                                                    const uint64_t scaled_op2,
                                                    const uint64_t prime) {
    uint64_t hi = __umul64hi(scaled_op2, op1);
    return (uint64_t) op1 * op2 - hi * prime;
};

__device__ __inline__ void butt_ntt_local(uint64_t &a, uint64_t &b,
                                          const uint64_t &w, const uint64_t &w_,
                                          const uint64_t p) {
    uint64_t two_p = 2 * p;
    uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
    if (a >= two_p)
        a -= two_p;
    b = a + (two_p - U);
    a += U;
}

__device__ void butt_intt_local(uint64_t &x, uint64_t &y, const uint64_t &w,
                                const uint64_t &w_, const uint64_t &p) {
    const uint64_t two_p = 2 * p;
    const uint64_t T = two_p - y + x;
    uint64_t new_x = x + y;
    if (new_x >= two_p)
        new_x -= two_p;
    if (T & 1)
        new_x += p;
    x = (new_x >> 1);
    y = mul_and_reduce_shoup(T, w, w_, p);
}

__global__ void Ntt8PointPerThreadPhase1(DeviceContext *dc, uint64_t *op,
                                         const int num_prime, const int N,
                                         const int start_prime_idx,
                                         const int radix) {
    extern __shared__ uint64_t temp[];
    uint64_t *primes = dc->qVec;
    const int m = 1;
    const int pad = 4;
    int Warp_t = threadIdx.x % pad;
    int WarpID = threadIdx.x / pad;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
         i += blockDim.x * gridDim.x) {
        // size of a block
        uint64_t local[8];
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t *a_np = op + np_idx * N;
        const uint64_t *prime_table = primes;
        const uint64_t *W = dc->qRootPowsDivTwo[np_idx];
        const uint64_t *W_ = dc->qRootPowsDivTwoShoup[np_idx];
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
                     pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
            local[j] = *(a_np + N_init + t / 4 * j);
        }
        __syncthreads();
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;
        for (int j = 0; j < 4; j++) {
            butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx],
                           prime);
        }
        for (int j = 0; j < 2; j++) {
            butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                           W_[2 * tw_idx + j], prime);
            butt_ntt_local(local[4 * j + 1], local[4 * j + 3],
                           W[2 * tw_idx + j], W_[2 * tw_idx + j], prime);
        }
        for (int j = 0; j < 4; j++) {
            butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                           W_[4 * tw_idx + j], prime);
        }
        for (int j = 0; j < 8; j++) {
            temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[j];
        }
        int tail = 0;
        __syncthreads();
#pragma unroll
        for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
            int m_idx2 = WarpID / (k / 4);
            int t_idx2 = WarpID % (k / 4);
            for (int l = 0; l < 8; l++) {
                local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k +
                                t_idx2 + (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            for (int j2 = 0; j2 < 4; j2++) {
                butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2],
                               W_[tw_idx2], prime);
            }
            for (int j2 = 0; j2 < 2; j2++) {
                butt_ntt_local(local[4 * j2], local[4 * j2 + 2],
                               W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2],
                               prime);
                butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                               W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2],
                               prime);
            }
            for (int j2 = 0; j2 < 4; j2++) {
                butt_ntt_local(local[2 * j2], local[2 * j2 + 1],
                               W[4 * tw_idx2 + j2], W_[4 * tw_idx2 + j2],
                               prime);
            }

            for (int l = 0; l < 8; l++) {
                temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                     (k / 4) * l] = local[l];
            }
            if (j == radix / 2)
                tail = 1;
            if (j == radix / 4)
                tail = 2;
            __syncthreads();
        }
        if (radix < 8)
            tail = (radix == 4) ? 2 : 1;
        if (tail == 1) {
            for (int l = 0; l < 8; l++) {
                local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
            }
            int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
            butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                           prime);
            butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                           prime);
            for (int l = 0; l < 8; l++) {
                temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
            }
        } else if (tail == 2) {
            for (int l = 0; l < 8; l++) {
                local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
            }
            int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
            butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                           prime);
            butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                           W_[2 * tw_idx2 + 1], prime);
            butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                           W_[2 * tw_idx2 + 2], prime);
            butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                           W_[2 * tw_idx2 + 3], prime);
            for (int l = 0; l < 8; l++) {
                temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
            }
        }
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            local[j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
        }
        for (int j = 0; j < 8; j++) {
            *(a_np + N_init + t / 4 * j) = local[j];
        }
    }
}

__global__ void Ntt8PointPerThreadPhase2(DeviceContext *dc, uint64_t *op,
                                         const int m, const int num_prime,
                                         const int N, const int start_prime_idx,
                                         const int radix) {
    extern __shared__ uint64_t temp[];
    uint64_t *primes = dc->qVec;
    int set = threadIdx.x / radix;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
         i += blockDim.x * gridDim.x) {
        // size of a block
        uint64_t local[8];
        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t *a_np = op + np_idx * N;
        const uint64_t *prime_table = primes;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
            local[j] = *(a_np + N_init + t / 4 * j);
        }
        int tw_idx = m + m_idx;
        const uint64_t *W = dc->qRootPowsDivTwo[np_idx];
        const uint64_t *W_ = dc->qRootPowsDivTwoShoup[np_idx];
        for (int j = 0; j < 4; j++) {
            butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx],
                           prime);
        }
        for (int j = 0; j < 2; j++) {
            butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                           W_[2 * tw_idx + j], prime);
            butt_ntt_local(local[4 * j + 1], local[4 * j + 3],
                           W[2 * tw_idx + j], W_[2 * tw_idx + j], prime);
        }
        for (int j = 0; j < 4; j++) {
            butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                           W_[4 * tw_idx + j], prime);
        }
        for (int j = 0; j < 8; j++) {
            temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
        }
        int tail = 0;
        __syncthreads();
#pragma unroll
        for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
            for (int l = 0; l < 8; l++) {
                local[l] = temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 +
                                (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            for (int j2 = 0; j2 < 4; j2++) {
                butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2],
                               W_[tw_idx2], prime);
            }
            for (int j2 = 0; j2 < 2; j2++) {
                butt_ntt_local(local[4 * j2], local[4 * j2 + 2],
                               W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2],
                               prime);
                butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                               W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2],
                               prime);
            }
            for (int j2 = 0; j2 < 4; j2++) {
                butt_ntt_local(local[2 * j2], local[2 * j2 + 1],
                               W[4 * tw_idx2 + j2], W_[4 * tw_idx2 + j2],
                               prime);
            }

            for (int l = 0; l < 8; l++) {
                temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                    local[l];
            }
            if (j == t / 8)
                tail = 1;
            if (j == t / 16)
                tail = 2;
            __syncthreads();
        }
        if (tail == 1) {
            for (int l = 0; l < 8; l++) {
                local[l] = temp[set * 8 * radix + 8 * t_idx + l];
            }
            int tw_idx2 = t * tw_idx + 4 * t_idx;
            butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                           prime);
            butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                           prime);
            for (int l = 0; l < 8; l++) {
                temp[set * 8 * radix + 8 * t_idx + l] = local[l];
            }
        } else if (tail == 2) {
            for (int l = 0; l < 8; l++) {
                local[l] = temp[set * 8 * radix + 8 * t_idx + l];
            }
            int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
            butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
            butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                           prime);
            butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                           prime);
            butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                           W_[2 * tw_idx2 + 1], prime);
            butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                           W_[2 * tw_idx2 + 2], prime);
            butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                           W_[2 * tw_idx2 + 3], prime);
            for (int l = 0; l < 8; l++) {
                temp[set * 8 * radix + 8 * t_idx + l] = local[l];
            }
        }
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
            for (int k = 0; k < 3; k++) {
                if (local[j] >= prime)
                    local[j] -= prime;
            }
        }
        for (int j = 0; j < 8; j++) {
            *(a_np + N_init + t / 4 * j) = local[j];
        }
    }
}

__global__ void Intt8PointPerThreadPhase2OoP(DeviceContext *dc,
                                             const uint64_t *in, uint64_t *out,
                                             const int m, const int num_prime,
                                             const int N,
                                             const int start_prime_idx,
                                             const int radix) {
    extern __shared__ uint64_t temp[];
    int set = threadIdx.x / radix;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
         i += blockDim.x * gridDim.x) {
        // size of a block
        uint64_t local[8];
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        const uint64_t *in_addr = in + np_idx * N;
        uint64_t *out_addr = out + np_idx * N;
        const uint64_t *prime_table = dc->qVec;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t_idx;
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            temp[set * 8 * radix + t_idx + t / 4 * j] =
                *(in_addr + N_init + t / 4 * j);
        }
        __syncthreads();
        for (int l = 0; l < 8; l++) {
            local[l] = temp[set * 8 * radix + 8 * t_idx + l];
        }
        int tw_idx = m + m_idx;
        int tw_idx2 = (t / 4) * tw_idx + t_idx;
        const uint64_t *WInv = dc->qRootPowsInvDivTwo[np_idx];
        const uint64_t *WInv_ = dc->qRootPowsInvDivTwoShoup[np_idx];
        for (int j = 0; j < 4; j++) {
            butt_intt_local(local[2 * j], local[2 * j + 1],
                            WInv[4 * tw_idx2 + j], WInv_[4 * tw_idx2 + j],
                            prime);
        }
        for (int j = 0; j < 2; j++) {
            butt_intt_local(local[4 * j], local[4 * j + 2],
                            WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j],
                            prime);
            butt_intt_local(local[4 * j + 1], local[4 * j + 3],
                            WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j],
                            prime);
        }
        for (int j = 0; j < 4; j++) {
            butt_intt_local(local[j], local[j + 4], WInv[tw_idx2],
                            WInv_[tw_idx2], prime);
        }
        int tail = 0;
        __syncthreads();
        for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 8 * t_idx + l] = local[l];
        }
        __syncthreads();
#pragma unroll
        for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
            for (int l = 0; l < 8; l++) {
                local[l] = temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 +
                                (k / 4) * l];
            }
            tw_idx2 = j * tw_idx + m_idx2;
            for (int l = 0; l < 4; l++) {
                butt_intt_local(local[2 * l], local[2 * l + 1],
                                WInv[4 * tw_idx2 + l], WInv_[4 * tw_idx2 + l],
                                prime);
            }
            for (int l = 0; l < 2; l++) {
                butt_intt_local(local[4 * l], local[4 * l + 2],
                                WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l],
                                prime);
                butt_intt_local(local[4 * l + 1], local[4 * l + 3],
                                WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l],
                                prime);
            }
            for (int l = 0; l < 4; l++) {
                butt_intt_local(local[l], local[l + 4], WInv[tw_idx2],
                                WInv_[tw_idx2], prime);
            }
            for (int l = 0; l < 8; l++) {
                temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                    local[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (tail == 1) {
            for (int j = 0; j < 8; j++) {
                local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
            }
            butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx],
                            prime);
        } else if (tail == 2) {
            for (int j = 0; j < 8; j++) {
                local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
            }
            butt_intt_local(local[0], local[2], WInv[2 * tw_idx],
                            WInv_[2 * tw_idx], prime);
            butt_intt_local(local[1], local[3], WInv[2 * tw_idx],
                            WInv_[2 * tw_idx], prime);
            butt_intt_local(local[4], local[6], WInv[2 * tw_idx + 1],
                            WInv_[2 * tw_idx + 1], prime);
            butt_intt_local(local[5], local[7], WInv[2 * tw_idx + 1],
                            WInv_[2 * tw_idx + 1], prime);
            butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx],
                            prime);
        }
        for (int j = 0; j < 8; j++) {
            *(out_addr + N_init + t / 4 * j) = local[j];
        }
    }
}

__global__ void Intt8PointPerThreadPhase1OoP(DeviceContext *dc,
                                             const uint64_t *in, uint64_t *out,
                                             const int m, const int num_prime,
                                             const int N,
                                             const int start_prime_idx, int pad,
                                             int radix) {
    extern __shared__ uint64_t temp[];
    int Warp_t = threadIdx.x % pad;
    int WarpID = threadIdx.x / pad;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
         i += blockDim.x * gridDim.x) {
        // size of a block
        uint64_t local[8];
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        const uint64_t *in_addr = in + np_idx * N;
        uint64_t *out_addr = out + np_idx * N;
        const uint64_t *prime_table = dc->qVec;
        const uint64_t *WInv = dc->qRootPowsInvDivTwo[np_idx];
        const uint64_t *WInv_ = dc->qRootPowsInvDivTwoShoup[np_idx];
        uint64_t prime = prime_table[np_idx];
        int N_init =
            2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
            local[j] = *(in_addr + N_init + t / 4 / radix * j);
        }
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;
        int tw_idx2 = radix * tw_idx + WarpID;
        for (int j = 0; j < 4; j++) {
            butt_intt_local(local[2 * j], local[2 * j + 1],
                            WInv[4 * tw_idx2 + j], WInv_[4 * tw_idx2 + j],
                            prime);
        }
        for (int j = 0; j < 2; j++) {
            butt_intt_local(local[4 * j], local[4 * j + 2],
                            WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j],
                            prime);
            butt_intt_local(local[4 * j + 1], local[4 * j + 3],
                            WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j],
                            prime);
        }
        for (int j = 0; j < 4; j++) {
            butt_intt_local(local[j], local[j + 4], WInv[tw_idx2],
                            WInv_[tw_idx2], prime);
        }
        for (int j = 0; j < 8; j++) {
            temp[Warp_t * (eradix + pad) + 8 * WarpID + j] = local[j];
        }
        int tail = 0;
        __syncthreads();
#pragma unroll
        for (int j = radix / 8, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = WarpID / (k / 4);
            int t_idx2 = WarpID % (k / 4);
            for (int l = 0; l < 8; l++) {
                local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k +
                                t_idx2 + (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            for (int l = 0; l < 4; l++) {
                butt_intt_local(local[2 * l], local[2 * l + 1],
                                WInv[4 * tw_idx2 + l], WInv_[4 * tw_idx2 + l],
                                prime);
            }
            for (int l = 0; l < 2; l++) {
                butt_intt_local(local[4 * l], local[4 * l + 2],
                                WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l],
                                prime);
                butt_intt_local(local[4 * l + 1], local[4 * l + 3],
                                WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l],
                                prime);
            }
            for (int l = 0; l < 4; l++) {
                butt_intt_local(local[l], local[l + 4], WInv[tw_idx2],
                                WInv_[tw_idx2], prime);
            }
            for (int l = 0; l < 8; l++) {
                temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                     (k / 4) * l] = local[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (radix < 8)
            tail = (radix == 4) ? 2 : 1;
        for (int l = 0; l < 8; l++) {
            local[l] = temp[Warp_t * (eradix + pad) + WarpID + radix * l];
        }
        if (tail == 1) {
            butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx],
                            prime);
        } else if (tail == 2) {
            butt_intt_local(local[0], local[2], WInv[2 * tw_idx],
                            WInv_[2 * tw_idx], prime);
            butt_intt_local(local[1], local[3], WInv[2 * tw_idx],
                            WInv_[2 * tw_idx], prime);
            butt_intt_local(local[4], local[6], WInv[2 * tw_idx + 1],
                            WInv_[2 * tw_idx + 1], prime);
            butt_intt_local(local[5], local[7], WInv[2 * tw_idx + 1],
                            WInv_[2 * tw_idx + 1], prime);
            butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx],
                            prime);
            butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx],
                            prime);
        }
        for (int j = 0; j < 8; j++) {
            if (local[j] >= prime)
                local[j] -= prime;
        }
        N_init =
            t / 4 / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
            *(out_addr + N_init + t / 4 * j) = local[j];
        }
    }
}