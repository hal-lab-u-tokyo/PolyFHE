#include "hifive/kernel/ntt.hpp"

// Butterfly
/** Computer one butterfly in forward NTT
 * x[0] = x[0] + pow * x[1] % mod
 * x[1] = x[0] - pow * x[1] % mod
 */
__device__ void d_ct_butterfly(uint64_t &x, uint64_t &y, const uint64_t &tw,
                               const uint64_t &mod) {
    y = (y * tw) % mod;
    const uint64_t x_copy = x;
    x = (x + y) % mod;
    y = (x_copy - y + mod) % mod;
}

__device__ void d_ct_butterfly_shoup(uint64_t &x, uint64_t &y,
                                     const uint64_t &tw,
                                     const uint64_t &tw_shoup,
                                     const uint64_t &mod) {
    const uint64_t hi = __umul64hi(y, tw_shoup);
    uint64_t tw_y = y * tw - hi * mod;
    if (tw_y >= mod) {
        tw_y -= mod;
    }
    const uint64_t mod2 = 2 * mod;
    uint64_t tmp = x - mod2;
    x = tmp + (tmp >> 63) * mod2;
    if (x >= mod) {
        x -= mod;
    }
    y = x + mod2 - tw_y;
    if (y >= mod) {
        y -= mod;
    }
    x += tw_y;
    if (x >= mod) {
        x -= mod;
    }
    if (y >= mod) {
        y -= mod;
    }
}

/** Computer one butterfly in inverse NTT
 * x[0] = (x[0] + pow * x[1]) / 2 % mod
 * x[1] = (x[0] - pow * x[1]) / 2 % mod
 */
__device__ void d_gs_butterfly(uint64_t &x, uint64_t &y, const uint64_t &tw,
                               const uint64_t &mod) {
    const uint64_t x_copy = x;
    const uint64_t y_copy = y;
    x = (x_copy + y_copy) % mod;
    y = (((x_copy - y_copy + mod) % mod) * tw) % mod;
}

__device__ void d_fntt8(uint64_t *s, const uint64_t *tw, uint64_t tw_idx,
                        uint64_t mod) {
    // stage 1
    d_ct_butterfly(s[0], s[4], tw[tw_idx], mod);
    d_ct_butterfly(s[1], s[5], tw[tw_idx], mod);
    d_ct_butterfly(s[2], s[6], tw[tw_idx], mod);
    d_ct_butterfly(s[3], s[7], tw[tw_idx], mod);
    // stage 2
    d_ct_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    d_ct_butterfly(s[1], s[3], tw[2 * tw_idx], mod);
    d_ct_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    d_ct_butterfly(s[5], s[7], tw[2 * tw_idx + 1], mod);
    // stage 3
    d_ct_butterfly(s[0], s[1], tw[4 * tw_idx], mod);
    d_ct_butterfly(s[2], s[3], tw[4 * tw_idx + 1], mod);
    d_ct_butterfly(s[4], s[5], tw[4 * tw_idx + 2], mod);
    d_ct_butterfly(s[6], s[7], tw[4 * tw_idx + 3], mod);
}

__device__ void d_fntt8_shoup(uint64_t *s, const uint64_t *tw,
                              const uint64_t *tw_shoup, uint64_t tw_idx,
                              uint64_t mod) {
    // stage 1
    d_ct_butterfly_shoup(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
    d_ct_butterfly_shoup(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
    d_ct_butterfly_shoup(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
    d_ct_butterfly_shoup(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);

    // stage 2
    d_ct_butterfly_shoup(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    d_ct_butterfly_shoup(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
    d_ct_butterfly_shoup(s[4], s[6], tw[2 * tw_idx + 1],
                         tw_shoup[2 * tw_idx + 1], mod);
    d_ct_butterfly_shoup(s[5], s[7], tw[2 * tw_idx + 1],
                         tw_shoup[2 * tw_idx + 1], mod);

    // stage 3
    d_ct_butterfly_shoup(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
    d_ct_butterfly_shoup(s[2], s[3], tw[4 * tw_idx + 1],
                         tw_shoup[4 * tw_idx + 1], mod);
    d_ct_butterfly_shoup(s[4], s[5], tw[4 * tw_idx + 2],
                         tw_shoup[4 * tw_idx + 2], mod);
    d_ct_butterfly_shoup(s[6], s[7], tw[4 * tw_idx + 3],
                         tw_shoup[4 * tw_idx + 3], mod);
}

__device__ void d_fntt4_shoup(uint64_t *s, const uint64_t *tw,
                              const uint64_t *tw_shoup, uint64_t tw_idx,
                              uint64_t mod) {
    // stage 1
    d_ct_butterfly_shoup(s[0], s[2], tw[tw_idx], tw_shoup[tw_idx], mod);
    d_ct_butterfly_shoup(s[1], s[3], tw[tw_idx], tw_shoup[tw_idx], mod);
    // stage 2
    d_ct_butterfly_shoup(s[0], s[1], tw[tw_idx * 2], tw_shoup[tw_idx * 2], mod);
    d_ct_butterfly_shoup(s[2], s[3], tw[tw_idx * 2 + 1],
                         tw_shoup[tw_idx * 2 + 1], mod);
}

__device__ void d_intt8(uint64_t *s, const uint64_t *tw, uint64_t tw_idx,
                        uint64_t mod) {
    // stage 1
    d_gs_butterfly(s[0], s[1], tw[4 * tw_idx], mod);
    d_gs_butterfly(s[2], s[3], tw[4 * tw_idx + 1], mod);
    d_gs_butterfly(s[4], s[5], tw[4 * tw_idx + 2], mod);
    d_gs_butterfly(s[6], s[7], tw[4 * tw_idx + 3], mod);

    // stage 2
    d_gs_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    d_gs_butterfly(s[1], s[3], tw[2 * tw_idx], mod);
    d_gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    d_gs_butterfly(s[5], s[7], tw[2 * tw_idx + 1], mod);

    // stage 3
    d_gs_butterfly(s[0], s[4], tw[tw_idx], mod);
    d_gs_butterfly(s[1], s[5], tw[tw_idx], mod);
    d_gs_butterfly(s[2], s[6], tw[tw_idx], mod);
    d_gs_butterfly(s[3], s[7], tw[tw_idx], mod);
}

__device__ void d_intt4(uint64_t *s, const uint64_t *tw, uint64_t tw_idx,
                        uint64_t mod) {
    // stage 1
    d_gs_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    d_gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    // stage 2
    d_gs_butterfly(s[0], s[4], tw[tw_idx], mod);
    d_gs_butterfly(s[2], s[6], tw[tw_idx], mod);
}

__device__ void d_fntt_remaining2(uint64_t *buff, const int n,
                                  const int root_pow_idx, NTTParams *params) {
    const int thread_idx = threadIdx.x % (n / 8);
    for (int i = thread_idx; i < n / 2; i += n / 8) {
        d_ct_butterfly_shoup(
            buff[i * 2], buff[i * 2 + 1], params->roots_pow[root_pow_idx + i],
            params->roots_pow_shoup[root_pow_idx + i], params->q);
    }
}

__device__ void d_fntt_remaining4(uint64_t *buff, const int n,
                                  const int root_pow_idx, NTTParams *params) {
    const int thread_idx = threadIdx.x % (n / 8);
    for (int i = thread_idx; i < n / 4; i += n / 8) {
        uint64_t local[4];
        for (int j = 0; j < 4; j++) {
            local[j] = buff[i * 4 + j];
        }
        const int tw_idx = root_pow_idx + i;
        d_fntt4_shoup(local, params->roots_pow, params->roots_pow_shoup, tw_idx,
                      params->q);
        for (int j = 0; j < 4; j++) {
            buff[i * 4 + j] = local[j];
        }
    }
}

__device__ void NTTPhase1Batched(uint64_t *buff, NTTParams *params) {
    uint64_t local[8];

    for (int i = threadIdx.x; i < params->batch * params->n1 / 8;
         i += blockDim.x) {
        const int batch_idx = i / (params->n1 / 8);
        const int thread_idx = i % (params->n1 / 8);
        uint64_t *buff_base = buff + batch_idx * params->n1;

        int t = params->n1;
        int last_m = 1;
        for (int m = 1; m <= params->n1 / 8; m *= 8) {
            t = t / 8;
            const int i = thread_idx / t;
            const int j1 = 8 * i * t;
            const int j = j1 + thread_idx % t;
            for (int k = 0; k < 8; k++) {
                local[k] = buff_base[j + k * t];
            }
            d_fntt8_shoup(local, params->roots_pow, params->roots_pow_shoup,
                          i + m, params->q);
            for (int k = 0; k < 8; k++) {
                buff_base[j + k * t] = local[k];
            }
            last_m = m;
            __syncthreads();
        }

        last_m = last_m * 8;
        const int remaining = params->n1 / last_m;
        const int root_pow_idx = last_m;
        if (remaining == 2) {
            d_fntt_remaining2(buff_base, params->n1, root_pow_idx, params);
        } else if (remaining == 4) {
            d_fntt_remaining4(buff_base, params->n1, root_pow_idx, params);
        }
    }
}

__device__ void NTTPhase2Batched(uint64_t *buff, NTTParams *params) {
    uint64_t local[8];
    for (int i = threadIdx.x; i < params->batch * params->n2 / 8;
         i += blockDim.x) {
        const int batch_idx = i / (params->n2 / 8);
        const int thread_idx = i % (params->n2 / 8);
        uint64_t *buff_base = buff + batch_idx * params->n2;

        int t = params->n2;
        const int root_idx_base = params->n1 + blockIdx.x;
        int last_m = 1;
        for (int m = 1; m <= params->n2 / 8; m *= 8) {
            t = t / 8;
            const int i = thread_idx / t;
            const int j1 = 8 * i * t;
            const int j = j1 + thread_idx % t;
            const int root_idx = root_idx_base * m + i;
            for (int k = 0; k < 8; k++) {
                local[k] = buff_base[j + k * t];
            }
            d_fntt8_shoup(local, params->roots_pow, params->roots_pow_shoup,
                          root_idx, params->q);
            for (int k = 0; k < 8; k++) {
                buff_base[j + k * t] = local[k];
            }
            last_m = m;
            __syncthreads();
        }

        last_m = last_m * 8;
        const int remaining = params->n2 / last_m;
        const int root_pow_idx = root_idx_base * last_m;
        if (remaining == 2) {
            d_fntt_remaining2(buff_base, params->n2, root_pow_idx, params);
        } else if (remaining == 4) {
            d_fntt_remaining4(buff_base, params->n2, root_pow_idx, params);
        }
    }
}