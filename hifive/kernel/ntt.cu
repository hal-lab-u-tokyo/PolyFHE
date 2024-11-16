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

__device__ void d_fntt_remaining2(DeviceContext *dc, uint64_t *buff,
                                  const int n, const int root_pow_idx,
                                  const int batch_idx, const int thread_idx) {
    for (int i = thread_idx; i < n / 2; i += n / 8) {
        d_ct_butterfly_shoup(buff[i * 2], buff[i * 2 + 1],
                             dc->qRootPows[batch_idx][root_pow_idx + i],
                             dc->qRootPowsShoup[batch_idx][root_pow_idx + i],
                             dc->qVec[batch_idx]);
    }
}

__device__ void d_fntt_remaining4(DeviceContext *dc, uint64_t *buff,
                                  const int n, const int root_pow_idx,
                                  const int batch_idx, const int thread_idx) {
    for (int i = thread_idx; i < n / 4; i += n / 8) {
        uint64_t local[4];
        for (int j = 0; j < 4; j++) {
            local[j] = buff[i * 4 + j];
        }
        const int tw_idx = root_pow_idx + i;
        d_fntt4_shoup(local, dc->qRootPows[batch_idx],
                      dc->qRootPowsShoup[batch_idx], tw_idx,
                      dc->qVec[batch_idx]);
        for (int j = 0; j < 4; j++) {
            buff[i * 4 + j] = local[j];
        }
    }
}

__device__ void LoadPhase1FromGmem(DeviceContext *dc, const int batch,
                                   uint64_t *smem, const uint64_t *gmem) {
    for (int i = threadIdx.x; i < batch * dc->N1 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N1 / 8);
        const int thread_idx = i % (dc->N1 / 8);
        const int smem_offset = thread_idx + i * dc->N1 / 8;
        const int gmem_offset = blockIdx.x + smem_offset * dc->N2;
        smem[smem_offset + batch_idx * dc->N1] =
            gmem[gmem_offset + batch_idx * dc->N];
    }
}

__device__ void StorePhase1ToGmem(DeviceContext *dc, const int batch,
                                  const uint64_t *smem, uint64_t *gmem) {
    for (int i = threadIdx.x; i < batch * dc->N1 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N1 / 8);
        const int thread_idx = i % (dc->N1 / 8);
        const int smem_offset = thread_idx + i * dc->N1 / 8;
        const int gmem_offset = blockIdx.x + smem_offset * dc->N2;
        gmem[gmem_offset + batch_idx * dc->N] =
            smem[smem_offset + batch_idx * dc->N1];
    }
}

__device__ void LoadPhase2FromGmem(DeviceContext *dc, const int batch,
                                   uint64_t *smem, const uint64_t *gmem) {
    for (int i = threadIdx.x; i < batch * dc->N2 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N2 / 8);
        const int thread_idx = i % (dc->N2 / 8);
        const int smem_offset = thread_idx + i * dc->N2 / 8;
        const int gmem_offset = blockIdx.x * dc->N2 + smem_offset;
        smem[smem_offset + batch_idx * dc->N2] =
            gmem[gmem_offset + batch_idx * dc->N];
    }
}

__device__ void StorePhase2ToGmem(DeviceContext *dc, const int batch,
                                  const uint64_t *smem, uint64_t *gmem) {
    for (int i = threadIdx.x; i < batch * dc->N2 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N2 / 8);
        const int thread_idx = i % (dc->N2 / 8);
        const int smem_offset = thread_idx + i * dc->N2 / 8;
        const int gmem_offset = blockIdx.x * dc->N2 + smem_offset;
        gmem[gmem_offset + batch_idx * dc->N] =
            smem[smem_offset + batch_idx * dc->N2];
    }
}

__device__ void NTTPhase1Batched(DeviceContext *dc, const int batch,
                                 uint64_t *smem) {
    uint64_t local[8];

    for (int i = threadIdx.x; i < batch * dc->N1 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N1 / 8);
        const int thread_idx = i % (dc->N1 / 8);
        uint64_t *smem_base = smem + batch_idx * dc->N1;

        int t = dc->N1;
        int last_m = 1;
        for (int m = 1; m <= dc->N1 / 8; m *= 8) {
            t = t / 8;
            const int i = thread_idx / t;
            const int j1 = 8 * i * t;
            const int j = j1 + thread_idx % t;
            for (int k = 0; k < 8; k++) {
                local[k] = smem_base[j + k * t];
            }
            d_fntt8_shoup(local, dc->qRootPows[batch_idx],
                          dc->qRootPowsShoup[batch_idx], i + m,
                          dc->qVec[batch_idx]);
            for (int k = 0; k < 8; k++) {
                smem_base[j + k * t] = local[k];
            }
            last_m = m;
            __syncthreads();
        }

        last_m = last_m * 8;
        const int remaining = dc->N1 / last_m;
        const int root_pow_idx = last_m;
        if (remaining == 2) {
            d_fntt_remaining2(dc, smem_base, dc->N1, root_pow_idx, batch_idx,
                              thread_idx);
        } else if (remaining == 4) {
            d_fntt_remaining4(dc, smem_base, dc->N1, root_pow_idx, batch_idx,
                              thread_idx);
        }
    }
}

__device__ void NTTPhase2Batched(DeviceContext *dc, const int batch,
                                 uint64_t *smem) {
    uint64_t local[8];
    for (int i = threadIdx.x; i < batch * dc->N2 / 8; i += blockDim.x) {
        const int batch_idx = i / (dc->N2 / 8);
        const int thread_idx = i % (dc->N2 / 8);
        uint64_t *smem_base = smem + batch_idx * dc->N2;

        int t = dc->N2;
        const int root_idx_base = dc->N1 + blockIdx.x;
        int last_m = 1;
        for (int m = 1; m <= dc->N2 / 8; m *= 8) {
            t = t / 8;
            const int i = thread_idx / t;
            const int j1 = 8 * i * t;
            const int j = j1 + thread_idx % t;
            const int root_idx = root_idx_base * m + i;
            for (int k = 0; k < 8; k++) {
                local[k] = smem_base[j + k * t];
            }
            d_fntt8_shoup(local, dc->qRootPows[batch_idx],
                          dc->qRootPowsShoup[batch_idx], root_idx,
                          dc->qVec[batch_idx]);
            for (int k = 0; k < 8; k++) {
                smem_base[j + k * t] = local[k];
            }
            last_m = m;
            __syncthreads();
        }

        last_m = last_m * 8;
        const int remaining = dc->N2 / last_m;
        const int root_pow_idx = root_idx_base * last_m;
        if (remaining == 2) {
            d_fntt_remaining2(dc, smem_base, dc->N2, root_pow_idx, batch_idx,
                              thread_idx);
        } else if (remaining == 4) {
            d_fntt_remaining4(dc, smem_base, dc->N2, root_pow_idx, batch_idx,
                              thread_idx);
        }
    }
}