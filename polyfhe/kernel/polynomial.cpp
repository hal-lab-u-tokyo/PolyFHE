#include "polyfhe/kernel/polynomial.hpp"

#include <cstdint>
#include <iostream>

#include "polyfhe/kernel/device_context.hpp"

void Add_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
           const int start_limb, const int end_limb) {
    for (int i = start_limb; i < end_limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[(i - start_limb) * params->N + j] =
                (a[i * params->N + j] + b[i * params->N + j]) % qi;
        }
    }
}

void Sub_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
           const int start_limb, const int end_limb) {
    for (int i = start_limb; i < end_limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[(i - start_limb) * params->N + j] =
                (a[i * params->N + j] + qi - b[i * params->N + j]) % qi;
        }
    }
}

void Mult_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b,
            const int start_limb, const int end_limb) {
    for (int i = start_limb; i < end_limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[(i - start_limb) * params->N + j] =
                (a[i * params->N + j] * b[i * params->N + j]) % qi;
        }
    }
}

void ModUp_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
             const int end_limb) {
    for (int k = 0; k < params->K; k++) {
        uint64_t *dst_k = dst + (params->limb + k) * params->N;
        for (int n = 0; n < params->N; n++) {
            uint64_t sum = 0;
            for (int l = start_limb; l < end_limb; l++) {
                sum += src[l * params->N + n];
            }
            dst_k[n] = sum;
        }
    }
}

void NTT_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
           const int end_limb) {
    for (int l = start_limb; l < end_limb; l++) {
        const uint64_t q = params->ntt_params->q[l];
        uint64_t *dst_l = dst + (l - start_limb) * params->N;
        uint64_t *src_l = src + l * params->N;
        // copy src to dst
        for (int i = 0; i < params->N; i++) {
            dst_l[i] = src_l[i];
        }
        const uint64_t root = params->ntt_params->root[l];
        uint64_t t = params->N;
        uint64_t j1, j2;
        for (int m = 1; m < params->N; m *= 2) {
            t = t / 2;
            for (int i = 0; i < m; i++) {
                j1 = 2 * i * t;
                j2 = j1 + t - 1;
                for (int j = j1; j <= j2; j++) {
                    uint64_t u = dst_l[j];
                    uint64_t v = (dst_l[j + t] *
                                  params->ntt_params->roots_pow[l][m + i]) %
                                 q;
                    dst_l[j] = (u + v) % q;
                    dst_l[j + t] = (u - v + q) % q;
                }
            }
        }
    }
}

// N2 * N1-point NTT
void NTTPhase1_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb) {
    uint64_t *buff = new uint64_t[params->n1];

    for (int l = start_limb; l < end_limb; l++) {
        uint64_t *dst_l = dst + l * params->N;
        uint64_t *src_l = src + (l - start_limb) * params->N;
        const uint64_t q = params->ntt_params->q[l];
        for (int iter = 0; iter < params->n2; iter++) {
            // copy to buff
            for (int i = 0; i < params->n1; i++) {
                buff[i] = src_l[i * params->n2 + iter];
            }

            uint64_t t = params->n1;
            uint64_t j1, j2;
            for (int m = 1; m < params->n1; m *= 2) {
                t = t / 2;
                for (int i = 0; i < m; i++) {
                    j1 = 2 * i * t;
                    j2 = j1 + t - 1;
                    for (int j = j1; j <= j2; j++) {
                        uint64_t u = buff[j];
                        uint64_t v = (buff[j + t] *
                                      params->ntt_params->roots_pow[l][m + i]) %
                                     q;
                        buff[j] = (u + v) % q;
                        buff[j + t] = (u - v + q) % q;
                    }
                }
            }

            // copy back
            for (int i = 0; i < params->n1; i++) {
                dst_l[i * params->n2 + iter] = buff[i];
            }
        }
    }
}

// N1 * N2-point NTT
void NTTPhase2_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb) {
    uint64_t *buff = new uint64_t[params->n2];

    for (int l = start_limb; l < end_limb; l++) {
        uint64_t *src_l = src + l * params->N;
        uint64_t *dst_l = dst + l * params->N;
        const uint64_t q = params->ntt_params->q[l];

        for (int iter = 0; iter < params->n1; iter++) {
            // copy to buff
            for (int i = 0; i < params->n2; i++) {
                buff[i] = src_l[iter * params->n2 + i];
            }

            uint64_t t = params->n2;
            uint64_t j1, j2;
            for (int m = 1; m < params->n2; m *= 2) {
                t = t / 2;
                for (int i = 0; i < m; i++) {
                    j1 = 2 * i * t;
                    j2 = j1 + t - 1;
                    for (int j = j1; j <= j2; j++) {
                        uint64_t u = buff[j];
                        int root_idx = params->n1 * m + i + iter * m;
                        uint64_t v =
                            (buff[j + t] *
                             params->ntt_params->roots_pow[l][root_idx]) %
                            q;
                        buff[j] = (u + v) % q;
                        buff[j + t] = (u - v + q) % q;
                    }
                }
            }

            // copy back
            for (int i = 0; i < params->n2; i++) {
                dst_l[iter * params->n2 + i] = buff[i];
            }
        }
    }
}

void iNTT_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
            const int end_limb) {
    for (int l = start_limb; l < end_limb; l++) {
        const uint64_t q = params->ntt_params->q[l];
        uint64_t *dst_l = dst + (l - start_limb) * params->N;
        uint64_t *src_l = src + l * params->N;
        // copy src to dst
        for (int i = 0; i < params->N; i++) {
            dst_l[i] = src_l[i];
        }
        uint64_t t = 1;
        uint64_t j1, j2, h;
        for (int m = params->N; m > 1; m >>= 1) {
            j1 = 0;
            h = m >> 1;
            for (int i = 0; i < h; i++) {
                j2 = j1 + t - 1;
                for (int j = j1; j <= j2; j++) {
                    uint64_t u = dst_l[j];
                    uint64_t v = dst_l[j + t];
                    uint64_t root = params->ntt_params->roots_pow_inv[l][h + i];
                    dst_l[j] = (u + v) % q;
                    dst_l[j + t] = (((u + q - v) % q) * root) % q;
                }
                j1 += t << 1;
            }
            t <<= 1;
        }
        for (int i = 0; i < params->N; i++) {
            dst_l[i] = (dst_l[i] * params->ntt_params->N_inv[l]) % q;
        }
    }
}