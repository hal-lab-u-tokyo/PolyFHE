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

void iNTT_h(Params *params, uint64_t *dst, uint64_t *src, const int start_limb,
            const int end_limb) {
    for (int batch_idx = start_limb; batch_idx < end_limb; batch_idx++) {
        const uint64_t q = params->ntt_params->q[batch_idx];
        uint64_t *dst_l = dst + (batch_idx - start_limb) * params->N;
        uint64_t *src_l = src + batch_idx * params->N;
        // copy src to dst
        for (int i = 0; i < params->N; i++) {
            dst_l[i] = src_l[i];
        }
        uint64_t t, step;
        for (int m = params->N / 2; m >= 1; m /= 2) {
            step = m * 2;
            t = params->N / step;
            for (int k = 0; k < params->N; k += step) {
                for (int j = 0; j < m; j++) {
                    uint64_t S =
                        params->ntt_params->roots_pow_inv[batch_idx][t * j];
                    uint64_t U = dst_l[k + j];
                    uint64_t V = dst_l[k + j + m];
                    dst_l[k + j] = (U + V) % q;
                    dst_l[k + j + m] = (((U - V + q) % q) * S) % q;
                }
            }
        }
        const uint64_t n_inv = params->ntt_params->N_inv[batch_idx];
        for (int i = 0; i < params->N; i++) {
            dst_l[i] = (dst_l[i] * n_inv) % q;
        }
    }
}

// N2 * N1-point NTT
void NTTPhase1_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb) {
    uint64_t *buff = new uint64_t[params->n1];
    for (int batch_idx = start_limb; batch_idx < end_limb; batch_idx++) {
        uint64_t q = params->ntt_params->q[batch_idx];
        for (int iter = 0; iter < params->n2; iter++) {
            // copy
            for (int i = 0; i < params->n1; i++) {
                buff[i] = src[iter * params->n1 + i];
            }

            uint64_t t = params->n1;
            // step. m is butterfly width
            for (int m = 1; m < params->n1; m *= 2) {
                t = t / 2;
                // group
                for (int k = 0; k < params->n1; k += 2 * m) {
                    for (int j = 0; j < m; j++) {
                        const int rootidx = t * j * params->n2;
                        // std::cout << "rootidx=" << rootidx << ", m=" << m
                        //           << ", j=" << j << ", k=" << k << std::endl;
                        uint64_t S =
                            params->ntt_params->roots_pow[batch_idx][rootidx];
                        uint64_t U = buff[k + j];
                        uint64_t V = (buff[k + j + m] * S) % q;
                        buff[k + j] = (U + V) % q;
                        buff[k + j + m] = (U - V + q) % q;
                    }
                }
            }

            // copy back
            for (int i = 0; i < params->n1; i++) {
                dst[iter * params->n1 + i] = buff[i];
            }
        }
    }
}

// N1 * N2-point NTT
void NTTPhase2_h(Params *params, uint64_t *dst, uint64_t *src,
                 const int start_limb, const int end_limb) {
    for (int batch_idx = start_limb; batch_idx < end_limb; batch_idx++) {
        uint64_t q = params->ntt_params->q[batch_idx];
        uint64_t *buff = new uint64_t[params->n2];
        for (int iter = 0; iter < params->n1; iter++) {
            // copy to buff
            for (int i = 0; i < params->n2; i++) {
                buff[i] = src[iter + params->n1 * i];
            }

            uint64_t t = params->n2;
            uint64_t j1, j2;
            for (int m = 1; m < params->n2; m *= 2) {
                t = t / 2;
                for (int k = 0; k < params->n2; k += 2 * m) {
                    for (int j = 0; j < m; j++) {
                        const int rootidx =
                            t * j * params->n1 + iter * params->n2 / (2 * m);
                        // std::cout << "k=" << k << ", j=" << j << ", m=" << m
                        //           << ", rootidx=" << rootidx << std::endl;
                        uint64_t S =
                            params->ntt_params->roots_pow[batch_idx][rootidx];
                        uint64_t U = buff[k + j];
                        uint64_t V = (buff[k + j + m] * S) % q;
                        uint64_t V_ = buff[k + j + m];
                        buff[k + j] = (U + V) % q;
                        buff[k + j + m] = (U - V + q) % q;
                    }
                }
            }

            // copy back
            for (int i = 0; i < params->n2; i++) {
                dst[iter + params->n1 * i] = buff[i];
            }
        }
    }
}

void iNTTPhase2_h(Params *params, uint64_t *dst, uint64_t *src,
                  const int start_limb, const int end_limb) {
    uint64_t *buff = new uint64_t[params->n2];

    for (int batch_idx = start_limb; batch_idx < end_limb; batch_idx++) {
        uint64_t q = params->ntt_params->q[batch_idx];
        for (int iter = 0; iter < params->n1; iter++) {
            // copy to buff
            for (int i = 0; i < params->n2; i++) {
                buff[i] = src[iter + params->n1 * i];
            }

            uint64_t t, step;
            for (int m = params->n2 / 2; m >= 1; m /= 2) {
                step = m * 2;
                t = params->n2 / step;
                for (int k = 0; k < params->n2; k += step) {
                    for (int j = 0; j < m; j++) {
                        const int rootidx =
                            t * j * params->n1 + iter * params->n2 / (2 * m);
                        // std::cout << "m=" << m << ", k=" << k << ", j=" << j
                        //           << ", rootidx=" << rootidx << std::endl;
                        uint64_t S = params->ntt_params
                                         ->roots_pow_inv[batch_idx][rootidx];
                        uint64_t U = buff[k + j];
                        uint64_t V = buff[k + j + m];
                        buff[k + j] = (U + V) % q;
                        buff[k + j + m] = (((U - V + q) % q) * S) % q;
                    }
                }
            }

            // copy back
            for (int i = 0; i < params->n2; i++) {
                dst[iter + params->n1 * i] = buff[i];
            }
        }
    }
}

void iNTTPhase1_h(Params *params, uint64_t *dst, uint64_t *src,
                  const int start_limb, const int end_limb) {
    uint64_t *buff = new uint64_t[params->n1];
    for (int batch_idx = start_limb; batch_idx < end_limb; batch_idx++) {
        uint64_t q = params->ntt_params->q[batch_idx];
        const uint64_t n_inv = params->ntt_params->N_inv[batch_idx];

        for (int iter = 0; iter < params->n2; iter++) {
            // copy
            for (int i = 0; i < params->n1; i++) {
                buff[i] = src[iter * params->n1 + i];
            }

            uint64_t t, step;
            uint64_t j1, j2;
            for (int m = params->n1 / 2; m >= 1; m /= 2) {
                step = m * 2;
                t = params->n1 / step;
                for (int k = 0; k < params->n1; k += 2 * m) {
                    for (int j = 0; j < m; j++) {
                        const int rootidx = t * j * params->n2;
                        uint64_t S = params->ntt_params
                                         ->roots_pow_inv[batch_idx][rootidx];
                        uint64_t U = buff[k + j];
                        uint64_t V = buff[k + j + m];
                        buff[k + j] = (U + V) % q;
                        buff[k + j + m] = (((U - V + q) % q) * S) % q;
                    }
                }
            }

            // scale Ninv and copy back
            for (int i = 0; i < params->n1; i++) {
                dst[iter * params->n1 + i] = (buff[i] * n_inv) % q;
            }
        }
    }
}