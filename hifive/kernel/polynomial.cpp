#include "hifive/kernel/polynomial.hpp"

void Add_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b) {
    for (int i = 0; i < params->limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[i * params->N + j] =
                (a[i * params->N + j] + b[i * params->N + j]) % qi;
        }
    }
}

void Sub_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b) {
    for (int i = 0; i < params->limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[i * params->N + j] =
                (a[i * params->N + j] + qi - b[i * params->N + j]) % qi;
        }
    }
}

void Mult_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b) {
    for (int i = 0; i < params->limb; i++) {
        uint64_t qi = params->ntt_params->q[i];
        for (int j = 0; j < params->N; j++) {
            dst[i * params->N + j] =
                (a[i * params->N + j] * b[i * params->N + j]) % qi;
        }
    }
}