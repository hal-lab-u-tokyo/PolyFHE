#include "hifive/kernel/polynomial.hpp"

void Add_h(Params *params, uint64_t *dst, uint64_t *a, uint64_t *b) {
    for (int i = 0; i < params->N * params->limb; i++) {
        dst[i] = a[i] + b[i];
    }
}