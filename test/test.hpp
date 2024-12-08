#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include "hifive/core/logger.hpp"
#include "hifive/kernel/device_context.hpp"
#include "hifive/kernel/polynomial.hpp"

// Test functions
void test_poly_add(FHEContext &context, const int N, const int L,
                   const int block_x, const int block_y);
void test_poly_mult(FHEContext &context, const int N, const int L,
                    const int block_x, const int block_y);
void test_ntt(FHEContext &context, const int N, const int N1, const int N2,
              const int L);

// Utils
std::pair<uint64_t *, uint64_t *> create_random_polynomial(
    const int N, const int L, const uint64_t *moduli);
std::pair<uint64_t *, uint64_t *> create_linear_polynomial(const int N,
                                                           const int L);
std::pair<uint64_t *, uint64_t *> create_zeros_polynomial(const int N,
                                                          const int L);