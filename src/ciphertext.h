#pragma once

#include <cstdint>

#include "gpu_utils.h"

// SEAL
#include "seal/seal.h"

namespace hifive {

class Ciphertext {
public:
    Ciphertext(const seal::Ciphertext &src);
    Ciphertext(uint64_t poly_modulus_degree, uint64_t coeff_modulus_size);
    ~Ciphertext() = default;

private:
    uint64_t poly_modulus_degree_ = 0;
    uint64_t coeff_modulus_size_ = 0;
    bool is_ntt_form_ = true;
    gpu_ptr<uint64_t> ax_;
    gpu_ptr<uint64_t> bx_;
};

} // namespace hifive