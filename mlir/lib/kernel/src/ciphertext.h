#pragma once

#include <cstdint>

#include "gpu_utils.h"

// Phantom
#include "phantom.h"

namespace hifive {

class Ciphertext {
public:
    Ciphertext(const PhantomCiphertext &src);
    Ciphertext(uint64_t poly_modulus_degree, uint64_t coeff_modulus_size);
    ~Ciphertext() = default;

    void CopyBack(PhantomCiphertext &dst) const;

    uint64_t poly_modulus_degree() const { return poly_modulus_degree_; }
    uint64_t coeff_modulus_size() const { return coeff_modulus_size_; }
    bool is_ntt_form() const { return is_ntt_form_; }

    uint64_t *ax() const { return ax_.get(); }
    uint64_t *bx() const { return bx_.get(); }

private:
    uint64_t poly_modulus_degree_ = 0;
    uint64_t coeff_modulus_size_ = 0;
    bool is_ntt_form_ = true;
    gpu_ptr ax_;
    gpu_ptr bx_;
};

} // namespace hifive