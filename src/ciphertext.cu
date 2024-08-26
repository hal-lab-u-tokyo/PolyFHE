#include "ciphertext.h"

// SEAL
#include "seal/seal.h"

namespace hifive {

Ciphertext::Ciphertext(const seal::Ciphertext &src) {
    poly_modulus_degree_ = src.poly_modulus_degree();
    coeff_modulus_size_ = src.coeff_modulus_size();

    const uint64_t poly_size =
        src.coeff_modulus_size() * src.poly_modulus_degree();
    ax_ = make_and_copy_gpu_ptr<uint64_t>((uint64_t *) src.data(0), poly_size);
    bx_ = make_and_copy_gpu_ptr<uint64_t>((uint64_t *) src.data(1), poly_size);
}

Ciphertext::Ciphertext(uint64_t poly_modulus_degree,
                       uint64_t coeff_modulus_size)
    : poly_modulus_degree_(poly_modulus_degree),
      coeff_modulus_size_(coeff_modulus_size) {
    const uint64_t poly_size = poly_modulus_degree_ * coeff_modulus_size_;
    ax_ = make_gpu_ptr<uint64_t>(poly_size);
    bx_ = make_gpu_ptr<uint64_t>(poly_size);
}

} // namespace hifive