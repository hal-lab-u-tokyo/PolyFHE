#include "ciphertext.h"

namespace hifive {

Ciphertext::Ciphertext(const PhantomCiphertext &src) {
    poly_modulus_degree_ = src.poly_modulus_degree();
    coeff_modulus_size_ = src.coeff_modulus_size();

    const uint64_t poly_size =
        src.coeff_modulus_size() * src.poly_modulus_degree();
    ax_ = hifive::gpu_ptr((uint64_t *) src.data(), poly_size);
    bx_ = hifive::gpu_ptr((uint64_t *) src.data() + poly_size, poly_size);
}

Ciphertext::Ciphertext(uint64_t poly_modulus_degree,
                       uint64_t coeff_modulus_size)
    : poly_modulus_degree_(poly_modulus_degree),
      coeff_modulus_size_(coeff_modulus_size) {
    const uint64_t poly_size = poly_modulus_degree_ * coeff_modulus_size_;
    ax_ = make_gpu_ptr(poly_size);
    bx_ = make_gpu_ptr(poly_size);
}

void Ciphertext::CopyBack(PhantomCiphertext &dst) const {
    const uint64_t poly_size = poly_modulus_degree_ * coeff_modulus_size_;
    checkCudaErrors(cudaMemcpy(dst.data(), ax_.get(),
                               poly_size * sizeof(uint64_t),
                               cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.data() + poly_size, bx_.get(),
                               poly_size * sizeof(uint64_t),
                               cudaMemcpyDeviceToDevice));
}

} // namespace hifive