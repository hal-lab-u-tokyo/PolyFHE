#include <cstdint>
#include <iostream>

#include "evaluate.h"
#include "polynomial.h"

extern __global__ void tensor_square_2x2_rns_poly();

namespace hifive {

void HAdd(const seal::SEALContext &context, Ciphertext &result,
          const Ciphertext &ct0, const Ciphertext &ct1,
          const gpu_ptr &modulus) {
    // Verify parameters.
    if (ct0.coeff_modulus_size() != ct1.coeff_modulus_size()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
    }
    if (ct0.poly_modulus_degree() != ct1.poly_modulus_degree()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
    }
    if ((!ct0.is_ntt_form()) | (!ct1.is_ntt_form())) {
        throw std::invalid_argument("Must be in NTT form");
    }

    const uint64_t poly_modulus_degree = ct0.poly_modulus_degree();
    const uint64_t coeff_modulus_size = ct0.coeff_modulus_size();
    dim3 blockSize(1024, 1, 1);
    dim3 gridSize(poly_modulus_degree / 1024, 1, 1);
    for (int i = 0; i < coeff_modulus_size; i++) {
        uint64_t *ct0_axi = ct0.ax() + i * poly_modulus_degree;
        uint64_t *ct1_axi = ct1.ax() + i * poly_modulus_degree;
        uint64_t *ct0_bxi = ct0.bx() + i * poly_modulus_degree;
        uint64_t *ct1_bxi = ct1.bx() + i * poly_modulus_degree;
        uint64_t *result_axi = result.ax() + i * poly_modulus_degree;
        uint64_t *result_bxi = result.bx() + i * poly_modulus_degree;
        poly_add_mod<<<blockSize, gridSize>>>(result_axi, ct0_axi, ct1_axi,
                                              modulus.get(), i);
        poly_add_mod<<<blockSize, gridSize>>>(result_bxi, ct0_bxi, ct1_bxi,
                                              modulus.get(), i);
    }
}

void HMult(const seal::SEALContext &context, Ciphertext &result,
           const Ciphertext &ct0, const Ciphertext &ct1,
           const gpu_ptr &modulus) {
    // Verify parameters.
    if (ct0.coeff_modulus_size() != ct1.coeff_modulus_size()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
    }
    if (ct0.poly_modulus_degree() != ct1.poly_modulus_degree()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
    }
    if ((!ct0.is_ntt_form()) | (!ct1.is_ntt_form())) {
        throw std::invalid_argument("Must be in NTT form");
    }

    const uint64_t poly_modulus_degree = ct0.poly_modulus_degree();
    const uint64_t coeff_modulus_size = ct0.coeff_modulus_size();
    dim3 blockSize(1024, 1, 1);
    dim3 gridSize(poly_modulus_degree / 1024, 1, 1);
    for (int i = 0; i < coeff_modulus_size; i++) {
        uint64_t *ct0_axi = ct0.ax() + i * poly_modulus_degree;
        uint64_t *ct1_axi = ct1.ax() + i * poly_modulus_degree;
        uint64_t *ct0_bxi = ct0.bx() + i * poly_modulus_degree;
        uint64_t *ct1_bxi = ct1.bx() + i * poly_modulus_degree;
        uint64_t *result_axi = result.ax() + i * poly_modulus_degree;
        uint64_t *result_bxi = result.bx() + i * poly_modulus_degree;
        poly_mult_mod<<<blockSize, gridSize>>>(result_axi, ct0_axi, ct1_axi,
                                               modulus.get(), i);
        poly_mult_mod<<<blockSize, gridSize>>>(result_bxi, ct0_bxi, ct1_bxi,
                                               modulus.get(), i);
    }
}

void NTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx) {
    const auto &s = cudaStreamPerThread;
    nwt_2d_radix8_forward_inplace(a.get(), d_ntt_table, batch_size, 0, s);
}

void iNTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx) {
    const auto &s = cudaStreamPerThread;
    nwt_2d_radix8_backward_inplace(a.get(), d_ntt_table, batch_size, 0, s);
}

} // namespace hifive