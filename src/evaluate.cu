#include "evaluate.h"

#include <iostream>

__global__ void poly_add(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b, DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] = (v0 + v1) > mod ? (v0 + v1 - mod) : (v0 + v1);
}
namespace hifive {

void Evaluator::Add(const PhantomContext &context, PhantomCiphertext &result, const PhantomCiphertext &ct0, const PhantomCiphertext &ct1) {
    if (ct0.chain_index() != ct1.chain_index()){
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (ct0.is_ntt_form() != ct1.is_ntt_form()) {
        throw std::invalid_argument("NTT form mismatch");
    }
    if (std::abs(ct0.scale() - ct1.scale()) > 1e-6) {
        throw std::invalid_argument("scale mismatch");
    }
    if (ct0.size() != ct1.size()) {
        throw std::invalid_argument("poly number mismatch");
    }
    std::cout << "## HAdd" << std::endl;

    auto &context_data = context.get_context_data(ct0.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &plain_modulus = parms.plain_modulus();
    auto coeff_modulus_size = coeff_modulus.size();
    auto poly_degree = context.gpu_rns_tables().n();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto rns_coeff_count = poly_degree * coeff_modulus_size;
    uint64_t *ct0_ax = ct0.data();
    uint64_t *ct0_bx = ct0.data() + rns_coeff_count;
    uint64_t *ct1_ax = ct1.data(); 
    uint64_t *ct1_bx = ct1.data() + rns_coeff_count;

    dim3 blockSize(1024, 1, 1);
    dim3 gridSize(poly_degree / 1024, 1, 1);
    for (int i = 0; i < coeff_modulus_size; i++){
        uint64_t *ct0_axi = ct0_ax + i * poly_degree;
        uint64_t *ct1_axi = ct1_ax + i * poly_degree;
        uint64_t *ct0_bxi = ct0_bx + i * poly_degree;
        uint64_t *ct1_bxi = ct1_bx + i * poly_degree;
        poly_add<<<blockSize, gridSize>>>(ct0_axi, ct0_axi, ct1_axi, base_rns, i);
        poly_add<<<blockSize, gridSize>>>(ct0_bxi, ct0_bxi ,ct1_bxi, base_rns, i);
    }
}

}