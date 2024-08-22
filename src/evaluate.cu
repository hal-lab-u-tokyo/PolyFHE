#include <iostream>

#include "evaluate.h"

__global__ void poly_add(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                         DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] = (v0 + v1) > mod ? (v0 + v1 - mod) : (v0 + v1);
}

__global__ void poly_mult(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                          DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] = (v0 * v1) % mod;
    if (idx == 0 && limb == 0) {
        printf("v0: %lu, v1: %lu, mod: %lu, d_out: %lu\n", v0, v1, mod,
               d_out[idx]);
    }
}

__global__ void poly_mult_accum(uint64_t *d_out, uint64_t *d_a, uint64_t *d_b,
                                DModulus *modulus, int limb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v0 = d_a[idx];
    uint64_t v1 = d_b[idx];
    uint64_t mod = modulus[limb].value();
    d_out[idx] += ((v0 * v1) % mod);
    d_out[idx] %= mod;
    if (idx == 0 && limb == 0) {
        printf("v0: %lu, v1: %lu, mod: %lu, d_out: %lu\n", v0, v1, mod,
               d_out[idx]);
    }
}
namespace hifive {

Evaluator::Evaluator() { gpu_context_ = std::make_unique<GPUContext>(); }

void Evaluator::Add(const PhantomContext &context, PhantomCiphertext &result,
                    const PhantomCiphertext &ct0,
                    const PhantomCiphertext &ct1) {
    if (ct0.chain_index() != ct1.chain_index()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
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
    uint64_t *result_ax = result.data();
    uint64_t *result_bx = result.data() + rns_coeff_count;
    uint64_t max_threads = gpu_context_->max_threads_per_block_;

    dim3 blockSize(max_threads, 1, 1);
    dim3 gridSize(poly_degree / max_threads, 1, 1);
    for (int i = 0; i < coeff_modulus_size; i++) {
        uint64_t *ct0_axi = ct0_ax + i * poly_degree;
        uint64_t *ct1_axi = ct1_ax + i * poly_degree;
        uint64_t *ct0_bxi = ct0_bx + i * poly_degree;
        uint64_t *ct1_bxi = ct1_bx + i * poly_degree;
        uint64_t *result_axi = result_ax + i * poly_degree;
        uint64_t *result_bxi = result_bx + i * poly_degree;
        poly_add<<<blockSize, gridSize>>>(result_axi, ct0_axi, ct1_axi,
                                          base_rns, i);
        poly_add<<<blockSize, gridSize>>>(result_bxi, ct0_bxi, ct1_bxi,
                                          base_rns, i);
    }
}

void Evaluator::Mult(const PhantomContext &context, PhantomCiphertext &result,
                     const PhantomCiphertext &ct0,
                     const PhantomCiphertext &ct1) {
    // TODO: Support the case `result` = (`ct0` or `ct1`)
    // Currently, we override `result`
    if (ct0.parms_id() != ct1.parms_id()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
    }
    if (ct0.chain_index() != ct1.chain_index()) {
        throw std::invalid_argument("ct0 and ct1 parameter mismatch");
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

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(ct0.chain_index());
    auto &parms = context_data.parms();
    auto base_rns = context.gpu_rns_tables().modulus();
    size_t coeff_modulus_size = parms.coeff_modulus().size();
    size_t poly_degree = parms.poly_modulus_degree();
    uint32_t ct0_size = ct0.size();
    uint32_t ct1_size = ct1.size();
    auto rns_coeff_count = poly_degree * coeff_modulus_size;

    uint32_t dest_size = ct0_size + ct1_size - 1;
    result.resize(context, ct0.chain_index(), dest_size, cudaStreamPerThread);

    // d0 = ct0_ax * ct1_ax
    // d2 = ct0_bx * ct1_bx
    // d1 = ct0_ax * ct1_bx + ct0_bx * ct1_ax
    uint64_t *ct0_ax = ct0.data();
    uint64_t *ct0_bx = ct0.data() + rns_coeff_count;
    uint64_t *ct1_ax = ct1.data();
    uint64_t *ct1_bx = ct1.data() + rns_coeff_count;
    uint64_t *result_d0 = result.data();
    uint64_t *result_d1 = result.data() + rns_coeff_count;
    uint64_t *result_d2 = result.data() + 2 * rns_coeff_count;
    uint64_t max_threads = gpu_context_->max_threads_per_block_;
    dim3 blockSize(max_threads, 1, 1);
    dim3 gridSize(poly_degree / max_threads, 1, 1);
    for (int i = 0; i < coeff_modulus_size; i++) {
        uint64_t *ct0_axi = ct0_ax + i * poly_degree;
        uint64_t *ct1_axi = ct1_ax + i * poly_degree;
        uint64_t *ct0_bxi = ct0_bx + i * poly_degree;
        uint64_t *ct1_bxi = ct1_bx + i * poly_degree;
        uint64_t *result_d0i = result_d0 + i * poly_degree;
        uint64_t *result_d1i = result_d1 + i * poly_degree;
        uint64_t *result_d2i = result_d2 + i * poly_degree;
        poly_mult<<<blockSize, gridSize>>>(result_d0i, ct0_axi, ct1_axi,
                                           base_rns, i);
        poly_mult<<<blockSize, gridSize>>>(result_d2i, ct0_bxi, ct1_bxi,
                                           base_rns, i);
        poly_mult<<<blockSize, gridSize>>>(result_d1i, ct0_axi, ct1_bxi,
                                           base_rns, i);
        poly_mult_accum<<<blockSize, gridSize>>>(result_d1i, ct0_bxi, ct1_axi,
                                                 base_rns, i);
    }
    result.set_scale(ct0.scale() * ct1.scale());
}

} // namespace hifive