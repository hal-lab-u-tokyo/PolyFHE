#include <cstdint>
#include <iostream>

#include "evaluate.h"
#include "polynomial.h"

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

void Evaluator::Relin(const PhantomContext &context, PhantomCiphertext &ct,
                      const PhantomRelinKey &relin_keys) {
    // Extract encryption parameters.
    const auto s = cudaStreamPerThread;
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped = ct.chain_index() - 1;
    phantom::DRNSTool &rns_tool =
        context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto modulus_QP = context.gpu_rns_tables().modulus();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;
    auto size_Ql_n = size_Ql * n;
    auto size_QlP_n = size_QlP * n;
    std::cout << "\tsize_Ql: " << size_Ql << std::endl;
    std::cout << "\tsize_Q: " << size_Q << std::endl;
    std::cout << "\tsize_QlP: " << size_QlP << std::endl;
    uint64_t *d0 = ct.data();
    uint64_t *d1 = ct.data() + size_Ql_n;
    uint64_t *d2 = ct.data() + 2 * size_Ql_n;

    // iNTT + ModUp + NTT
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    auto t_mod_up =
        phantom::util::make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    rns_tool.modup(t_mod_up.get(), d2, context.gpu_rns_tables(),
                   phantom::scheme_type::ckks, s);

    // KeySwitch
    auto cx = phantom::util::make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    auto reduction_threshold =
        (1 << (phantom::arith::bits_per_uint64 -
               static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) -
        1;
    key_switch_inner_prod(cx.get(), t_mod_up.get(),
                          relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                          reduction_threshold, s);

    // iNTT + ModDown + NTT
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(),
                                  phantom::scheme_type::ckks, s);
    }

    // Add d2 to d0, d1
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;

        auto ct_i = ct.data() + i * size_Ql_n;
        add_to_ct_kernel<<<size_Ql_n / phantom::util::blockDimGlb.x,
                           phantom::util::blockDimGlb, 0, s>>>(
            ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
    }

    ct.resize(context, ct.chain_index(), 2, cudaStreamPerThread);
}

void Evaluator::Rescale(const PhantomContext &context, PhantomCiphertext &ct) {
    const auto &stream = cudaStreamPerThread;

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &parms = context_data.parms();
    auto max_chain_index = parms.coeff_modulus().size();

    // Verify parameters.
    if (ct.chain_index() == max_chain_index) {
        throw std::invalid_argument("end of modulus switching chain reached");
    }

    // Modulus switching with scaling
    auto &rns_tool = context_data.gpu_rns_tool();
    size_t coeff_mod_size = parms.coeff_modulus().size();
    size_t poly_degree = parms.poly_modulus_degree();
    size_t encrypted_size = ct.size();

    auto next_index_id = context.get_next_index(ct.chain_index());
    auto &next_context_data = context.get_context_data(next_index_id);
    auto &next_parms = next_context_data.parms();

    auto ct_copy = phantom::util::make_cuda_auto_ptr<uint64_t>(
        encrypted_size * coeff_mod_size * poly_degree, stream);
    cudaMemcpyAsync(
        ct_copy.get(), ct.data(),
        encrypted_size * coeff_mod_size * poly_degree * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice, stream);

    // resize and empty the data array
    ct.resize(context, next_index_id, encrypted_size, stream);

    rns_tool.divide_and_round_q_last_ntt(ct_copy.get(), encrypted_size,
                                         context.gpu_rns_tables(), ct.data(),
                                         stream);

    ct.set_ntt_form(ct.is_ntt_form());
    ct.set_scale(ct.scale() /
                 static_cast<double>(parms.coeff_modulus().back().value()));
}
} // namespace hifive