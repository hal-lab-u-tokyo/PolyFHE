#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "phantom-fhe/examples/example.h"
#include "phantom-fhe/include/butterfly.cuh"
#include "phantom-fhe/include/phantom.h"
#include "polyfhe/kernel/device_context.hpp"
#include "polyfhe/kernel/ntt-phantom.hpp"
#include "polyfhe/kernel/polynomial.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#define EPSINON 0.001

void entry_kernel(Params *params_d, Params *params_h, PhantomContext &context,
                  uint64_t *in0, uint64_t *in1, uint64_t *out0, uint64_t *out2,
                  bool if_benchmark);

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

__global__ static void NTTPhase1(Params *params, uint64_t *inout,
                                 const uint64_t *twiddles,
                                 const uint64_t *twiddles_shoup,
                                 const DModulus *modulus, size_t coeff_mod_size,
                                 size_t start_mod_idx,
                                 size_t excluded_range_start,
                                 size_t excluded_range_end) {
    extern __shared__ uint64_t buffer[];
    uint64_t samples[8];
    const size_t n_tower = params->N / 8;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n_tower * coeff_mod_size; tid += blockDim.x * gridDim.x) {
        const size_t twr_idx = tid / n_tower + start_mod_idx;
        if (twr_idx >= excluded_range_start && twr_idx < excluded_range_end) {
            continue;
        }

        const uint64_t size_P = params->K;
        const uint64_t size_QP = params->KL;
        size_t twr_idx2 =
            (twr_idx >= start_mod_idx + coeff_mod_size - size_P
                 ? size_QP - (start_mod_idx + coeff_mod_size - twr_idx)
                 : twr_idx);

        const size_t group = params->n1 / 8;
        const size_t pad = params->per_block_pad;
        const size_t pad_tid = threadIdx.x % pad;
        const size_t pad_idx = threadIdx.x / pad;
        size_t n_idx = tid % n_tower;
        const size_t n_init =
            n_tower / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

        uint64_t *in_data_ptr = inout + twr_idx * params->N;

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = *(in_data_ptr + n_init + n_tower * j);
        }

        d_poly_fnwt_phase1(params, inout, buffer, samples, twiddles,
                           twiddles_shoup, modulus, twr_idx, twr_idx2, n_init,
                           tid);
    }
}

__global__ static void NTTPhase2(Params *params, uint64_t *inout,
                                 const uint64_t *twiddles,
                                 const uint64_t *twiddles_shoup,
                                 const DModulus *modulus, size_t coeff_mod_size,
                                 size_t start_mod_idx,
                                 size_t excluded_range_start,
                                 size_t excluded_range_end) {
    extern __shared__ uint64_t buffer[];
    uint64_t samples[8];
    const size_t n_tower = params->N / 8;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n_tower * coeff_mod_size); tid += blockDim.x * gridDim.x) {
        // prime idx
        size_t twr_idx = coeff_mod_size - 1 - (tid / n_tower) + start_mod_idx;
        if (twr_idx >= excluded_range_start && twr_idx < excluded_range_end) {
            continue;
        }

        uint64_t n_init;
        d_poly_fnwt_phase2(params, inout, buffer, samples, twiddles,
                           twiddles_shoup, modulus, coeff_mod_size,
                           start_mod_idx, twr_idx, &n_init, tid);

        uint64_t *out_ptr = inout + twr_idx * params->N;
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(out_ptr + n_init + params->n2 / 8 * j) = samples[j];
        }
    }
}

void ConvertPhantomToParams(Params &params, const PhantomContext &context) {
    const DModulus *d_modulus = context.gpu_rns_tables().modulus();
    const DNTTTable &ntt_tables = context.gpu_rns_tables();
    uint64_t *d_tmp;
    cudaMalloc(&d_tmp, params.L * sizeof(uint64_t));
    for (int i = 0; i < params.L; i++) {
        cudaMemcpy(d_tmp + i, d_modulus[i].data(), sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
    }
    params.qVec = d_tmp;

    uint64_t *d_modulus_const_ratio;
    cudaMalloc(&d_modulus_const_ratio, 2 * params.L * sizeof(uint64_t));
    for (int i = 0; i < params.L; i++) {
        cudaMemcpy(d_modulus_const_ratio + 2 * i, d_modulus[i].const_ratio(),
                   2 * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    }
    params.modulus_const_ratio = d_modulus_const_ratio;

    // NTT tables
    params.itwiddle = ntt_tables.itwiddle();
    params.itwiddle_shoup = ntt_tables.itwiddle_shoup();
    params.n_inv = ntt_tables.n_inv_mod_q();
    params.n_inv_shoup = ntt_tables.n_inv_mod_q_shoup();

    // DRNSTool
    for (int i = 0; i < params.L + 1; i++) {
        auto &context_data = context.get_context_data(i);
        phantom::DRNSTool &rns_tool = context_data.gpu_rns_tool();
        params.rns_tools.push_back(&rns_tool);
    }

    // DNTTTable
    params.ntt_tables = &context.gpu_rns_tables();
}

void example_ckks(PhantomContext &context, const double &scale) {
    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomCKKSEncoder encoder(context);
    const auto &s = cudaStreamPerThread;

    size_t slot_count = encoder.slot_count();
    vector<cuDoubleComplex> input1, input2, result;
    size_t msg_size1 = slot_count;
    size_t msg_size2 = slot_count;
    input1.reserve(msg_size1);
    input2.reserve(msg_size2);
    double rand_real, rand_imag;
    srand(time(0));
    for (size_t i = 0; i < msg_size1; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input1.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    for (size_t i = 0; i < msg_size2; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input2.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Input vector 1: length = " << msg_size1 << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: length = " << msg_size2 << endl;
    print_vector(input2, 3, 7);

    PhantomPlaintext x_plain, y_plain;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2, scale, y_plain);
    std::cout << "x_plain.chain_index(): " << x_plain.chain_index()
              << std::endl;

    PhantomCiphertext x_cipher, y_cipher;
    secret_key.encrypt_symmetric(context, x_plain, x_cipher);
    secret_key.encrypt_symmetric(context, y_plain, y_cipher);
    std::cout << "x_plain.chain_index(): " << x_plain.chain_index()
              << std::endl;
    std::cout << "x_cipher.chain_index(): " << x_cipher.chain_index()
              << std::endl;

    // PolyFHE's HMult
    PhantomCiphertext xy_cipher_polyfhe = x_cipher;
    uint64_t poly_degree = context.gpu_rns_tables().n();
    auto &context_data = context.get_context_data(x_cipher.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    uint64_t coeff_mod_size = coeff_modulus.size();

    std::cout << "coeff_mod_size: " << coeff_mod_size << std::endl;

    Params params_h(std::log2(poly_degree), coeff_mod_size, 5);
    ConvertPhantomToParams(params_h, context);
    Params *params_d;
    checkCudaErrors(cudaMalloc((void **) &params_d, sizeof(Params)));
    checkCudaErrors(cudaMemcpy(params_d, &params_h, sizeof(Params),
                               cudaMemcpyHostToDevice));

    uint64_t *in1 = x_cipher.data();
    uint64_t *in2 = y_cipher.data();
    xy_cipher_polyfhe.resize(3, coeff_mod_size, poly_degree, s);
    uint64_t *res = xy_cipher_polyfhe.data();

    const int beta = std::ceil((params_h.L + 1) / params_h.alpha);
    const int sizeQP = coeff_mod_size + params_h.alpha;
    const int sizeQPNBeta = poly_degree * sizeQP * beta;
    std::cout << "beta: " << beta << std::endl;
    uint64_t *res_modup_polyfhe, *res_modup_polyfhe2, *res_modup_phantom;
    checkCudaErrors(cudaMalloc((void **) &res_modup_polyfhe,
                               sizeQPNBeta * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **) &res_modup_polyfhe2,
                               sizeQPNBeta * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **) &res_modup_phantom,
                               sizeQPNBeta * sizeof(uint64_t)));

    // PolyFHE's HMult
    entry_kernel(params_d, &params_h, context, in1, in2, res, res_modup_polyfhe,
                 true);
    checkCudaErrors(cudaDeviceSynchronize());
    /*
    for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
        uint64_t *inout_i = res_modup_polyfhe + beta_idx * poly_degree * sizeQP;
        const int per_block_pad = params_h.per_block_pad;
        const int start_modulus_idx = 0;
        const int excluded_range_start = params_h.alpha * beta_idx;
        const int excluded_range_end = params_h.alpha * (beta_idx + 1);
        NTTPhase1<<<4096, (params_h.n1 / 8) * per_block_pad,
                    (params_h.n1 + per_block_pad + 1) * per_block_pad *
                        sizeof(uint64_t)>>>(
            params_d, inout_i, params_h.ntt_tables->twiddle(),
            params_h.ntt_tables->twiddle_shoup(),
            params_h.ntt_tables->modulus(), 20, start_modulus_idx,
            excluded_range_start, excluded_range_end);
        checkCudaErrors(cudaDeviceSynchronize());
        NTTPhase2<<<4096, 128, 128 * 8 * sizeof(uint64_t)>>>(
            params_d, inout_i, params_h.ntt_tables->twiddle(),
            params_h.ntt_tables->twiddle_shoup(),
            params_h.ntt_tables->modulus(), 20, start_modulus_idx,
            excluded_range_start, excluded_range_end);
    }
    */

    // Phantom's HMult
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace_debug(context, xy_cipher, relin_keys,
                              res_modup_phantom);
    checkCudaErrors(cudaDeviceSynchronize());
    // rescale_to_next_inplace(context, xy_cipher);
    std::cout << "xy_cipher.chain_index(): " << xy_cipher.chain_index()
              << std::endl;

    /*
     */
    // Check if PolyFHE's HMult and Phantom's HMult are the same
    uint64_t *h_res_polyfhe =
        (uint64_t *) malloc(poly_degree * coeff_mod_size * sizeof(uint64_t));
    uint64_t *h_res_phantom =
        (uint64_t *) malloc(poly_degree * coeff_mod_size * sizeof(uint64_t));

    bool correctness = true;
    for (int idx = 0; idx < xy_cipher.size(); idx++) {
        std::cout << "idx: " << idx << std::endl;
        correctness = true;
        uint64_t *d_res_polyfhe =
            xy_cipher_polyfhe.data() + idx * poly_degree * coeff_mod_size;
        uint64_t *d_res_phantom =
            xy_cipher.data() + idx * poly_degree * coeff_mod_size;
        checkCudaErrors(
            cudaMemcpy(h_res_polyfhe, d_res_polyfhe,
                       poly_degree * coeff_mod_size * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
        checkCudaErrors(
            cudaMemcpy(h_res_phantom, d_res_phantom,
                       poly_degree * coeff_mod_size * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
        for (int i = 0; i < poly_degree * coeff_mod_size; i++) {
            if (h_res_polyfhe[i] != h_res_phantom[i]) {
                correctness = false;
                cout << "  PolyFHE != Phantom at index " << i << endl;
                cout << "   PolyFHE: " << h_res_polyfhe[i] << endl;
                cout << "   Phantom: " << h_res_phantom[i] << endl;
                break;
            }
        }
        if (correctness) {
            cout << "  OK" << endl;
        } else {
            cout << "  Fail" << endl;
        }
    }

    // Check t_modup_ptr
    uint64_t *h_modup_polyfhe =
        (uint64_t *) malloc(sizeQPNBeta * sizeof(uint64_t));
    uint64_t *h_modup_phantom =
        (uint64_t *) malloc(sizeQPNBeta * sizeof(uint64_t));
    checkCudaErrors(cudaMemcpy(h_modup_polyfhe, res_modup_polyfhe,
                               sizeQPNBeta * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_modup_phantom, res_modup_phantom,
                               sizeQPNBeta * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));
    std::cout << "Modup result" << std::endl;
    correctness = true;
    for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
        for (int j = 0; j < poly_degree * params_h.KL; j++) {
            int i = beta_idx * poly_degree * params_h.KL + j;
            if (h_modup_polyfhe[i] != h_modup_phantom[i]) {
                cout << "  PolyFHE != Phantom at index[" << beta_idx << "]["
                     << j << "]" << endl;
                cout << "   PolyFHE: " << h_modup_polyfhe[i] << endl;
                cout << "   Phantom: " << h_modup_phantom[i] << endl;
                correctness = false;
                break;
            }
        }
    }
    if (correctness) {
        cout << "  OK" << endl;
    } else {
        cout << "  Fail" << endl;
    }

    std::vector<double> elapsed_list;
    for (int iter = 0; iter < 7; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
        relinearize_inplace_debug(context, xy_cipher, relin_keys,
                                  res_modup_phantom);
        checkCudaErrors(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        if (iter != 0) {
            elapsed_list.push_back(elapsed);
        }
    }
    double avg_time =
        std::accumulate(elapsed_list.begin(), elapsed_list.end(), 0.0) /
        elapsed_list.size();
    std::cout << "Average elapsed time (Phantom): " << avg_time << " us"
              << std::endl;

    /*
    PhantomPlaintext xy_plain;
    secret_key.decrypt(context, xy_cipher, xy_plain);
    encoder.decode(context, xy_plain, result);

    cout << "Result: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        if (i >= msg_size1)
            correctness &= result[i] == input2[i];
        else if (i >= msg_size2)
            correctness &= result[i] == input1[i];
        else
            correctness &= result[i] == cuCmul(input1[i], input2[i]);
    }
    if (correctness) {
        cout << "Correctness check passed!" << endl;
    } else {
        cout << "Correctness check failed!" << endl;
    }
     */
}

int main() {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 16;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    /*
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
    parms.set_special_modulus_size(2);
    */
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60}));
    parms.set_special_modulus_size(6);
    PhantomContext context(parms);
    print_parameters(context);
    example_ckks(context, scale);
}