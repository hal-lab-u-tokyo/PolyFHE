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

    PhantomPlaintext x_plain, y_plain_const;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2, scale, y_plain_const);
    std::cout << "x_plain.chain_index(): " << x_plain.chain_index()
              << std::endl;

    PhantomCiphertext x_cipher;
    secret_key.encrypt_symmetric(context, x_plain, x_cipher);

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
    uint64_t *in2 = y_plain_const.data();
    PhantomCiphertext res_cipher;
    res_cipher.resize(2, coeff_mod_size, poly_degree, s);
    uint64_t *res = res_cipher.data();
    uint64_t *res_dummy = res;

    // PolyFHE's CMult
    entry_kernel(params_d, &params_h, context, in1, in2, res, res_dummy, true);
    checkCudaErrors(cudaDeviceSynchronize());

    // Phantom's CMult
    // multiply_plain_inplace(context, x_cipher_phantom, y_plain_const);
    PhantomCiphertext res_phantom =
        multiply_plain(context, x_cipher, y_plain_const);
    checkCudaErrors(cudaDeviceSynchronize());

    // Check if PolyFHE's HMult and Phantom's HMult are the same
    uint64_t *h_res_polyfhe =
        (uint64_t *) malloc(poly_degree * coeff_mod_size * sizeof(uint64_t));
    uint64_t *h_res_phantom =
        (uint64_t *) malloc(poly_degree * coeff_mod_size * sizeof(uint64_t));

    bool correctness = true;
    for (int idx = 0; idx < x_cipher.size(); idx++) {
        std::cout << "idx: " << idx << std::endl;
        correctness = true;
        uint64_t *d_res_polyfhe = res + idx * poly_degree * coeff_mod_size;
        uint64_t *d_res_phantom =
            res_phantom.data() + idx * poly_degree * coeff_mod_size;
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

    std::vector<double> elapsed_list;
    for (int iter = 0; iter < 7; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        PhantomCiphertext xy_cipher =
            multiply_plain(context, x_cipher, y_plain_const);
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