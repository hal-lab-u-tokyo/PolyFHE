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
                  uint64_t **relin_keys, uint64_t *in0, uint64_t *in1,
                  uint64_t *out0, uint64_t *out2, bool if_benchmark, int n_opt);

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
    cudaMalloc(&d_tmp, params.KL * sizeof(uint64_t));
    for (int i = 0; i < params.KL; i++) {
        cudaMemcpy(d_tmp + i, d_modulus[i].data(), sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
    }
    params.qVec = d_tmp;

    uint64_t *d_modulus_const_ratio;
    cudaMalloc(&d_modulus_const_ratio, 2 * params.KL * sizeof(uint64_t));
    for (int i = 0; i < params.KL; i++) {
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

void example_ckks(PhantomContext &context, const double &scale, int dnum,
                  int n_opt) {
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

    PhantomCiphertext xy_cipher_polyfhe = x_cipher;
    uint64_t poly_degree = context.gpu_rns_tables().n();
    auto &context_data = context.get_context_data(x_cipher.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    uint64_t coeff_mod_size = coeff_modulus.size();

    std::cout << "coeff_mod_size: " << coeff_mod_size << std::endl;

    Params params_h(std::log2(poly_degree), coeff_mod_size, dnum);
    std::cout << "L: " << params_h.L << std::endl;
    std::cout << "alpha: " << params_h.alpha << std::endl;
    std::cout << "dnum: " << params_h.dnum << std::endl;

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
    uint64_t *zero_array = (uint64_t *) malloc(sizeQPNBeta * sizeof(uint64_t));
    for (int i = 0; i < sizeQPNBeta; i++) {
        zero_array[i] = 0;
    }
    std::cout << "beta: " << beta << std::endl;
    uint64_t *res_modup_polyfhe, *res_modup_polyfhe2, *res_modup_phantom;
    checkCudaErrors(cudaMalloc((void **) &res_modup_polyfhe,
                               sizeQPNBeta * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **) &res_modup_polyfhe2,
                               sizeQPNBeta * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **) &res_modup_phantom,
                               sizeQPNBeta * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(res_modup_polyfhe, zero_array,
                               sizeQPNBeta * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));

    // PolyFHE's HMult
    std::cout << "Entry kernel" << std::endl;
    entry_kernel(params_d, &params_h, context, relin_keys.public_keys_ptr(),
                 in1, in2, res, res_modup_polyfhe, true, n_opt);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());

    // Phantom's HMult
    std::cout << "Phantom" << std::endl;
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace_debug(context, xy_cipher, relin_keys,
                              res_modup_phantom);
    checkCudaErrors(cudaDeviceSynchronize());
    // rescale_to_next_inplace(context, xy_cipher);
    std::cout << "xy_cipher.chain_index(): " << xy_cipher.chain_index()
              << std::endl;

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
    std::cout << "params_h.KL: " << params_h.KL << std::endl;
    std::cout << "poly_degree: " << poly_degree << std::endl;
    correctness = true;
    for (int beta_idx = 0; beta_idx < 2; beta_idx++) {
        std::cout << "beta_idx: " << beta_idx << std::endl;
        for (int i = 0; i < params_h.KL; i++) {
            for (int j = 0; j < poly_degree; j++) {
                int idx =
                    beta_idx * params_h.KL * poly_degree + i * poly_degree + j;
                if (h_modup_polyfhe[idx] != h_modup_phantom[idx]) {
                    cout << "  PolyFHE != Phantom at index[" << beta_idx << "]["
                         << i << "][" << j << "]" << endl;
                    cout << "   PolyFHE: " << h_modup_polyfhe[idx] << endl;
                    cout << "   Phantom: " << h_modup_phantom[idx] << endl;
                    correctness = false;
                    break;
                }
            }
        }
    }
    if (correctness) {
        cout << "  OK" << endl;
    } else {
        cout << "  Fail" << endl;
    }

    std::vector<double> elapsed_list;
    std::vector<double> elapsed_list_cuda;
    for (int iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaEvent_t ce_start, ce_stop;
        cudaEventCreate(&ce_start);
        cudaEventCreate(&ce_stop);
        cudaEventRecord(ce_start);
        PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
        relinearize_inplace_debug(context, xy_cipher, relin_keys,
                                  res_modup_phantom);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaEventRecord(ce_stop);
        cudaEventSynchronize(ce_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, ce_start, ce_stop);
        cudaEventDestroy(ce_start);
        cudaEventDestroy(ce_stop);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        if (iter != 0) {
            elapsed_list.push_back(elapsed);
            elapsed_list_cuda.push_back(milliseconds);
        }
    }
    double avg_time =
        std::accumulate(elapsed_list.begin(), elapsed_list.end(), 0.0) /
        elapsed_list.size();
    double avg_time_cuda = std::accumulate(elapsed_list_cuda.begin(),
                                           elapsed_list_cuda.end(), 0.0) /
                           elapsed_list_cuda.size();
    std::cout << "Average elapsed time (Phanotm CudaEvent): " << avg_time_cuda
              << " ms" << std::endl;
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

enum class ParamSize {
    Small,
    Medium,
    Large,
    Large2,
};

int main(int argc, char **argv) {
    // argv[1]: n_divide
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n_divide>" << std::endl;
        return 1;
    }
    int n_opt = atoi(argv[1]);
    srand(time(NULL));
    double scale = pow(2.0, 40);
    EncryptionParameters parms(scheme_type::ckks);

    ParamSize prmsize = ParamSize::Small;
    size_t poly_modulus_degree;
    int dnum;

    if (prmsize == ParamSize::Small) {
        poly_modulus_degree = 1 << 15;
        // L = 14, dnum = 5, alpha = 3
        dnum = 5;
        parms.set_coeff_modulus(CoeffModulus::Create(
            poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                  40, 40, 40, 40, 60, 60, 60}));
        parms.set_special_modulus_size(3);
    } else if (prmsize == ParamSize::Medium) {
        poly_modulus_degree = 1 << 16;
        dnum = 5;
        parms.set_coeff_modulus(CoeffModulus::Create(
            poly_modulus_degree,
            {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60}));
        parms.set_special_modulus_size(6);
    } else if (prmsize == ParamSize::Large) {
        poly_modulus_degree = 1 << 16;
        // L = 34
        parms.set_coeff_modulus(CoeffModulus::Create(
            poly_modulus_degree,
            {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60}));
        dnum = 7;
        parms.set_special_modulus_size(5); // dnum = ceil((L+ 1) / alpha)
    } else if (prmsize == ParamSize::Large2) {
        // L = 35, k = 4, dnum = 9
        poly_modulus_degree = 1 << 14;
        parms.set_coeff_modulus(CoeffModulus::Create(
            poly_modulus_degree, {60, 40, 40, 40, 40, 40, 60, 60}));
        dnum = 3;
    }

    parms.set_poly_modulus_degree(poly_modulus_degree);

    PhantomContext context(parms);
    print_parameters(context);
    example_ckks(context, scale, dnum, n_opt);
}