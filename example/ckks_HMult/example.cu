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

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#define EPSINON 0.001

void entry_kernel(Params *params_d, Params *params_h, PhantomContext &context,
                  uint64_t *in0, uint64_t *in1, uint64_t *out0,
                  bool if_benchmark);

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

void ConvertPhantomToParams(Params &params, const DModulus *d_modulus,
                            const DNTTTable &ntt_tables) {
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
}

/*
__global__ void poly_add(uint64_t *res, uint64_t *in1, uint64_t *in2,
                         uint64_t *modulus, uint64_t degree,
                         uint64_t mod_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < degree * mod_size) {
        uint64_t idx_mod = idx / degree;
        uint64_t idx_poly = idx % degree;
        uint64_t data_idx = idx_mod * degree + idx_poly;
        res[data_idx] = (in1[data_idx] + in2[data_idx]) % modulus[idx_mod];
    }
}
*/

__global__ void poly_mult(uint64_t *res, uint64_t *in1, uint64_t *in2,
                          Params *params) {
    int degree = params->N;
    int mod_size = params->L;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < degree * mod_size; idx += blockDim.x * gridDim.x) {
        uint64_t idx_mod = idx / degree;
        uint64_t idx_poly = idx % degree;
        uint64_t data_idx = idx_mod * degree + idx_poly;
        res[data_idx] = multiply_and_barrett_reduce_uint64(
            in1[data_idx], in2[data_idx], params->qVec[idx_mod],
            params->modulus_const_ratio + idx_mod * 2);
    }
}

__global__ void poly_mult_shoup(uint64_t *res, uint64_t *in1, uint64_t *in2,
                                uint64_t *in2_shoup, Params *params) {
    int degree = params->N;
    int mod_size = params->L;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < degree * mod_size; idx += blockDim.x * gridDim.x) {
        uint64_t idx_mod = idx / degree;
        uint64_t idx_poly = idx % degree;
        uint64_t data_idx = idx_mod * degree + idx_poly;
        res[data_idx] = multiply_and_reduce_shoup(in1[data_idx], in2[idx_mod],
                                                  in2_shoup[idx_mod],
                                                  params->qVec[idx_mod]);
    }
}

__global__ static void poly_inplace_inwt_radix8_phase1(
    uint64_t *inout, const size_t coeff_mod_size, const size_t start_mod_idx,
    Params *params) {
    extern __shared__ uint64_t buff[];
    uint64_t samples[8];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (params->N / 8 * coeff_mod_size); i += blockDim.x * gridDim.x) {
        d_poly_inplace_inwt_radix8_phase1(inout, params, coeff_mod_size,
                                          start_mod_idx, buff, samples, i);
        // prime idx
        size_t n_twr = params->N / 8;
        size_t n_idx = i % n_twr;
        size_t twr_idx = i / n_twr + start_mod_idx;
        size_t group = params->n1 / 8;
        // pad address
        size_t pad_tid = threadIdx.x % params->pad;
        size_t pad_idx = threadIdx.x / params->pad;
        const size_t n_init = n_twr / group * pad_idx + pad_tid +
                              params->pad * (n_idx / (group * params->pad));
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(inout + twr_idx * params->N + n_init + n_twr * j) = samples[j];
        }
    }
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
    const DNTTTable &ntt_tables = context.gpu_rns_tables();

    std::cout << "coeff_mod_size: " << coeff_mod_size << std::endl;

    Params params_h;
    params_h.N = poly_degree;
    params_h.L = coeff_mod_size;
    params_h.logN = log2(poly_degree);
    params_h.n1 = 1 << (params_h.logN / 2);
    params_h.n2 = params_h.N / params_h.n1;
    ConvertPhantomToParams(params_h, context.gpu_rns_tables().modulus(),
                           ntt_tables);
    Params *params_d;
    checkCudaErrors(cudaMalloc((void **) &params_d, sizeof(Params)));
    checkCudaErrors(cudaMemcpy(params_d, &params_h, sizeof(Params),
                               cudaMemcpyHostToDevice));

    uint64_t *in1 = x_cipher.data();
    uint64_t *in2 = y_cipher.data();
    xy_cipher_polyfhe.resize(3, coeff_mod_size, poly_degree, s);
    uint64_t *res = xy_cipher_polyfhe.data();

    entry_kernel(params_d, &params_h, context, in1, in2, res, true);
    checkCudaErrors(cudaDeviceSynchronize());
    /*
    size_t block_size = 128;
    size_t grid_size = 4096;
    uint64_t *res2 = res + params_h.L * params_h.N * 2;
    auto &rns_tool = context.get_context_data(xy_cipher_polyfhe.chain_index())
                         .gpu_rns_tool();
    poly_mult_shoup<<<grid_size, block_size>>>(
        res2, res2, rns_tool.partQlHatInv_mod_Ql_concat(),
        rns_tool.partQlHatInv_mod_Ql_concat_shoup(), params_d);
    checkCudaErrors(cudaDeviceSynchronize());
    */

    // Phantom's HMult
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace(context, xy_cipher, relin_keys);
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

    for (int idx = 0; idx < xy_cipher.size(); idx++) {
        std::cout << "idx: " << idx << std::endl;
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
        bool correctness = true;
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
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
    parms.set_special_modulus_size(2);
    PhantomContext context(parms);
    print_parameters(context);
    example_ckks(context, scale);
}