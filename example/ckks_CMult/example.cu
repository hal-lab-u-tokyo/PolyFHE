#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "phantom-fhe/examples/example.h"
#include "phantom-fhe/include/phantom.h"
#include "phantom-fhe/include/util.cuh"
#include "polyfhe/kernel/device_context.hpp"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#define EPSINON 0.001

void entry_kernel(Params *params_d, Params *params_h, uint64_t *in0,
                  uint64_t *in1, uint64_t *out0, bool if_benchmark);

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

void ConvertPhantomToParams(Params &params, const DModulus *d_modulus) {
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

__global__ void poly_mult(uint64_t *res, uint64_t *in1, uint64_t *in2,
                          Params *params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int degree = params->N;
    int mod_size = params->L;
    if (idx < degree * mod_size) {
        uint64_t idx_mod = idx / degree;
        uint64_t idx_poly = idx % degree;
        uint64_t data_idx = idx_mod * degree + idx_poly;
        res[data_idx] = multiply_and_barrett_reduce_uint64(
            in1[data_idx], in2[data_idx], params->qVec[idx_mod],
            params->modulus_const_ratio + idx_mod * 2);
    }
}
*/

void example_ckks(PhantomContext &context, const double &scale) {
    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    vector<cuDoubleComplex> input1, input2_const, result;
    size_t msg_size1 = slot_count;
    size_t msg_size2 = slot_count;
    input1.reserve(msg_size1);
    input2_const.reserve(msg_size2);
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
        input2_const.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Input vector 1: length = " << msg_size1 << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: length = " << msg_size2 << endl;
    print_vector(input2_const, 3, 7);

    PhantomPlaintext x_plain, const_plain;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2_const, scale, const_plain);

    PhantomCiphertext x_sym_cipher;
    secret_key.encrypt_symmetric(context, x_plain, x_sym_cipher);

    uint64_t *in1 = x_sym_cipher.data();
    uint64_t *in2 = const_plain.data();
    uint64_t *res = x_sym_cipher.data();
    uint64_t poly_degree = context.gpu_rns_tables().n();
    auto &context_data = context.get_context_data(x_sym_cipher.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    uint64_t coeff_mod_size = coeff_modulus.size();

    Params params_h;
    params_h.N = poly_degree;
    params_h.L = coeff_mod_size;
    ConvertPhantomToParams(params_h, context.gpu_rns_tables().modulus());
    Params *params_d;
    checkCudaErrors(cudaMalloc((void **) &params_d, sizeof(Params)));
    checkCudaErrors(cudaMemcpy(params_d, &params_h, sizeof(Params),
                               cudaMemcpyHostToDevice));
    entry_kernel(params_d, &params_h, in1, in2, res, false);

    /*
    // ax
    poly_mult<<<4096, 512>>>(res, in1, in2, params_d);
    checkCudaErrors(cudaDeviceSynchronize());
    // bx
    in1 = in1 + poly_degree * coeff_mod_size;
    res = res + poly_degree * coeff_mod_size;
    poly_mult<<<4096, 512>>>(res, in1, in2, params_d);
    checkCudaErrors(cudaDeviceSynchronize());
    */

    PhantomPlaintext x_plus_y_sym_plain;
    x_sym_cipher.set_scale(x_sym_cipher.scale() * const_plain.scale());
    secret_key.decrypt(context, x_sym_cipher, x_plus_y_sym_plain);
    encoder.decode(context, x_plus_y_sym_plain, result);

    cout << "Result: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        if (i >= msg_size1)
            correctness &= result[i] == input2_const[i];
        else if (i >= msg_size2)
            correctness &= result[i] == input1[i];
        else
            correctness &= result[i] == cuCmul(input1[i], input2_const[i]);
    }
    if (correctness) {
        cout << "Correctness check passed!" << endl;
    } else {
        cout << "Correctness check failed!" << endl;
    }
}

int main() {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    PhantomContext context(parms);
    print_parameters(context);
    example_ckks(context, scale);
}