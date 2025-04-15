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

uint64_t *convert_DModulus_to_uint64_t(const DModulus *d_modulus, int len) {
    uint64_t *d_modulus_new;
    cudaMalloc(&d_modulus_new, len * sizeof(uint64_t));
    for (int i = 0; i < len; i++) {
        cudaMemcpy(d_modulus_new + i, d_modulus[i].data(), sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
    }
    return d_modulus_new;
}

void example_ckks(PhantomContext &context, const double &scale) {
    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);

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

    PhantomPlaintext x_plain, y_plain;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2, scale, y_plain);

    PhantomCiphertext x_sym_cipher, y_sym_cipher;
    secret_key.encrypt_symmetric(context, x_plain, x_sym_cipher);
    secret_key.encrypt_symmetric(context, y_plain, y_sym_cipher);

    uint64_t *in1 = x_sym_cipher.data();
    uint64_t *in2 = y_sym_cipher.data();
    uint64_t *res = x_sym_cipher.data();
    uint64_t poly_degree = context.gpu_rns_tables().n();
    auto &context_data = context.get_context_data(x_sym_cipher.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    uint64_t coeff_mod_size = coeff_modulus.size();
    uint64_t *modulus = convert_DModulus_to_uint64_t(
        context.gpu_rns_tables().modulus(), coeff_mod_size);
    checkCudaErrors(cudaGetLastError());

    Params params_h;
    Params *params_d;
    checkCudaErrors(cudaMalloc((void **) &params_d, sizeof(Params)));
    params_h.N = poly_degree;
    params_h.L = coeff_mod_size;
    params_h.qVec = modulus;
    checkCudaErrors(cudaMemcpy(params_d, &params_h, sizeof(Params),
                               cudaMemcpyHostToDevice));
    entry_kernel(params_d, &params_h, in1, in2, res, true);
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