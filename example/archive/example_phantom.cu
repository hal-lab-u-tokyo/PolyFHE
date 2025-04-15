#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "examples/example.h"
#include "gtest/gtest.h"
#include "phantom.h"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#define EPSINON 0.001

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError_t err, const char *filename, int line);
inline void __checkCudaErrors(cudaError_t err, const char *filename, int line) {
    assert(filename);
    if (cudaSuccess != err) {
        const char *ename = cudaGetErrorName(err);
        printf(
            "CUDA API Error %04d: \"%s\" from file <%s>, "
            "line %i.\n",
            err, ((ename != NULL) ? ename : "Unknown"), filename, line);
        // exit(err);
    }
}

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

void example_ckks_encode_decode(PhantomContext &context, const double &scale) {
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    // Encode / Decode check
    vector<double> input(slot_count);
    vector<double> result;
    PhantomPlaintext pt;
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = (double) rand() / RAND_MAX;
    }
    encoder.encode(context, input, scale, pt, 1);
    encoder.decode(context, pt, result);

    bool correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= compare_double(result[i], input[i]);
    }
    EXPECT_EQ(correctness, true);
}

void example_ckks_enc_sym(PhantomContext &context, const double &scale) {
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    vector<double> input(slot_count);
    vector<double> result;
    PhantomPlaintext pt;
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = (double) rand() / RAND_MAX;
    }
    encoder.encode(context, input, scale, pt, 1);

    // Symmetric encryption check
    PhantomCiphertext x_symmetric_cipher;
    PhantomPlaintext x_symmetric_plain;
    secret_key.encrypt_symmetric(context, pt, x_symmetric_cipher);
    secret_key.decrypt(context, x_symmetric_cipher, x_symmetric_plain);
    encoder.decode(context, x_symmetric_plain, result);

    bool correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= compare_double(result[i], input[i]);
    }
    EXPECT_EQ(correctness, true);
}

void example_ckks_enc_asymm(PhantomContext &context, const double &scale) {
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    vector<double> input(slot_count);
    vector<double> result;
    PhantomPlaintext pt;
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = (double) rand() / RAND_MAX;
    }
    encoder.encode(context, input, scale, pt, 1);

    // Asymmetric encryption check
    PhantomCiphertext x_asymmetric_cipher;
    PhantomPlaintext x_asymmetric_plain;
    public_key.encrypt_asymmetric(context, pt, x_asymmetric_cipher);
    secret_key.decrypt(context, x_asymmetric_cipher, x_asymmetric_plain);
    encoder.decode(context, x_asymmetric_plain, result);

    bool correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= compare_double(result[i], input[i]);
    }
    EXPECT_EQ(correctness, true);
}

void example_ckks_add(PhantomContext &context, const double &scale) {
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

    cout << "Input vector 1: length = " << msg_size1 << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: length = " << msg_size2 << endl;
    print_vector(input2, 3, 7);

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
    poly_add<<<4096, 512>>>(res, in1, in2, modulus, poly_degree,
                            coeff_mod_size);
    checkCudaErrors(cudaDeviceSynchronize());
    in1 = x_sym_cipher.data() + poly_degree * coeff_mod_size;
    in2 = y_sym_cipher.data() + poly_degree * coeff_mod_size;
    res = x_sym_cipher.data() + poly_degree * coeff_mod_size;
    poly_add<<<4096, 512>>>(res, in1, in2, modulus, poly_degree,
                            coeff_mod_size);
    checkCudaErrors(cudaDeviceSynchronize());

    PhantomPlaintext x_plus_y_sym_plain;
    secret_key.decrypt(context, x_sym_cipher, x_plus_y_sym_plain);
    encoder.decode(context, x_plus_y_sym_plain, result);

    cout << "Result: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        if (i >= msg_size1)
            correctness &= result[i] == input2[i];
        else if (i >= msg_size2)
            correctness &= result[i] == input1[i];
        else
            correctness &= result[i] == cuCadd(input1[i], input2[i]);
    }
    EXPECT_EQ(correctness, true);
}

TEST(TestEncode, logN15_L20_alpha1) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    PhantomContext context(parms);
    example_ckks_encode_decode(context, scale);
}

TEST(TestEncode, logN15_L18_alpha2) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 60, 60}));
    parms.set_special_modulus_size(2);
    PhantomContext context(parms);
    example_ckks_encode_decode(context, scale);
}

TEST(TestEncode, logN15_L18_alpha3) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));
    parms.set_special_modulus_size(3);
    PhantomContext context(parms);
    example_ckks_encode_decode(context, scale);
}

TEST(TestEncode, logN15_L16_alpha4) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
    parms.set_special_modulus_size(4);
    PhantomContext context(parms);
    example_ckks_encode_decode(context, scale);
}

TEST(TestEncode, logN16_L40_alpha15) {
    srand(time(NULL));
    double scale = pow(2.0, 50);
    size_t poly_modulus_degree = 1 << 16;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    parms.set_special_modulus_size(15);
    PhantomContext context(parms);
    example_ckks_encode_decode(context, scale);
}

TEST(TestEncrypt, logN15_L20_alpha1) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    PhantomContext context(parms);
    example_ckks_enc_sym(context, scale);
    example_ckks_enc_asymm(context, scale);
}

TEST(TestEncrypt, logN15_L18_alpha2) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 60, 60}));
    parms.set_special_modulus_size(2);
    PhantomContext context(parms);
    example_ckks_enc_sym(context, scale);
    example_ckks_enc_asymm(context, scale);
}

TEST(TestEncrypt, logN15_L18_alpha3) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));
    parms.set_special_modulus_size(3);
    PhantomContext context(parms);
    example_ckks_enc_sym(context, scale);
    example_ckks_enc_asymm(context, scale);
}

TEST(TestEncrypt, logN15_L16_alpha4) {
    srand(time(NULL));
    double scale = pow(2.0, 40);
    size_t poly_modulus_degree = 1 << 15;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
    parms.set_special_modulus_size(4);
    PhantomContext context(parms);
    example_ckks_enc_sym(context, scale);
    example_ckks_enc_asymm(context, scale);
}

TEST(TestEncrypt, logN16_L40_alpha15) {
    srand(time(NULL));
    double scale = pow(2.0, 50);
    size_t poly_modulus_degree = 1 << 16;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    parms.set_special_modulus_size(15);
    PhantomContext context(parms);
    example_ckks_enc_sym(context, scale);
    example_ckks_enc_asymm(context, scale);
}

TEST(TestHAdd, logN15_L20_alpha1) {
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
    example_ckks_add(context, scale);
}