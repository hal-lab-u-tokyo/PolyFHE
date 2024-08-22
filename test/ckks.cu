#include <gtest/gtest.h>

#include <vector>

#include "evaluate.h"
#include "example.h"
#include "phantom.h"

TEST(GPUContextTest, GetGPUInfo) { EXPECT_EQ(1, 1); }

TEST(CKKS, Encrypt) {
    // std::vector v_alpha = {1, 2, 3, 4, 15};
    std::vector v_alpha = {3};
    for (auto alpha : v_alpha) {
        std::cout << "alpha: " << alpha << std::endl;
        phantom::EncryptionParameters parms(phantom::scheme_type::ckks);

        size_t poly_modulus_degree = 1 << 15;
        double scale = pow(2.0, 40);
        switch (alpha) {
        case 1:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        case 2:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 60, 60}));
            parms.set_special_modulus_size(alpha);
            break;
        case 3:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 60, 60, 60}));
            parms.set_special_modulus_size(alpha);
            break;
        case 4:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 60, 60, 60, 60}));
            // hybrid key-switching
            parms.set_special_modulus_size(alpha);
            break;
        case 15:
            poly_modulus_degree = 1 << 16;
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree,
                {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
            parms.set_special_modulus_size(alpha);
            scale = pow(2.0, 50);
            break;
        default:
            throw std::invalid_argument("unsupported alpha params");
        }
        PhantomContext context(parms);
        print_parameters(context);

        // KeyGen
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

        // Encoder
        PhantomCKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        std::cout << "Number of slots: " << slot_count << std::endl;

        std::vector<cuDoubleComplex> x_msg, y_msg;
        double rand_real, rand_imag;

        size_t x_size = slot_count;
        size_t y_size = slot_count;
        x_msg.reserve(x_size);
        for (size_t i = 0; i < x_size; i++) {
            // rand_real = (double) rand() / RAND_MAX;
            // rand_imag = (double) rand() / RAND_MAX;
            rand_real = i * 1.0;
            rand_imag = 0.0;
            x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }
        std::cout << "Message vector of X: " << std::endl;
        print_vector(x_msg, 3, 7);

        y_msg.reserve(y_size);
        for (size_t i = 0; i < y_size; i++) {
            // rand_real = (double) rand() / RAND_MAX;
            // rand_imag = (double) rand() / RAND_MAX;
            rand_real = i * 1.0;
            rand_imag = 0.0;
            y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }
        std::cout << "Message vector of Y: " << std::endl;
        print_vector(y_msg, 3, 7);

        PhantomPlaintext x_plain;
        PhantomPlaintext y_plain;
        PhantomPlaintext xy_plain;

        encoder.encode(context, x_msg, scale, x_plain);
        encoder.encode(context, y_msg, scale, y_plain);

        // Encrypt
        PhantomCiphertext x_cipher;
        PhantomCiphertext y_cipher;
        PhantomCiphertext xy_cipher;
        public_key.encrypt_asymmetric(context, x_plain, x_cipher);
        public_key.encrypt_asymmetric(context, y_plain, y_cipher);
        std::cout << "x_cipher size: " << x_cipher.size() << std::endl;

        // Evaluate
        hifive::Evaluator evaluator;
        evaluator.Mult(context, xy_cipher, x_cipher, y_cipher);

        // Decrypt
        std::cout << "Result vector: " << std::endl;
        PhantomPlaintext x_plain_result = secret_key.decrypt(context, x_cipher);
        auto result = encoder.decode<cuDoubleComplex>(context, x_plain_result);
        print_vector(result, 3, 7);
    }
}