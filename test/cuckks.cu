#include "test.h"

TEST(cuCKKS, Params) {
    // Phantom
    std::vector v_alpha = {1, 2, 3, 4, 15};
    for (auto alpha : v_alpha) {
        phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        switch (alpha) {
        case 1:
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));

            break;
        case 2:
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 60, 60}));
            break;
        case 3:
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 60, 60, 60}));
            break;
        case 4:
            parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 60, 60, 60, 60}));
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
            break;
        default:
            throw std::invalid_argument("Invalid alpha");
        }

        PhantomContext context(parms);
    }
}

TEST(cuCKKS, Encrypt) {
    const uint64_t alpha = 3;
    const uint64_t poly_modulus_degree = 1 << 15;

    phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // Encode and Encrypt
    size_t slot_count = encoder.slot_count();
    const double scale = pow(2.0, 40);
    std::vector<double> input(slot_count, 0);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = i;
    }
    PhantomPlaintext x_plain;
    PhantomCiphertext x_encrypted;
    encoder.encode(context, input, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_encrypted);

    // Decrypt and Decode
    PhantomPlaintext x_decoded;
    std::vector<double> output;
    secret_key.decrypt(context, x_encrypted, x_decoded);
    encoder.decode(context, x_decoded, output);
    for (size_t i = 0; i < slot_count; i++) {
        EXPECT_NEAR(input[i], output[i], 1e-6);
    }
}

TEST(cuCKKS, HAdd) {
    const uint64_t alpha = 3;
    const uint64_t poly_modulus_degree = 1 << 15;

    phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // Encode and Encrypt
    size_t slot_count = encoder.slot_count();
    const double scale = pow(2.0, 40);
    std::vector<double> input(slot_count, 0);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = i;
    }
    PhantomPlaintext x_plain;
    PhantomCiphertext x_encrypted, y_encrypted;
    encoder.encode(context, input, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_encrypted);
    public_key.encrypt_asymmetric(context, x_plain, y_encrypted);

    // Copy to GPU
    hifive::Ciphertext ct_x(x_encrypted);
    hifive::Ciphertext ct_y(y_encrypted);
    hifive::Ciphertext ct_xy(x_encrypted.poly_modulus_degree(),
                             x_encrypted.coeff_modulus_size());
    std::vector<uint64_t> tmp_modulus(parms.coeff_modulus().size());
    for (size_t i = 0; i < parms.coeff_modulus().size(); i++) {
        tmp_modulus[i] = parms.coeff_modulus()[i].value();
    }
    hifive::gpu_ptr d_coeff_modulus = hifive::make_and_copy_gpu_ptr(
        tmp_modulus.data(), parms.coeff_modulus().size());

    hifive::HAdd(ct_xy, ct_x, ct_y, d_coeff_modulus);

    // Copy back to CPU
    ct_xy.CopyBack(x_encrypted);

    // Decrypt and Decode
    PhantomPlaintext x_decoded;
    std::vector<double> output;
    secret_key.decrypt(context, x_encrypted, x_decoded);
    encoder.decode(context, x_decoded, output);
    for (size_t i = 0; i < slot_count; i++) {
        EXPECT_NEAR(input[i] * 2, output[i], 1e-5);
    }
}

TEST(cuCKKS, HMult) {
    const uint64_t alpha = 3;
    const uint64_t poly_modulus_degree = 1 << 15;

    phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // Encode and Encrypt
    size_t slot_count = encoder.slot_count();
    const double scale = pow(2.0, 40);
    std::vector<double> input(slot_count, 0);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = i;
    }
    PhantomPlaintext x_plain;
    PhantomCiphertext x_encrypted, y_encrypted;
    encoder.encode(context, input, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_encrypted);
    public_key.encrypt_asymmetric(context, x_plain, y_encrypted);

    // Copy to GPU
    hifive::Ciphertext ct_x(x_encrypted);
    hifive::Ciphertext ct_y(y_encrypted);
    hifive::Ciphertext ct_xy(x_encrypted.poly_modulus_degree(),
                             x_encrypted.coeff_modulus_size());
    std::vector<uint64_t> tmp_modulus(parms.coeff_modulus().size());
    for (size_t i = 0; i < parms.coeff_modulus().size(); i++) {
        tmp_modulus[i] = parms.coeff_modulus()[i].value();
    }
    hifive::gpu_ptr d_coeff_modulus = hifive::make_and_copy_gpu_ptr(
        tmp_modulus.data(), parms.coeff_modulus().size());

    hifive::HMult(ct_xy, ct_x, ct_y, d_coeff_modulus);

    // Copy back to CPU
    ct_xy.CopyBack(x_encrypted);

    // Decrypt and Decode
    PhantomPlaintext x_decoded;
    std::vector<double> output;
    secret_key.decrypt(context, x_encrypted, x_decoded);
    encoder.decode(context, x_decoded, output);
    for (size_t i = 0; i < slot_count; i++) {
        EXPECT_NEAR(input[i] * input[i], output[i], 1e-6);
    }
}