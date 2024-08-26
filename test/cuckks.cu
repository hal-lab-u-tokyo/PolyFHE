#include "test.h"

TEST(cuCKKS, Params) {
    // SEAL
    std::vector v_alpha = {1, 2, 3, 4, 15};
    for (auto alpha : v_alpha) {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        switch (alpha) {
        case 1:
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));

            break;
        case 2:
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 60, 60}));
            break;
        case 3:
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 60, 60, 60}));
            break;
        case 4:
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 60, 60, 60, 60}));
            break;
        case 15:
            poly_modulus_degree = 1 << 16;
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_modulus_degree,
                {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
            break;
        default:
            throw std::invalid_argument("Invalid alpha");
        }

        seal::SEALContext context(parms);
    }
}

TEST(cuCKKS, Encrypt) {
    const uint64_t alpha = 3;
    const uint64_t poly_modulus_degree = 1 << 15;

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));

    seal::SEALContext context(parms);
    seal::KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::Encryptor encryptor(context, public_key);
    seal::Decryptor decryptor(context, secret_key);
    seal::CKKSEncoder encoder(context);

    // Encode and Encrypt
    size_t slot_count = encoder.slot_count();
    const double scale = pow(2.0, 40);
    std::vector<double> input(slot_count, 0);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = i;
    }
    seal::Plaintext x_plain;
    seal::Ciphertext x_encrypted;
    encoder.encode(input, scale, x_plain);
    encryptor.encrypt(x_plain, x_encrypted);

    // Decrypt and Decode
    seal::Plaintext x_decoded;
    std::vector<double> output;
    decryptor.decrypt(x_encrypted, x_decoded);
    encoder.decode(x_decoded, output);
    for (size_t i = 0; i < slot_count; i++) {
        EXPECT_NEAR(input[i], output[i], 1e-6);
    }
}

TEST(cuCKKS, HAdd) {
    const uint64_t alpha = 3;
    const uint64_t poly_modulus_degree = 1 << 15;

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 60, 60, 60}));

    seal::SEALContext context(parms);
    seal::KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::Encryptor encryptor(context, public_key);
    seal::Decryptor decryptor(context, secret_key);
    seal::CKKSEncoder encoder(context);

    // Encode and Encrypt
    size_t slot_count = encoder.slot_count();
    const double scale = pow(2.0, 40);
    std::vector<double> input(slot_count, 0);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = i;
    }
    seal::Plaintext x_plain;
    seal::Ciphertext x_encrypted, y_encrypted;
    encoder.encode(input, scale, x_plain);
    encryptor.encrypt(x_plain, x_encrypted);
    encryptor.encrypt(x_plain, y_encrypted);

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

    hifive::HAdd(context, ct_xy, ct_x, ct_y, d_coeff_modulus);

    // Copy back to CPU
    std::cout << "Copying back to CPU" << std::endl;
    ct_xy.CopyBack(x_encrypted);

    // Decrypt and Decode
    seal::Plaintext x_decoded;
    std::vector<double> output;
    decryptor.decrypt(x_encrypted, x_decoded);
    encoder.decode(x_decoded, output);
    for (size_t i = 0; i < slot_count; i++) {
        // EXPECT_NEAR(input[i], output[i], 1e-6);
    }
}