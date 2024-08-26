#include <gtest/gtest.h>

// SEAL
#include "seal/seal.h"

TEST(SEAL, Params) {
    // SEAL
    std::vector v_alpha = {1, 2, 3, 4, 15};
    for (auto alpha : v_alpha) {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
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