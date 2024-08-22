#include <gtest/gtest.h>

#include <vector>

#include "phantom.h"
#include "example.h"

TEST(GPUContextTest, GetGPUInfo) {
    EXPECT_EQ(1, 1);    
}


TEST(CKKS, Encrypt){
    std::vector v_alpha = {1, 2, 3, 4, 15};
    for (auto alpha: v_alpha){
        std::cout << "alpha: " << alpha << std::endl;
        phantom::EncryptionParameters parms(phantom::scheme_type::ckks);

        size_t poly_modulus_degree = 1 << 15;
        double scale = pow(2.0, 40);
        switch (alpha) {
            case 1:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(
                        phantom::arith::CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                                   40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
                break;
            case 2:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
                parms.set_special_modulus_size(alpha);
                break;
            case 3:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60}));
                parms.set_special_modulus_size(alpha);
                break;
            case 4:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
                // hybrid key-switching
                parms.set_special_modulus_size(alpha);
                break;
            case 15:
                poly_modulus_degree = 1 << 16;
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
                        poly_modulus_degree,
                        {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                         50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
                parms.set_special_modulus_size(alpha);
                scale = pow(2.0, 50);
                break;
            default:
                throw std::invalid_argument("unsupported alpha params");
        }
        PhantomContext context(parms);
        print_parameters(context);
    }

}