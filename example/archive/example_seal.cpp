#include <iostream>
#include <vector>

#include "seal/seal.h"
#include "ckks.hpp"

void print_vec(const std::vector<double> &vec, int len = 8) {
    for (int i = 0; i < len; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void compare_result(const std::vector<double> &res, const std::vector<double> &expected){
    bool is_equal = true;
    for (size_t i = 0; i < res.size(); i++)
    {
        if (abs(res[i] - expected[i]) > 0.00001)
        {
            is_equal = false;
        }
    }
    std::cout << "Expected result: ";
    for (int i = 0; i < 8; i++)
    {
        std::cout << expected[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Decrypted result: ";
    for (int i = 0; i < 8; i++)
    {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl; 
    std::cout << "Diff: ";
    for (int i = 0; i < 8; i++)
    {
        std::cout << expected[i] - res[i] << " ";
    }
    std::cout << std::endl;
    
    if (is_equal)
    {
        std::cout << "Decryption successful!" << std::endl;
    }
    else
    {
        std::cout << "Decryption failed!" << std::endl;
    }
}

int main(){
    std::cout << "Starting CKKS example..." << std::endl;
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    double scale = pow(2.0, 40);

    seal::SEALContext context(parms);
    seal::KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    seal::Encryptor encryptor(context, public_key);
    seal::Evaluator evaluator(context);
    seal::CKKSEncoder encoder(context);
    seal::Decryptor decryptor(context, secret_key);

    size_t slot_count = encoder.slot_count();
    std::cout << "Number of slots: " << slot_count << std::endl;

    std::vector<double> input;
    std::vector<double> input2;
    for (size_t i = 0; i < slot_count; i++)
    {
        input.push_back(static_cast<double>(i) / (slot_count - 1));
        input2.push_back(static_cast<double>(i) / (slot_count - 1));
    }

    std::cout << "Input vector: " << std::endl;
    print_vec(input, 8);
    print_vec(input2, 8);
    
    seal::Plaintext pt_in1, pt_in2;
    seal::Ciphertext ct_in1, ct_in2;
    encoder.encode(input, scale, pt_in1);
    encoder.encode(input2, scale, pt_in2);
    encryptor.encrypt(pt_in1, ct_in1);
    encryptor.encrypt(pt_in2, ct_in2);


    // Evaluate Add
    std::cout << "### Evaluating Add ###" << std::endl;
    seal::Ciphertext ct_add;
    seal::Ciphertext ct_add_seal;
    seal::Plaintext pt_res_add, pt_res_add_seal;
    std::vector<double> vec_res_add;
    std::vector<double> vec_expected_add;
        
    polyfhe_add(ct_add, ct_in1, ct_in2, parms);
    evaluator.add(ct_in1, ct_in2, ct_add_seal);
    decryptor.decrypt(ct_add, pt_res_add);
    decryptor.decrypt(ct_add_seal, pt_res_add_seal);
    encoder.decode(pt_res_add, vec_res_add);
    for (size_t i = 0; i < slot_count; i++)
    {
        vec_expected_add.push_back(input[i] + input2[i]);
    }
    compare_result(vec_res_add, vec_expected_add);

    // Evaluate Multiply
    // SEAL's multiply
    // Given input tuples of polynomials x = (x[0], x[1], x[2]), y = (y[0], y[1]), computes
    // x = (x[0] * y[0], x[0] * y[1] + x[1] * y[0], x[1] * y[1])        
    std::cout << "### Evaluating Multiply ###" << std::endl;
    seal::Ciphertext ct_mult;
    seal::Ciphertext ct_mult_seal;
    seal::Plaintext pt_res_mult, pt_res_mult_seal;
    std::vector<double> vec_res_mult;
    std::vector<double> vec_expected_mult;
    evaluator.multiply(ct_in1, ct_in2, ct_mult_seal);
    evaluator.relinearize_inplace(ct_mult_seal, relin_keys);
    polyfhe_multiply(ct_mult, ct_in1, ct_in2, parms);
    polyfhe_relinearize(ct_mult, relin_keys, parms, context);

    // TODO: decrypt
    
    bool is_equal_mult = true;
    for (size_t i = 0; i < ct_mult_seal.size(); i++){
        for (size_t j = 0; j < ct_mult_seal.coeff_modulus_size(); j++){
            for (size_t k = 0; k < ct_mult_seal.poly_modulus_degree(); k++){
                size_t idx = j * ct_mult_seal.poly_modulus_degree() + k;
                if (ct_mult_seal.data(i)[idx] != ct_mult.data(i)[idx]){
                    std::cout << "Mismatch at index " << i << ", " << j << ", " << k << std::endl;
                    is_equal_mult = false;
                    break;
                }
            }
        }
    }
    if (is_equal_mult){
        std::cout << "Multiplication successful!" << std::endl;
    } else {
        std::cout << "Multiplication failed!" << std::endl;
    }

    
}