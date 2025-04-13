#include "ckks.hpp"

#include <iostream>

using namespace std;

void polyfhe_add(seal::Ciphertext &result, const seal::Ciphertext &x,
                 const seal::Ciphertext &y,
                 const seal::EncryptionParameters &parms) {
    // Copy the input ciphertexts with malloc
    result = y;
    // Add
    for (size_t i = 0; i < x.coeff_modulus_size(); i++) {
        for (size_t j = 0; j < x.poly_modulus_degree(); j++) {
            size_t idx = i * x.poly_modulus_degree() + j;
            uint64_t mod = *parms.coeff_modulus()[i].data();
            result.data(0)[idx] = (x.data(0)[idx] + y.data(0)[idx]) % mod;
            result.data(1)[idx] = (x.data(1)[idx] + y.data(1)[idx]) % mod;
        }
    }
}

void polyfhe_multiply(seal::Ciphertext &result, const seal::Ciphertext &x,
                      const seal::Ciphertext &y,
                      const seal::EncryptionParameters &parms) {
    result = y;
    result.resize(3);
    for (size_t i = 0; i < x.coeff_modulus_size(); i++) {
        for (size_t j = 0; j < x.poly_modulus_degree(); j++) {
            size_t idx = i * x.poly_modulus_degree() + j;
            uint64_t mod = *parms.coeff_modulus()[i].data();
            uint128_t x0 = static_cast<uint128_t>(x.data(0)[idx]);
            uint128_t y0 = static_cast<uint128_t>(y.data(0)[idx]);
            uint128_t x1 = static_cast<uint128_t>(x.data(1)[idx]);
            uint128_t y1 = static_cast<uint128_t>(y.data(1)[idx]);
            result.data(0)[idx] = (x0 * y0) % mod;
            result.data(1)[idx] = (x0 * y1 + x1 * y0) % mod;
            result.data(2)[idx] = (x1 * y1) % mod;
        }
    }
}

void polyfhe_ntt(uint64_t *data, size_t modulus_size,
                 const seal::util::NTTTables *ntt_table) {
    size_t N = ntt_table[0].coeff_count();
    for (size_t idx_mod = 0; idx_mod < modulus_size; idx_mod++) {
        uint64_t q = ntt_table[idx_mod].modulus().value();
        uint64_t *data_i = data + idx_mod * N;
        size_t t = N;
        for (size_t m = 1; m < N; m *= 2) {
            t = t / 2;
            for (size_t k = 0; k < N; k += 2 * m) {
                for (size_t j = 0; j < m; j++) {
                    uint64_t S = ntt_table[idx_mod]
                                     .get_from_root_powers()[j * t]
                                     .operand;
                    uint64_t U = data_i[k + j];
                    uint64_t V =
                        (data_i[k + j + m] * static_cast<uint128_t>(S)) % q;
                    data_i[k + j] = (U + V) % q;
                    data_i[k + j + m] = (U - V + q) % q;
                }
            }
        }
    }
}

void polyfhe_intt(uint64_t *data, size_t modulus_size,
                  const seal::util::NTTTables *ntt_table) {
    size_t N = ntt_table[0].coeff_count();
    for (size_t idx_mod = 0; idx_mod < modulus_size; idx_mod++) {
        uint64_t q = ntt_table[idx_mod].modulus().value();
        uint64_t *data_i = data + idx_mod * N;
        size_t t, step;
        for (size_t m = N / 2; m >= 1; m /= 2) {
            step = m * 2;
            t = N / step;
            for (size_t k = 0; k < N; k += step) {
                for (size_t j = 0; j < m; j++) {
                    uint64_t S = ntt_table[idx_mod]
                                     .get_from_inv_root_powers()[j * t]
                                     .operand;
                    uint64_t U = data_i[k + j];
                    uint64_t V = data_i[k + j + m];
                    data_i[k + j] = (U + V) % q;
                    data_i[k + j + m] =
                        (((U - V + q) % q) * static_cast<uint128_t>(S)) % q;
                }
            }
        }
        for (size_t j = 0; j < N; j++) {
            data_i[j] = (static_cast<uint128_t>(data_i[j]) *
                         ntt_table[idx_mod].inv_degree_modulo().operand) %
                        q;
        }
    }
}

void polyfhe_relinearize(seal::Ciphertext &ct, const seal::RelinKeys &key,
                         const seal::EncryptionParameters &parms,
                         const seal::SEALContext &context) {
    std::shared_ptr<const seal::SEALContext::ContextData> key_context_data =
        context.key_context_data();
    const seal::util::NTTTables *key_ntt_table =
        key_context_data->small_ntt_tables();

    std::cout << "# before iNTT: ";
    for (int i = 0; i < 8; i++) {
        std::cout << ct.data(2)[i] << " ";
    }
    std::cout << std::endl;

    size_t decomp_modulus_size = parms.coeff_modulus().size();
    polyfhe_intt(ct.data(2), decomp_modulus_size, key_ntt_table);

    std::cout << "# after iNTT: ";
    for (int i = 0; i < 8; i++) {
        std::cout << ct.data(2)[i] << " ";
    }
    std::cout << std::endl;
}