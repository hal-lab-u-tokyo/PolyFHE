#pragma once

#include <seal/seal.h>

void polyfhe_add(seal::Ciphertext &result, const seal::Ciphertext &x, const seal::Ciphertext &y, const seal::EncryptionParameters &parms);
void polyfhe_multiply(seal::Ciphertext &result, const seal::Ciphertext &x, const seal::Ciphertext &y, const seal::EncryptionParameters &parms);
void polyfhe_relinearize(seal::Ciphertext &ct, const seal::RelinKeys &key, const seal::EncryptionParameters &parms, const seal::SEALContext &context);