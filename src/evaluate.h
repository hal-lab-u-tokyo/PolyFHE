#pragma once

#include <memory>

#include "ciphertext.h"
#include "gpu_utils.h"

// Phantom
#include "phantom.h"

// SEAL
#include "seal/seal.h"

namespace hifive {

void HAdd(const seal::SEALContext &context, Ciphertext &result,
          const Ciphertext &ct0, const Ciphertext &ct1, const gpu_ptr &modulus);

void HMult(const seal::SEALContext &context, Ciphertext &result,
           const Ciphertext &ct0, const Ciphertext &ct1,
           const gpu_ptr &modulus);

void NTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx);

void iNTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx);

} // namespace hifive