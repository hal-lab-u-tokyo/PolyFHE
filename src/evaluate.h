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

void NTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx);

void iNTT(DNTTTable &d_ntt_table, gpu_ptr &a, int batch_size, int start_idx);

class Evaluator {
public:
    Evaluator();
    ~Evaluator() = default;

    void Add(const PhantomContext &context, PhantomCiphertext &result,
             const PhantomCiphertext &ct0, const PhantomCiphertext &ct1);

    void Mult(const PhantomContext &context, PhantomCiphertext &result,
              const PhantomCiphertext &ct0, const PhantomCiphertext &ct1);

    void Relin(const PhantomContext &context, PhantomCiphertext &ct,
               const PhantomRelinKey &rk);

    void Rescale(const PhantomContext &context, PhantomCiphertext &ct);

private:
    void ModUp(uint64_t *dst, const uint64_t *in, const DNTTTable &ntt_tables,
               phantom::DRNSTool &rns_tool);

    std::unique_ptr<GPUContext> gpu_context_;
};

} // namespace hifive