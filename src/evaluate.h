#pragma once

#include <memory>

#include "gpucontext.h"
#include "phantom.h"

namespace hifive {

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