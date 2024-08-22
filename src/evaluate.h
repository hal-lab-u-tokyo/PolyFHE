#pragma once

#include <memory>

#include "gpucontext.h"
#include "phantom.h"

namespace hifive {

class Evaluator {
public:
    Evaluator(PhantomContext &phantom);
    ~Evaluator() = default;

    void Add(const PhantomContext &context, PhantomCiphertext &result,
             const PhantomCiphertext &ct0, const PhantomCiphertext &ct1);

private:
    std::unique_ptr<GPUContext> gpu_context_;
    // std::unique_ptr<PhantomContext> phantom_context_;
};

} // namespace hifive