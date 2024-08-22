#pragma once

#include "phantom.h"

namespace hifive {

class Evaluator {
public:
  Evaluator(PhantomContext &phantom){};
  ~Evaluator() = default;

  void Add(const PhantomContext &context, PhantomCiphertext &result, const PhantomCiphertext &ct0, const PhantomCiphertext &ct1);
};

} // namespace hifive