#pragma once

#include "phantom.h"

namespace hifive {

class Evaluator {
public:
  Evaluator(PhantomContext &phantom){};
  ~Evaluator() = default;

  void Add(PhantomCiphertext &result, const PhantomCiphertext &op0, const PhantomCiphertext &op1);
};

} // namespace hifive