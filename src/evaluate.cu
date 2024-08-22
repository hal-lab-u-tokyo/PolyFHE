#include "evaluate.h"

#include <iostream>

namespace hifive {

void Evaluator::Add(PhantomCiphertext &result, const PhantomCiphertext &op0, const PhantomCiphertext &op1) {
    if (op0.chain_index() != op1.chain_index()){
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (op0.is_ntt_form() != op1.is_ntt_form()) {
        throw std::invalid_argument("NTT form mismatch");
    }
    if (std::abs(op0.scale() - op1.scale()) > 1e-6) {
        throw std::invalid_argument("scale mismatch");
    }
    if (op0.size() != op1.size()) {
        throw std::invalid_argument("poly number mismatch");
    }
    std::cout << "HAdd" << std::endl;
}

}