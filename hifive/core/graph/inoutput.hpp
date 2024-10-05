#pragma once

#include <vector>

namespace hifive {
namespace core {
class Shape : public std::vector<int> {};

class InOutput {
public:
private:
    Shape m_shape;
};
} // namespace core
} // namespace hifive