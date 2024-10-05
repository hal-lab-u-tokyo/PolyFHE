#pragma once

#include "hifive/core/graph/node.hpp"

namespace hifive {
namespace core {
class Edge {
public:
private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;
};
} // namespace core
} // namespace hifive