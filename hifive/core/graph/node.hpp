#pragma once

#include <memory>
#include <set>
#include <string>

#include "hifive/core/graph/inoutput.hpp"

namespace hifive {
namespace core {
class Edge;
class Node {
public:
private:
    std::string m_op_type;
    std::set<std::shared_ptr<Edge>> m_in_edges;
    std::set<std::shared_ptr<Edge>> m_out_edges;
    std::vector<std::shared_ptr<InOutput>> m_inputs;
    std::vector<std::shared_ptr<InOutput>> m_outputs;
};
} // namespace core
} // namespace hifive