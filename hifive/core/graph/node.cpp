#include "hifive/core/graph/node.hpp"

#include "hifive/core/logger.hpp"
namespace hifive {
namespace core {

Node::Node(std::string op_type) : m_op_type(op_type), m_id(-1) {}

std::vector<VariableType> Node::get_input_types() {
    std::vector<VariableType> types;
    // TODO: consider other types
    for (auto edge : m_in_edges) {
        types.push_back(VariableType::U64_PTR);
    }
    return types;
}

std::vector<VariableType> Node::get_output_types() {
    std::vector<VariableType> types;
    // TODO: consider other types
    for (auto edge : m_out_edges) {
        types.push_back(VariableType::U64_PTR);
    }
    return types;
}
} // namespace core
} // namespace hifive