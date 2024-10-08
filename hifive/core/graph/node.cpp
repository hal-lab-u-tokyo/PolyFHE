#include "hifive/core/graph/node.hpp"

#include "hifive/core/logger.hpp"
namespace hifive {
namespace core {
std::vector<VariableType> Node::get_input_types() {
    std::vector<VariableType> types;
    if (m_op_type == "Add") {
        types.push_back(VariableType::U64_PTR);
        types.push_back(VariableType::U64_PTR);
    } else {
        LOG_WARN("Node type not supported: %s\n", m_op_type.c_str());
    }
    return types;
}

std::vector<VariableType> Node::get_output_types() {
    std::vector<VariableType> types;
    if (m_op_type == "Add") {
        types.push_back(VariableType::U64_PTR);
    } else {
        LOG_WARN("Node type not supported: %s\n", m_op_type.c_str());
    }
    return types;
}
} // namespace core
} // namespace hifive