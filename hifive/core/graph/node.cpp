#include "hifive/core/graph/node.hpp"

#include <map>
#include <string>

#include "hifive/core/logger.hpp"
namespace hifive {
namespace core {

std::string to_string(OpType op_type) {
    switch (op_type) {
    case OpType::Add:
        return "Add";
    case OpType::Sub:
        return "Sub";
    case OpType::Mult:
        return "Mult";
    case OpType::BConv:
        return "BConv";
    case OpType::ModDown:
        return "ModDown";
    case OpType::ModUp:
        return "ModUp";
    case OpType::NTT:
        return "NTT";
    case OpType::iNTT:
        return "iNTT";
    case OpType::End:
        return "End";
    case OpType::Init:
        return "Init";
    default:
        return "Unknown";
    }
}

Node::Node(std::string op_name) : m_id(-1) {
    std::map<std::string, OpType> op_map = {
        {"Add", OpType::Add},
        {"Sub", OpType::Sub},
        {"Mult", OpType::Mult},
        {"BConv", OpType::BConv},
        {"ModDown", OpType::ModDown},
        {"ModUp", OpType::ModUp},
        {"NTT", OpType::NTT},
        {"iNTT", OpType::iNTT},
        {"NTTPhase1", OpType::NTTPhase1},
        {"NTTPhase2", OpType::NTTPhase2},
        {"iNTTPhase1", OpType::iNTTPhase1},
        {"iNTTPhase2", OpType::iNTTPhase2},
        {"End", OpType::End},
        {"Init", OpType::Init},
        {"HAdd", OpType::HAdd},
        {"HMult", OpType::HMult},
    };
    if (op_map.find(op_name) == op_map.end()) {
        LOG_ERROR("Unknown op_name: %s\n", op_name.c_str());
        exit(1);
    }
    m_op_type = op_map[op_name];
}

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