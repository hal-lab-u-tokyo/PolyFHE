#include "polyfhe/core/graph/node.hpp"

#include <map>
#include <sstream>
#include <string>

#include "polyfhe/core/logger.hpp"
namespace polyfhe {
namespace core {

bool is_ntt_op(OpType op_type) {
    return (op_type == OpType::NTT || op_type == OpType::iNTT ||
            op_type == OpType::NTTPhase1 || op_type == OpType::iNTTPhase1 ||
            op_type == OpType::NTTPhase2 || op_type == OpType::iNTTPhase2);
}

std::string to_str(OpType op_type) {
    switch (op_type) {
    case OpType::Add:
        return "Add";
    case OpType::Sub:
        return "Sub";
    case OpType::Mult:
        return "Mult";
    case OpType::MultKeyAccum:
        return "MultKeyAccum";
    case OpType::MultConst:
        return "MultConst";
    case OpType::Decomp:
        return "Decomp";
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
    case OpType::NTTPhase1:
        return "NTTPhase1";
    case OpType::NTTPhase2:
        return "NTTPhase2";
    case OpType::iNTTPhase1:
        return "iNTTPhase1";
    case OpType::iNTTPhase2:
        return "iNTTPhase2";
    case OpType::End:
        return "End";
    case OpType::Init:
        return "Init";
    default:
        return "Unknown";
    }
}

std::string toString(BlockPhase block_phase) {
    switch (block_phase) {
    case BlockPhase::NTTPhase0:
        return "NTTPhase0";
    case BlockPhase::NTTPhase1:
        return "NTTPhase1";
    case BlockPhase::NTTPhase2:
        return "NTTPhase2";
    default:
        return "Unknown";
    }
}

MemoryAccessPattern OpType_access_pattern(OpType op_type) {
    if (is_ntt_op(op_type)) {
        return MemoryAccessPattern::LimbWise;
    } else if (op_type == OpType::BConv || op_type == OpType::ModDown ||
               op_type == OpType::ModUp) {
        return MemoryAccessPattern::SlotWise;
    } else if (op_type == OpType::Add || op_type == OpType::Sub ||
               op_type == OpType::Mult || op_type == OpType::MultConst ||
               op_type == OpType::MultKeyAccum) {
        return MemoryAccessPattern::ElementWise;
    } else if (op_type == OpType::End || op_type == OpType::Init) {
        return MemoryAccessPattern::NoAccess;
    } else {
        return MemoryAccessPattern::YetSet;
    }
}

Node::Node(OpType op_type)
    : m_op_type(op_type), m_access_pattern(OpType_access_pattern(op_type)) {}

Node::Node(std::string op_label) : m_id(-1) {
    // We have to convert string op's name to OpType
    std::map<std::string, OpType> op_map = {
        {"Add", OpType::Add},
        {"Sub", OpType::Sub},
        {"Mult", OpType::Mult},
        {"MultConst", OpType::MultConst},
        {"MultKeyAccum", OpType::MultKeyAccum},
        {"Decomp", OpType::Decomp},
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

    // format of op_label: {op_name}_{info}
    std::vector<std::string> op_label_vec;
    std::stringstream ss(op_label);
    std::string token;
    while (std::getline(ss, token, '_')) {
        op_label_vec.push_back(token);
    }
    if (op_map.find(op_label_vec[0]) == op_map.end()) {
        LOG_ERROR("Unknown op_name: %s\n", op_label_vec[0].c_str());
        exit(1);
    }
    m_op_type = op_map[op_label_vec[0]];
    m_access_pattern = OpType_access_pattern(m_op_type);

    if (m_op_type == OpType::Init || m_op_type == OpType::End) {
        // No label
    } else if (m_op_type == OpType::MultKeyAccum) {
        // {op_name}_{start_idx}_{end_idx}_{beta}
        if (op_label_vec.size() != 4) {
            LOG_ERROR("Illegal op_label: %s\n", op_label.c_str());
        }
        set_limb_range(std::stoi(op_label_vec[1]), std::stoi(op_label_vec[2]));
        set_beta(std::stoi(op_label_vec[3]));
    } else if (m_op_type == OpType::BConv) {
        set_beta_idx(std::stoi(op_label_vec[1]));
    } else if (core::is_ntt_op(m_op_type)) {
        // {op_name}_{start_idx}_{end_idx}_{exclude_start_idx}_{exclude_end_idx}
        if (op_label_vec.size() != 5) {
            LOG_ERROR("Illegal op_label: %s\n", op_label.c_str());
        }
        set_limb_range(std::stoi(op_label_vec[1]), std::stoi(op_label_vec[2]));
        set_exclude_idx(std::stoi(op_label_vec[3]), std::stoi(op_label_vec[4]));
    } else {
        // {op_name}_{start_idx}_{end_idx}
        if (op_label_vec.size() != 3) {
            for (auto &s : op_label_vec) {
                LOG_INFO("%s\n", s.c_str());
            }
            LOG_ERROR("Illegal op_label: %s\n", op_label.c_str());
        }
        set_limb_range(std::stoi(op_label_vec[1]), std::stoi(op_label_vec[2]));
    }
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
} // namespace polyfhe