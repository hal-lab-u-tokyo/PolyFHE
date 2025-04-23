#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "polyfhe/core/logger.hpp"

namespace polyfhe {
namespace core {
class Edge;
class Node;

enum class VariableType { U64, U64_PTR };

enum class MemoryAccessPattern {
    LimbWise,    // e.g., NTT
    SlotWise,    // e.g., BConv
    ElementWise, // e.g., Add
    NoAccess,    // e.g., Init, End
    YetSet,
};

enum class BlockPhase {
    NTTPhase0,
    NTTPhase1,
    NTTPhase2,
};

inline std::ostream &operator<<(std::ostream &os, BlockPhase phase) {
    switch (phase) {
    case BlockPhase::NTTPhase0:
        os << "NTTPhase0";
        break;
    case BlockPhase::NTTPhase1:
        os << "NTTPhase1";
        break;
    case BlockPhase::NTTPhase2:
        os << "NTTPhase2";
        break;
    default:
        os << "Unknown";
        break;
    }
    return os;
}

enum class OpType {
    // For PolyGraph
    Add,
    Sub,
    Mult,
    Decomp,
    BConv,
    ModDown,
    ModUp,
    NTT,
    NTTPhase1,
    NTTPhase2,
    iNTT,
    iNTTPhase1,
    iNTTPhase2,
    End,
    Init,
    // For FHEGraph
    // TODO: separate FHEGraph and PolyGraph
    HAdd,
    HMult
};
std::string toStringOpType(OpType op_type);
std::string toString(BlockPhase block_phase);
MemoryAccessPattern OpType_access_pattern(OpType op_type);
bool is_ntt_op(OpType op_type);

class Node : public std::enable_shared_from_this<Node> {
public:
    Node() {};
    explicit Node(std::string op_name);
    explicit Node(OpType op_type);
    virtual ~Node() = default;

    // Edge
    void add_incoming(std::shared_ptr<Edge> edge) {
        m_in_edges.push_back(edge);
    }
    void add_outgoing(std::shared_ptr<Edge> edge) {
        m_out_edges.push_back(edge);
    }
    std::vector<std::shared_ptr<Edge>> &get_in_edges() { return m_in_edges; }
    std::vector<std::shared_ptr<Edge>> &get_out_edges() { return m_out_edges; }
    std::vector<VariableType> get_input_types();
    std::vector<VariableType> get_output_types();
    int get_idx_in_inedge(std::shared_ptr<Edge> edge) const {
        for (size_t i = 0; i < m_in_edges.size(); i++) {
            if (m_in_edges[i] == edge) {
                return i;
            }
        }
        return -1;
    }
    int get_idx_in_outedge(std::shared_ptr<Edge> edge) const {
        for (size_t i = 0; i < m_out_edges.size(); i++) {
            if (m_out_edges[i] == edge) {
                return i;
            }
        }
        return -1;
    }

    // Operation
    virtual OpType get_op_type() { return m_op_type; }
    std::string get_op_type_str() { return toStringOpType(m_op_type); }
    void set_op_type(OpType op_type) { m_op_type = op_type; }
    std::string get_op_name() {
        return toStringOpType(m_op_type) + "_" + std::to_string(m_id);
    }
    virtual std::vector<std::shared_ptr<Node>> get_nodes() {
        return {shared_from_this()};
    }

    // Access pattern
    void set_access_pattern(MemoryAccessPattern access_pattern) {
        m_access_pattern = access_pattern;
    }
    MemoryAccessPattern get_access_pattern() { return m_access_pattern; }
    bool if_one_to_one() {
        return (m_out_edges.size() == 1) && (m_in_edges.size() == 1);
    }

    // ID
    void set_id(int id) { m_id = id; }
    int get_id() { return m_id; }

    // Only for lowerings
    void add_top_poly_op(std::shared_ptr<Node> node) {
        m_top_poly_ops.push_back(node);
    }
    void add_bottom_poly_op(std::shared_ptr<Node> node) {
        m_bottom_poly_ops.push_back(node);
    }
    std::vector<std::shared_ptr<Node>> &get_top_poly_ops() {
        return m_top_poly_ops;
    }
    std::vector<std::shared_ptr<Node>> &get_bottom_poly_ops() {
        return m_bottom_poly_ops;
    }

    // Block phase
    void set_block_phase(BlockPhase block_phase) {
        m_block_phase = block_phase;
    }
    BlockPhase get_block_phase() { return m_block_phase; }

    // Subgraph
    void set_idx_subgraph(int idx) { idx_subgraph = idx; }
    int get_idx_subgraph() { return idx_subgraph; }

protected:
    OpType m_op_type;
    std::vector<std::shared_ptr<Edge>> m_in_edges;
    std::vector<std::shared_ptr<Edge>> m_out_edges;
    int m_id;
    MemoryAccessPattern m_access_pattern = MemoryAccessPattern::YetSet;
    BlockPhase m_block_phase;
    int idx_subgraph = -1;

private:
    // Only for lowerings
    std::vector<std::shared_ptr<Node>> m_top_poly_ops;
    std::vector<std::shared_ptr<Node>> m_bottom_poly_ops;
};

} // namespace core
} // namespace polyfhe