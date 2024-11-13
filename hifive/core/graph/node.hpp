#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "hifive/core/logger.hpp"

namespace hifive {
namespace core {
class Edge;
class Node;

enum class VariableType { U64, U64_PTR };
enum class MemoryAccessPattern {
    LimbWise,    // e.g., NTT
    SlotWise,    // e.g., BConv
    ElementWise, // e.g., Add
    NotDefined   // e.g., Init, End
};

class Node : public std::enable_shared_from_this<Node> {
public:
    Node(){};
    explicit Node(std::string op_type);
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
    virtual std::string get_op_type() { return m_op_type; }
    void set_op_type(std::string op_type) { m_op_type = op_type; }
    std::string get_op_name() { return m_op_type + std::to_string(m_id); }
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

protected:
    std::string m_op_type;
    std::vector<std::shared_ptr<Edge>> m_in_edges;
    std::vector<std::shared_ptr<Edge>> m_out_edges;
    int m_id;
    MemoryAccessPattern m_access_pattern;

private:
    // Only for lowerings
    std::vector<std::shared_ptr<Node>> m_top_poly_ops;
    std::vector<std::shared_ptr<Node>> m_bottom_poly_ops;
};

class FusedNode : public Node {
public:
    FusedNode(std::string op_type) : Node(op_type) {}
    FusedNode() : Node() {}
    void add_fused_node(std::shared_ptr<Node> node) {
        for (auto n : node->get_nodes()) {
            m_fused_nodes.push_back(n);
        }
        std::string op_type = get_op_type();
        set_op_type(op_type);
    }

    // Operation
    std::string get_op_type() override {
        std::string n = "";
        for (auto node : m_fused_nodes) {
            n += node->get_op_type() + "_";
        }
        n.pop_back();
        return n;
    }
    std::vector<std::shared_ptr<Node>> get_nodes() override {
        return m_fused_nodes;
    }

private:
    std::vector<std::shared_ptr<Node>> m_fused_nodes;
};

} // namespace core
} // namespace hifive