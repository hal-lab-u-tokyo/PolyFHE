#pragma once

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
};

class Node {
public:
    Node(){};
    explicit Node(std::string op_type);
    virtual ~Node() = default;

    void add_incoming(std::shared_ptr<Edge> edge) { m_in_edges.insert(edge); }
    void add_outgoing(std::shared_ptr<Edge> edge) { m_out_edges.insert(edge); }
    std::set<std::shared_ptr<Edge>> &get_in_edges() { return m_in_edges; }
    std::set<std::shared_ptr<Edge>> &get_out_edges() { return m_out_edges; }

    virtual std::string get_op_type() { return m_op_type; }
    void set_op_type(std::string op_type) { m_op_type = op_type; }
    std::string get_op_name() { return m_op_type + std::to_string(m_id); }
    MemoryAccessPattern get_access_pattern() { return m_access_pattern; }

    void set_id(int id) { m_id = id; }
    int get_id() { return m_id; }

    std::vector<VariableType> get_input_types();
    std::vector<VariableType> get_output_types();

    bool if_one_to_one() {
        return (m_out_edges.size() == 1) && (m_in_edges.size() == 1);
    }

protected:
    std::string m_op_type;
    std::set<std::shared_ptr<Edge>> m_in_edges;
    std::set<std::shared_ptr<Edge>> m_out_edges;
    int m_id;
    MemoryAccessPattern m_access_pattern;
};

class FusedNode : public Node {
public:
    FusedNode(std::string op_type) : Node(op_type) {}
    FusedNode() : Node() {}
    void add_fused_node(std::shared_ptr<Node> node) {
        m_fused_nodes.push_back(node);
        std::string op_type = get_op_type();
        LOG_INFO("Fused node: %s\n", op_type.c_str());
        set_op_type(op_type);
    }
    std::string get_op_type() override {
        std::string n = "";
        for (auto node : m_fused_nodes) {
            n += node->get_op_type() + "_";
        }
        n.pop_back();
        return n;
    }

private:
    std::vector<std::shared_ptr<Node>> m_fused_nodes;
};

} // namespace core
} // namespace hifive