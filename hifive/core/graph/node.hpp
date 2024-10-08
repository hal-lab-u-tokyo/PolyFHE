#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace hifive {
namespace core {
class Edge;

enum class VariableType { U64, U64_PTR };
class Node {
public:
    Node(std::string op_type) : m_op_type(op_type), m_id(-1) {}

    void add_incoming(std::shared_ptr<Edge> edge) { m_in_edges.insert(edge); }
    void add_outgoing(std::shared_ptr<Edge> edge) { m_out_edges.insert(edge); }
    std::set<std::shared_ptr<Edge>> &get_in_edges() { return m_in_edges; }
    std::set<std::shared_ptr<Edge>> &get_out_edges() { return m_out_edges; }

    std::string &get_op_type() { return m_op_type; }
    std::string get_op_name() { return m_op_type + std::to_string(m_id); }
    void set_id(int id) { m_id = id; }
    int get_id() { return m_id; }

    std::vector<VariableType> get_input_types();
    std::vector<VariableType> get_output_types();

private:
    std::string m_op_type;
    std::set<std::shared_ptr<Edge>> m_in_edges;
    std::set<std::shared_ptr<Edge>> m_out_edges;

    int m_id;
};
} // namespace core
} // namespace hifive