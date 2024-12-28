#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "hifive/core/graph/edge.hpp"
#include "hifive/core/graph/node.hpp"
namespace hifive {
namespace core {

enum class GraphType { FHE, Poly, Other };

class SubGraph {
public:
    void add_node(std::shared_ptr<Node> node) { m_nodes.push_back(node); }
    std::vector<std::shared_ptr<Node>> &get_nodes() { return m_nodes; }

    // idx
    void set_idx(int idx) { m_idx = idx; }
    int get_idx() { return m_idx; }

    // name
    std::string get_name() {
        if (m_name.empty()) {
            for (auto node : m_nodes) {
                m_name += node->get_op_type_str() + "_";
            }
            m_name.pop_back();
        }
        return m_name;
    }

    // block phase
    void set_block_phase(BlockPhase block_phase) {
        m_block_phase = block_phase;
    }
    BlockPhase get_block_phase() { return m_block_phase; }
    int get_nx_batch() { return m_nx_batch; }
    int get_ny_batch() { return m_ny_batch; }
    void set_nx_batch(int nx_batch) { m_nx_batch = nx_batch; }
    void set_ny_batch(int ny_batch) { m_ny_batch = ny_batch; }

private:
    std::vector<std::shared_ptr<Node>> m_nodes;
    int m_idx;
    std::string m_name;

    BlockPhase m_block_phase;
    int m_nx_batch;
    int m_ny_batch;
};

class Graph {
public:
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst);
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst,
                  std::string label);
    void add_node(std::shared_ptr<Node> node);
    void remove_node(std::shared_ptr<Node> node);

    std::vector<std::shared_ptr<Node>> &get_nodes() { return m_nodes; }

    // Get the number of nodes in the graph.
    // Don't use get_nodes().size() to get number of nodes
    // because get_nodes().size() returns the number of nodes including nullptr
    int get_nodes_size() {
        int count = 0;
        for (auto node : m_nodes) {
            if (node != nullptr) {
                count++;
            }
        }
        return count;
    }

    void set_init_node(std::shared_ptr<Node> node) { m_init_node = node; }
    void set_exit_node(std::shared_ptr<Node> node) { m_exit_node = node; }
    std::shared_ptr<Node> get_init_node() { return m_init_node; }
    std::shared_ptr<Node> get_exit_node() { return m_exit_node; }
    int get_init_node_id() {
        assert(m_init_node->get_id() == 0);
        return m_init_node->get_id();
    }

    GraphType get_graph_type() { return m_graph_type; }
    void set_graph_type(GraphType graph_type) { m_graph_type = graph_type; }

    // Subgraph
    void add_subgraph(std::shared_ptr<SubGraph> subgraph) {
        subgraph->set_idx(m_subgraphs.size());
        m_subgraphs.push_back(subgraph);
    }
    std::vector<std::shared_ptr<SubGraph>> &get_subgraphs() {
        return m_subgraphs;
    }

private:
    std::vector<std::shared_ptr<Node>> m_nodes;
    std::shared_ptr<Node> m_init_node;
    std::shared_ptr<Node> m_exit_node;
    GraphType m_graph_type;

    std::vector<std::shared_ptr<SubGraph>> m_subgraphs;
};

} // namespace core
} // namespace hifive