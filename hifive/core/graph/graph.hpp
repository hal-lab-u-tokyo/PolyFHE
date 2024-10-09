#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "hifive/core/graph/edge.hpp"
#include "hifive/core/graph/node.hpp"
namespace hifive {
namespace core {
class Graph {
public:
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst);
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst,
                  std::string label);
    void add_node(std::shared_ptr<Node> node);
    void remove_node(std::shared_ptr<Node> node) {
        for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
            if (*it == node) {
                m_nodes.erase(it);
                break;
            }
        }
    }

    std::vector<std::shared_ptr<Node>> &get_nodes() { return m_nodes; }

    void set_init_node(std::shared_ptr<Node> node) { m_init_node = node; }
    void set_exit_node(std::shared_ptr<Node> node) { m_exit_node = node; }
    std::shared_ptr<Node> get_init_node() { return m_init_node; }
    std::shared_ptr<Node> get_exit_node() { return m_exit_node; }
    int get_init_node_id() {
        assert(m_init_node->get_id() == 0);
        return m_init_node->get_id();
    }

private:
    std::vector<std::shared_ptr<Node>> m_nodes;
    std::shared_ptr<Node> m_init_node;
    std::shared_ptr<Node> m_exit_node;
};
} // namespace core
} // namespace hifive