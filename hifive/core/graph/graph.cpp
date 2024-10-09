#include "hifive/core/graph/graph.hpp"

#include "hifive/core/logger.hpp"

namespace hifive {
namespace core {

void Graph::add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst) {
    add_edge(src, dst, "");
}

void Graph::add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst,
                     std::string label) {
    if (src == nullptr || dst == nullptr) {
        LOG_ERROR("src or dst is nullptr\n");
        exit(1);
    }
    LOG_INFO("Adding edge from %s to %s\n", src->get_op_name().c_str(),
             dst->get_op_name().c_str());

    std::shared_ptr<Edge> edge = std::make_shared<Edge>(src, dst);
    src->add_outgoing(edge);
    dst->add_incoming(edge);

    if (!label.empty()) {
        // label is {shape0}_{shape1}
        const std::string delimiter = "_";
        std::string shape0_str = label.substr(0, label.find(delimiter));
        std::string shape1_str =
            label.substr(label.find(delimiter) + 1, label.length());
        if (shape0_str.empty() || shape1_str.empty()) {
            LOG_ERROR("Invalid label: %s\n", label.c_str());
            exit(1);
        }
        const int shape0 = std::stoi(shape0_str);
        const int shape1 = std::stoi(shape1_str);
        edge->set_shape({shape0, shape1});
    }
}

void Graph::add_node(std::shared_ptr<Node> node) {
    int id = m_nodes.size();
    m_nodes.push_back(node);
    node->set_id(id);
}

void Graph::remove_node(std::shared_ptr<Node> node) {
    for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
        if (*it == node) {
            // Since node_id is the index of the node in m_nodes,
            // we don't remove() the node from m_nodes to avoid
            // updating the node_id of other nodes.
            *it = nullptr;
            break;
        }
    }
}

} // namespace core
} // namespace hifive