#include "hifive/core/graph/graph.hpp"

#include "hifive/core/logger.h"

namespace hifive {
namespace core {

void Graph::add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst) {
    if (src == nullptr || dst == nullptr) {
        LOG_ERROR("src or dst is nullptr\n");
        exit(1);
    }
    LOG_INFO("Adding edge from %s to %s\n", src->get_op_name().c_str(),
             dst->get_op_name().c_str());
    std::shared_ptr<Edge> edge = std::make_shared<Edge>(src, dst);
    src->add_outgoing(edge);
    dst->add_incoming(edge);
}

void Graph::add_node(std::shared_ptr<Node> node) {
    int id = m_nodes.size();
    m_nodes.push_back(node);
    node->set_id(id);
}
} // namespace core
} // namespace hifive