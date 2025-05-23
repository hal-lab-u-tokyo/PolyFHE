#include "polyfhe/engine/pass/sort_subgraph_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {

bool has_sorted(
    std::shared_ptr<core::Node> node,
    std::vector<std::shared_ptr<core::SubGraph>>& sorted_subgraphs) {
    for (auto subgraph : sorted_subgraphs) {
        for (auto n : subgraph->get_nodes()) {
            if (n == node) {
                return true;
            }
        }
    }
    return false;
}

bool SortSubgraphPass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running SortSubgraphPass\n");

    std::vector<std::shared_ptr<core::SubGraph>> sorted_subgraphs;
    std::vector<std::shared_ptr<core::SubGraph>> waiting_subgraphs;

    for (auto subgraph : graph->get_subgraphs()) {
        bool is_ready = true;
        auto node = subgraph->get_nodes()[0];
        for (auto inedge : node->get_in_edges()) {
            auto src = inedge->get_src();
            if (src->get_op_type() == core::OpType::Init) {
                continue;
            }
            if (has_sorted(src, sorted_subgraphs)) {
                printf("Node %s is ready for node %s\n",
                       src->get_op_name().c_str(), node->get_op_name().c_str());
                continue;
            } else {
                is_ready = false;
                break;
            }
        }

        if (is_ready) {
            sorted_subgraphs.push_back(subgraph);
        } else {
            waiting_subgraphs.push_back(subgraph);
        }
    }

    // TODO: recursive
    for (auto subgraph : waiting_subgraphs) {
        sorted_subgraphs.push_back(subgraph);
    }

    graph->set_subgraphs(sorted_subgraphs);

    return true;
}
} // namespace engine
} // namespace polyfhe