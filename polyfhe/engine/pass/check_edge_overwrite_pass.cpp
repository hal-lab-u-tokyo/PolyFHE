#include "polyfhe/engine/pass/check_edge_overwrite_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {

std::shared_ptr<core::Edge> CanOverwriteOutEdge(
    const std::shared_ptr<core::Node> node) {
    if (node && node->get_out_edges().size() == 1) {
        auto overwrite_to = node->get_out_edges()[0];
        if (overwrite_to->get_level() == core::EdgeLevel::Global) {
            return overwrite_to;
        }
    }
    return nullptr;
}

bool CheckEdgeOverwritePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CheckEdgeOverwritePass\n");

    std::vector<std::shared_ptr<polyfhe::core::SubGraph>> subgraphs =
        graph->get_subgraphs();
    for (auto sgraph : subgraphs) {
        for (auto node : sgraph->get_nodes()) {
            for (auto outedge : node->get_out_edges()) {
                if (outedge->get_shared_counter() > 0) {
                    continue;
                }
                auto dst_node = outedge->get_dst();
                if (dst_node && dst_node->get_out_edges().size() == 1) {
                    auto overwrite_to = dst_node->get_out_edges()[0];
                    if (overwrite_to->get_level() ==
                        polyfhe::core::EdgeLevel::Global) {
                        outedge->set_overwrite_edge(overwrite_to);
                    } else {
                        // Check one hop further
                        auto next_overwrite =
                            CanOverwriteOutEdge(overwrite_to->get_dst());
                        if (next_overwrite) {
                            outedge->set_overwrite_edge(next_overwrite);
                        } else {
                            outedge->set_overwrite_edge(nullptr);
                        }
                    }
                }
            }
        }
    }
    return true;
}
} // namespace engine
} // namespace polyfhe