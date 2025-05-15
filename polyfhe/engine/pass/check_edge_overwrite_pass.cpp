#include "polyfhe/engine/pass/check_edge_overwrite_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {
bool CheckEdgeOverwritePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CheckEdgeOverwritePass\n");

    std::vector<std::shared_ptr<polyfhe::core::SubGraph>> subgraphs =
        graph->get_subgraphs();
    for (auto sgraph : subgraphs) {
        for (auto node : sgraph->get_nodes()) {
            for (auto outedge : node->get_out_edges()) {
                auto dst_node = outedge->get_dst();
                if (dst_node && dst_node->get_out_edges().size() == 1) {
                    auto overwrite_to = dst_node->get_out_edges()[0];
                    if (overwrite_to->get_level() !=
                        polyfhe::core::EdgeLevel::Global) {
                        continue;
                    }
                    outedge->set_overwrite_edge(overwrite_to);
                }
            }
        }
    }
    return true;
}
} // namespace engine
} // namespace polyfhe