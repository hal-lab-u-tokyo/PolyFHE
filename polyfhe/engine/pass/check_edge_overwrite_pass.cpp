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
            if ((node->get_out_edges().size() == 1) &&
                (node->get_in_edges().size() == 1)) {
                auto inedge = node->get_in_edges()[0];
                auto outedge = node->get_out_edges()[0];
                if (inedge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    if (outedge->get_level() ==
                        polyfhe::core::EdgeLevel::Global) {
                        // Use bottom edge as overwrite edge
                        inedge->set_overwrite_edge(outedge);
                    }
                }
            }
        }
    }
    return true;
}
} // namespace engine
} // namespace polyfhe