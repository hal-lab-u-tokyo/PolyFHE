#include "polyfhe/engine/pass/check_edge_same_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {
bool CheckEdgeSamePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CheckEdgeSamePass\n");

    std::vector<std::shared_ptr<polyfhe::core::SubGraph>> subgraphs =
        graph->get_subgraphs();
    for (auto sgraph : subgraphs) {
        for (auto node : sgraph->get_nodes()) {
            const int n_outedges = node->get_out_edges().size();
            if (n_outedges >= 2) {
                if (node->get_op_type() == core::OpType::MultKeyAccum) {
                    // MultKeyAccum's outedges are not same
                    continue;
                }
                auto outedge_0 = node->get_out_edges()[0];
                for (int i = 1; i < n_outedges; i++) {
                    node->get_out_edges()[i]->set_same_edge(outedge_0);
                }
            }
        }
    }
    return true;
}
} // namespace engine
} // namespace polyfhe