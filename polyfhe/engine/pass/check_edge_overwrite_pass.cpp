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
            if (node->get_out_edges().size() == 1) {
                auto outedge = node->get_out_edges()[0];
                if (outedge->get_level() != polyfhe::core::EdgeLevel::Global) {
                    continue;
                }
                auto dst = node->get_out_edges()[0]->get_dst();
                if (dst->get_op_type() == polyfhe::core::OpType::End) {
                    continue;
                }
                if (dst->get_out_edges().size() > 0) {
                    auto nextedge = dst->get_out_edges()[0];
                    if (nextedge->get_level() ==
                        polyfhe::core::EdgeLevel::Global) {
                        nextedge->set_overwrite_edge(outedge);
                    }
                }
            }
        }
    }
    return true;
}
} // namespace engine
} // namespace polyfhe