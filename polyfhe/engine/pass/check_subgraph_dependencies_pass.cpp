#include "polyfhe/engine/pass/check_subgraph_dependencies_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {
bool CheckSubgraphDependenciesPass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CheckSubgraphDependenciesPass\n");

    std::vector<std::shared_ptr<polyfhe::core::SubGraph>> subgraphs =
        graph->get_subgraphs();
    for (int i = 0; i < subgraphs.size(); i++) {
        auto sgraph = subgraphs[i];

        if (i == subgraphs.size() - 1) {
            sgraph->set_require_devicesync(true);
            break;
        }

        auto next_sgraph = subgraphs[i + 1];

        // Check if the output edge of current subgraph is included
        // in the input edge of next subgraph
        bool has_dependency = false;
        std::vector<std::shared_ptr<core::Edge>> inedges;
        for (auto node : next_sgraph->get_nodes()) {
            for (auto inedge : node->get_in_edges()) {
                if (inedge->get_level() == core::EdgeLevel::Global) {
                    inedges.push_back(inedge);
                }
            }
        }

        for (auto node : sgraph->get_nodes()) {
            for (auto outedge : node->get_out_edges()) {
                for (auto inedge : inedges) {
                    if (outedge == inedge) {
                        has_dependency = true;
                        break;
                    }
                }
            }
        }

        if (has_dependency) {
            sgraph->set_require_devicesync(true);
            LOG_INFO("Subgraph %d requires devicesync with subgraph %d\n", i,
                     i + 1);
        } else {
            sgraph->set_require_devicesync(false);
            LOG_INFO(
                "Subgraph %d does not require devicesync with subgraph %d\n", i,
                i + 1);
        }
    }

    return true;
}
} // namespace engine
} // namespace polyfhe