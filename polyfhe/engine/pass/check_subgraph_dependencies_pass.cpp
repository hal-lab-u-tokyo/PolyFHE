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

        // TODO: improve to use BlockPhase
        bool requires_devicesync = false;
        if (sgraph->if_contains_op(core::OpType::NTTPhase1) ||
            sgraph->if_contains_op(core::OpType::iNTTPhase1)) {
            if (next_sgraph->if_contains_op(core::OpType::NTTPhase2) ||
                next_sgraph->if_contains_op(core::OpType::iNTTPhase2)) {
                requires_devicesync = true;
            }
        } else if (sgraph->if_contains_op(core::OpType::NTTPhase2) ||
                   sgraph->if_contains_op(core::OpType::iNTTPhase2)) {
            if (next_sgraph->if_contains_op(core::OpType::NTTPhase1) ||
                next_sgraph->if_contains_op(core::OpType::iNTTPhase1)) {
                requires_devicesync = true;
            }
        }
        if (requires_devicesync) {
            sgraph->set_require_devicesync(true);
            LOG_INFO("Subgraph %s requires devicesync with subgraph %s\n",
                     sgraph->get_name().c_str(),
                     next_sgraph->get_name().c_str());
        } else {
            sgraph->set_require_devicesync(false);
        }
    }

    return true;
}
} // namespace engine
} // namespace polyfhe