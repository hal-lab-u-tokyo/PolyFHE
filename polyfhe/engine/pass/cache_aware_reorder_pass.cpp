
#include "polyfhe/engine/pass/cache_aware_reorder_pass.hpp"

#include <shared_mutex>

#include "polyfhe/core/config.hpp"
#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

void SetSubgraphLimbRange(std::shared_ptr<polyfhe::core::Graph>& graph) {
    for (auto subgraph : graph->get_subgraphs()) {
        int start_limb = -1;
        int end_limb = -1;
        bool can_define_common_range = true;
        for (auto node : subgraph->get_nodes()) {
            if (node->get_access_pattern() ==
                core::MemoryAccessPattern::SlotWise) {
                can_define_common_range = false;
                break;
            }
            if (start_limb == -1 && end_limb == -1) {
                start_limb = node->get_start_limb();
                end_limb = node->get_end_limb();
            } else {
                if (node->get_start_limb() != start_limb ||
                    node->get_end_limb() != end_limb) {
                    can_define_common_range = false;
                    break;
                }
            }
        }
        if (can_define_common_range) {
            LOG_INFO("Set subgraph[%d] limb range: [%d, %d]\n",
                     subgraph->get_idx(), start_limb, end_limb);
            subgraph->set_limb_range(start_limb, end_limb);
        }
    }
}

bool CacheAwareReorderpass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CacheAwareReorderpass\n");

    SetSubgraphLimbRange(graph);

    return true;
}
} // namespace engine
} // namespace polyfhe