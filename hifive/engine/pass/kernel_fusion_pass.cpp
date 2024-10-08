#include "hifive/engine/pass/kernel_fusion_pass.hpp"

#include "hifive/core/logger.hpp"

namespace hifive {
namespace engine {
bool KernelFusionPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running KernelFusionPass\n");

    const int n = graph->get_nodes().size();
    LOG_INFO("Number of nodes: %d\n", n);
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    stack.push_back(graph->get_init_node_id());
    while (!stack.empty()) {
        int node_idx = stack.back();
        stack.pop_back();
        if (visited[node_idx]) {
            continue;
        }
        visited[node_idx] = true;

        auto node = graph->get_nodes()[node_idx];
        LOG_INFO("Visiting node%d %s\n", node->get_id(),
                 node->get_op_name().c_str());

        for (auto edge : node->get_out_edges()) {
            LOG_INFO("\tVisiting edge %s -> %s\n",
                     edge->get_src()->get_op_name().c_str(),
                     edge->get_dst()->get_op_name().c_str());
            stack.push_back(edge->get_dst()->get_id());
        }
    }
    return true;
}
} // namespace engine
} // namespace hifive