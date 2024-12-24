#include "hifive/engine/pass/analyze_intra_node_pass.hpp"

namespace hifive {
namespace engine {
bool AnalyzeIntraNodePass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running AnalyzeIntraNodePass\n");

    const int n = graph->get_nodes().size();
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
        if (node == nullptr) {
            // Fused node is nullptr
            continue;
        }

        // Analyze Memory Access Pattern
        if (node->get_access_pattern() == core::MemoryAccessPattern::YetSet) {
            LOG_ERROR("Memory Access Pattern is not set for node %s\n",
                      node->get_op_name().c_str());
        }

        // Analyze required limb for the node

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    return true;
}
} // namespace engine
} // namespace hifive