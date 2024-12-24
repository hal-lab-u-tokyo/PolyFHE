#include "hifive/engine/pass/analyze_node_local_pass.hpp"

namespace hifive {
namespace engine {
bool AnalyzeNodeLocalPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running AnalyzeNodeLocalPass\n");

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

        // Analyze required limb for the node

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    return true;
}
} // namespace engine
} // namespace hifive