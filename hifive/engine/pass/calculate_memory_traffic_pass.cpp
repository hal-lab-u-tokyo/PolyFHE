#include "hifive/engine/pass/calculate_memory_traffic_pass.hpp"

namespace hifive {
namespace engine {
bool CalculateMemoryTrafficPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    uint64_t total_traffic = 0;
    LOG_INFO("Running CalculateMemoryTrafficPass\n");

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

        // Calculate memory traffic cost
        for (auto in_edge : node->get_in_edges()) {
        }

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    LOG_IMPORTANT("Total memory traffic: %lu KByte\n", total_traffic / 1000);
    return true;
}
} // namespace engine
} // namespace hifive