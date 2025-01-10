#include "calculate_smem_size_pass.hpp"

#include <optional>

#include "hifive/core/logger.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

bool CalculateSmemSizePass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running CalculateSmemSizePass\n");

    int idx_subgraph = 0;
    const int n = graph->get_nodes().size();
    std::vector<int> indegree(n, 0);
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    for (int i = 0; i < n; i++) {
        auto node = graph->get_nodes()[i];
        if (node == nullptr) {
            // Some node can be nullptr, i.e., NTT node is replaced with
            // NTTPhase1 and NTTPhase2 nodes
            continue;
        }
        indegree[i] = node->get_in_edges().size();
    }

    // Init node
    // Visit init node first because we don't include it in subgraph
    auto init_node = graph->get_init_node();
    visited[init_node->get_id()] = true;
    for (auto edge : init_node->get_out_edges()) {
        int dst_id = edge->get_dst()->get_id();
        indegree[dst_id] -= 1;
        if (indegree[dst_id] == 0) {
            stack.push_back(dst_id);
        }
    }

    while (!stack.empty()) {
        int node_idx = stack.back();
        stack.pop_back();
        if (visited[node_idx]) {
            continue;
        }
        visited[node_idx] = true;
        auto node = graph->get_nodes()[node_idx];
        if (node == nullptr) {
            continue;
        }

        for (auto edge : node->get_out_edges()) {
            int dst_id = edge->get_dst()->get_id();
            indegree[dst_id] -= 1;
            if (indegree[dst_id] == 0) {
                stack.push_back(dst_id);
            }
        }
    }

    for (auto subgraph : graph->get_subgraphs()) {
        LOG_INFO("Subgraph[%d]:\n", subgraph->get_idx());
        for (auto node : subgraph->get_nodes()) {
            LOG_INFO("  %s\n", node->get_op_name().c_str());
        }
    }
    return true;
}
} // namespace engine
} // namespace hifive