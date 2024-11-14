#include "extract_subgraph_pass.hpp"

#include <optional>

#include "hifive/core/logger.hpp"
#include "hifive/core/param.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

void ExtractSubgraph(
    std::shared_ptr<hifive::core::Node> node,
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph) {
    for (auto edge : node->get_out_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_dst());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            subgraph.push_back(edge->get_dst());
            ExtractSubgraph(edge->get_dst(), subgraph);
        }
    }
    for (auto edge : node->get_in_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_src());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            subgraph.push_back(edge->get_src());
            ExtractSubgraph(edge->get_src(), subgraph);
        }
    }
}

std::optional<std::vector<std::shared_ptr<hifive::core::Node>>>
CheckIfSubgraphNodesVisited(std::shared_ptr<hifive::core::Node> node,
                            std::vector<bool>& visited) {
    std::vector<std::shared_ptr<hifive::core::Node>> subgraph;
    ExtractSubgraph(node, subgraph);
    for (auto subnode : subgraph) {
        if (!visited[subnode->get_id()]) {
            return std::nullopt;
        }
    }
    return std::make_optional(subgraph);
}

bool ExtractSubgraphPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running ExtractSubgraphPass\n");

    int idx_subgraph = 0;
    const int n = graph->get_nodes().size();
    std::vector<int> indegree(n, 0);
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    for (int i = 0; i < n; i++) {
        auto node = graph->get_nodes()[i];
        if (node == nullptr) {
            LOG_ERROR("Node is nullptr\n");
            continue;
        }
        indegree[i] = node->get_in_edges().size();
    }

    stack.push_back(graph->get_init_node_id());
    while (!stack.empty()) {
        int node_idx = stack.back();
        stack.pop_back();
        if (visited[node_idx]) {
            continue;
        }
        visited[node_idx] = true;
        LOG_INFO("Visiting node %s\n",
                 graph->get_nodes()[node_idx]->get_op_name().c_str());
        auto node = graph->get_nodes()[node_idx];
        if (node == nullptr) {
            LOG_ERROR("Node is nullptr\n");
            continue;
        }

        // Check if subgraph nodes are visited and define larger node
        std::optional<std::vector<std::shared_ptr<hifive::core::Node>>>
            has_visited_subnodes = CheckIfSubgraphNodesVisited(node, visited);
        if (has_visited_subnodes) {
            LOG_INFO("Subgraph visited around %s\n",
                     node->get_op_name().c_str());
            for (auto subnode : *has_visited_subnodes) {
                subnode->set_idx_subgraph(idx_subgraph);
            }
            idx_subgraph += 1;
        }

        for (auto edge : node->get_out_edges()) {
            int dst_id = edge->get_dst()->get_id();
            indegree[dst_id] -= 1;
            if (indegree[dst_id] == 0) {
                stack.push_back(dst_id);
            }
        }
    }

    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_extract_subgraph_pass.dot");
    return true;
}
} // namespace engine
} // namespace hifive