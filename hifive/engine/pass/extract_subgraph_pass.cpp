#include "extract_subgraph_pass.hpp"

#include "hifive/core/logger.hpp"
#include "hifive/core/param.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

bool ExtractSubgraphPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running ExtractSubgraphPass\n");
    /*
        // Topological sort using DFS
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
                LOG_ERROR("Node is nullptr\n");
                continue;
            }

            for (auto edge : node->get_out_edges()) {
                edge->set_level(hifive::core::EdgeLevel::Shared);
                if (!CanReuse(node, edge->get_dst())) {
                    edge->set_level(hifive::core::EdgeLevel::Global);
                    continue;
                }
                LOG_INFO("Calculating footprint around.... %s\n",
                         node->get_op_name().c_str());
                std::vector<std::shared_ptr<hifive::core::Edge>> visited;
                uint64_t footprint_kb =
                    CalculateSubgraphSharedMemFootprint(node, visited) / 1000;
                LOG_INFO("Total shared mem %s: %lu KB\n",
                         node->get_op_name().c_str(), footprint_kb);
                if (footprint_kb > 120) {
                    edge->set_level(hifive::core::EdgeLevel::Global);
                    continue;
                }
                LOG_INFO("Reuse %s -> %s\n", node->get_op_name().c_str(),
                         edge->get_dst()->get_op_name().c_str());
            }

            for (auto edge : node->get_out_edges()) {
                stack.push_back(edge->get_dst()->get_id());
            }
        }
    */
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_extract_subgraph_pass.dot");
    return true;
}
} // namespace engine
} // namespace hifive