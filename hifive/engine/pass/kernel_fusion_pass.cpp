#include "hifive/engine/pass/kernel_fusion_pass.hpp"

#include "hifive/core/logger.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {
bool KernelFusionPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running KernelFusionPass\n");
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_before_kernel_fusion_pass.dot");

    const int n = graph->get_nodes().size();
    LOG_INFO("Number of nodes before fusion: %d\n", graph->get_nodes_size());
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

        // Fuse if one-to-one
        if (node->get_access_pattern() ==
            hifive::core::MemoryAccessPattern::ElementWise) {
            auto edge_to_next = *node->get_out_edges().begin();
            auto next_node = edge_to_next->get_dst();

            // Check if next node is element-wise
            if (next_node->get_access_pattern() !=
                hifive::core::MemoryAccessPattern::ElementWise) {
                continue;
            }

            // Fuse
            auto fused_node = std::make_shared<hifive::core::FusedNode>();
            fused_node->add_fused_node(node);
            fused_node->add_fused_node(next_node);
            fused_node->set_access_pattern(
                hifive::core::MemoryAccessPattern::ElementWise);

            // TODO: rethinking the edge management
            for (auto edge : node->get_in_edges()) {
                fused_node->add_incoming(edge);
                edge->set_dst(fused_node);
            }
            for (auto edge : next_node->get_in_edges()) {
                if (edge->get_src() != node) {
                    fused_node->add_incoming(edge);
                    edge->set_dst(fused_node);
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_dst() != next_node) {
                    fused_node->add_outgoing(edge);
                    edge->set_src(fused_node);
                }
            }
            for (auto edge : next_node->get_out_edges()) {
                fused_node->add_outgoing(edge);
                edge->set_src(fused_node);
            }

            // Update graph
            graph->add_node(fused_node);
            graph->remove_node(node);
            graph->remove_node(next_node);
            LOG_INFO("Fused %s + %s -> %s\n", node->get_op_name().c_str(),
                     next_node->get_op_name().c_str(),
                     fused_node->get_op_name().c_str());

            // Update DFS stack
            visited.push_back(false);
            stack.push_back(fused_node->get_id());
        }

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    LOG_IMPORTANT("Number of nodes after fusion: %d\n",
                  graph->get_nodes_size());
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_after_kernel_fusion_pass.dot");
    return true;
}
} // namespace engine
} // namespace hifive