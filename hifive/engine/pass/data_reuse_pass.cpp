#include "hifive/engine/pass/data_reuse_pass.hpp"

#include "hifive/core/logger.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

bool CanReuse(std::shared_ptr<hifive::core::Node> src,
              std::shared_ptr<hifive::core::Node> dst) {
    switch (dst->get_access_pattern()) {
    case hifive::core::MemoryAccessPattern::ElementWise:
        return true;
    case hifive::core::MemoryAccessPattern::SlotWise:
        return true;
    case hifive::core::MemoryAccessPattern::LimbWise:
        return src->get_access_pattern() ==
               hifive::core::MemoryAccessPattern::LimbWise;
    case hifive::core::MemoryAccessPattern::NotDefined:
        return false;
    default:
        LOG_ERROR("Unknown access pattern\n");
        return false;
    }
}

uint64_t CalculateSubgraphSharedMemFootprint(
    std::shared_ptr<hifive::core::Node> node,
    std::vector<std::shared_ptr<hifive::core::Edge>>& visited) {
    uint64_t footprint = 0;
    LOG_INFO("Footprint: %s\n", node->get_op_name().c_str());
    for (auto edge : node->get_out_edges()) {
        auto found = std::find(visited.begin(), visited.end(), edge);
        if (found != visited.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            visited.push_back(edge);
            footprint += edge->get_size_in_byte();
            footprint +=
                CalculateSubgraphSharedMemFootprint(edge->get_dst(), visited);
        }
    }
    for (auto edge : node->get_in_edges()) {
        auto found = std::find(visited.begin(), visited.end(), edge);
        if (found != visited.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            visited.push_back(edge);
            footprint += edge->get_size_in_byte();
            footprint +=
                CalculateSubgraphSharedMemFootprint(edge->get_src(), visited);
        }
    }
    return footprint;
}

bool DataReusePass::run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running DataReusePass\n");
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_before.dot");

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

        LOG_INFO("Visiting %s\n", node->get_op_name().c_str());
        for (auto edge : node->get_out_edges()) {
            edge->set_level(hifive::core::EdgeLevel::Shared);
            // if (!CanReuse(node, edge->get_dst())) {
            //     edge->set_level(hifive::core::EdgeLevel::Global);
            //     continue;
            // }
            LOG_INFO("Calculating footprint around.... %s\n",
                     node->get_op_name().c_str());
            std::vector<std::shared_ptr<hifive::core::Edge>> visited;
            uint64_t footprint =
                CalculateSubgraphSharedMemFootprint(node, visited);
            LOG_INFO("Total shared mem %s: %lu KB\n",
                     node->get_op_name().c_str(), footprint / 1000);
            // if (CalculateSubgraphSharedMemFootprint(edge->get_dst()) > 32) {
            //     edge->set_level(hifive::core::EdgeLevel::Global);
            //     continue;
            // }
            LOG_INFO("Reuse %s -> %s\n", node->get_op_name().c_str(),
                     edge->get_dst()->get_op_name().c_str());
        }

        /*
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
                    auto fused_node =
           std::make_shared<hifive::core::FusedNode>();
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
                    LOG_INFO("Fused %s + %s -> %s\n",
           node->get_op_name().c_str(), next_node->get_op_name().c_str(),
                             fused_node->get_op_name().c_str());

                    // Update DFS stack
                    visited.push_back(false);
                    stack.push_back(fused_node->get_id());
                }
        */

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_after.dot");
    return true;
}
} // namespace engine
} // namespace hifive