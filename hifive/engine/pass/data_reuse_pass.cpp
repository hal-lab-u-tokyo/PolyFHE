#include "hifive/engine/pass/data_reuse_pass.hpp"

#include "hifive/core/logger.hpp"
#include "hifive/core/param.hpp"
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
        return src->get_access_pattern() !=
               hifive::core::MemoryAccessPattern::LimbWise;
    case hifive::core::MemoryAccessPattern::NotDefined:
        return false;
    case hifive::core::MemoryAccessPattern::YetSet:
        LOG_ERROR("Yetset access pattern\n");
        return false;
    default:
        LOG_ERROR("Unknown access pattern\n");
        return false;
    }
}

uint64_t CalculateSharedMemSizePerEdge(
    std::shared_ptr<hifive::core::Edge> edge) {
    uint64_t size = 1;
    // width
    switch (edge->get_src()->get_block_phase()) {
    case hifive::core::BlockPhase::NTTPhase1:
        size *= hifive::N1;
        break;
    case hifive::core::BlockPhase::NTTPhase2:
        size *= hifive::N2;
        break;
    default:
        break;
    }

    // height
    size *= edge->get_shape(1);
    return size * sizeof(uint64_t);
}

uint64_t CalculateSubgraphSharedMemFootprint(
    std::shared_ptr<hifive::core::Node> node,
    std::vector<std::shared_ptr<hifive::core::Edge>>& visited) {
    uint64_t footprint = 0;
    for (auto edge : node->get_out_edges()) {
        auto found = std::find(visited.begin(), visited.end(), edge);
        if (found != visited.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            visited.push_back(edge);
            footprint += CalculateSharedMemSizePerEdge(edge);
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
            footprint += CalculateSharedMemSizePerEdge(edge);
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

    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_after.dot");
    return true;
}
} // namespace engine
} // namespace hifive