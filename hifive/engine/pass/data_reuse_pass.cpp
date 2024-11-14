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

        if (node->get_op_type() == "NTT") {
            // Partision NTT Node into NTTPhase1 and NTTPhase2
            auto ntt_phase1 = std::make_shared<hifive::core::Node>();
            auto ntt_phase2 = std::make_shared<hifive::core::Node>();
            ntt_phase1->set_op_type("NTTPhase1");
            ntt_phase2->set_op_type("NTTPhase2");
            ntt_phase1->set_access_pattern(
                hifive::core::MemoryAccessPattern::LimbWise);
            ntt_phase2->set_access_pattern(
                hifive::core::MemoryAccessPattern::LimbWise);
            // Update graph
            graph->add_node(ntt_phase1);
            graph->add_node(ntt_phase2);
            assert(node->get_out_edges().size() == 1);
            assert(node->get_in_edges().size() == 1);
            auto inedge = node->get_in_edges()[0];
            auto outedge = node->get_out_edges()[0];
            inedge->set_dst(ntt_phase1);
            ntt_phase1->add_incoming(inedge);
            outedge->set_src(ntt_phase2);
            ntt_phase2->add_outgoing(outedge);
            auto new_edge =
                std::make_shared<hifive::core::Edge>(ntt_phase1, ntt_phase2);
            // TODO: support if shape of inedge and outedge is different
            for (size_t i = 0; i < inedge->get_shape().size(); i++) {
                if (inedge->get_shape(i) != outedge->get_shape(i)) {
                    LOG_ERROR("Shape of inedge and outedge is different\n");
                }
            }
            new_edge->set_shape(inedge->get_shape());
            new_edge->update_name();
            new_edge->set_level(hifive::core::EdgeLevel::Global);
            ntt_phase1->add_outgoing(new_edge);
            ntt_phase2->add_incoming(new_edge);

            graph->remove_node(node);
            node = ntt_phase2;
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