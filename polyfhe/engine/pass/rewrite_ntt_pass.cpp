#include "polyfhe/engine/pass/rewrite_ntt_pass.hpp"

#include "polyfhe/core/logger.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

bool RewriteNTTPass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running RewriteNTTPass\n");

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

        if (node->get_op_type() == core::OpType::NTT) {
            // Partision NTT Node into NTTPhase1 and NTTPhase2
            auto ntt_phase1 = std::make_shared<polyfhe::core::Node>();
            auto ntt_phase2 = std::make_shared<polyfhe::core::Node>();
            ntt_phase1->set_op_type(core::OpType::NTTPhase1);
            ntt_phase2->set_op_type(core::OpType::NTTPhase2);
            ntt_phase1->set_access_pattern(
                polyfhe::core::MemoryAccessPattern::LimbWise);
            ntt_phase2->set_access_pattern(
                polyfhe::core::MemoryAccessPattern::LimbWise);
            ntt_phase1->set_block_phase(core::BlockPhase::NTTPhase1);
            ntt_phase2->set_block_phase(core::BlockPhase::NTTPhase2);
            // Update graph
            graph->add_node(ntt_phase1);
            graph->add_node(ntt_phase2);
            assert(node->get_in_edges().size() == 1);
            auto inedge = node->get_in_edges()[0];
            inedge->set_dst(ntt_phase1);
            ntt_phase1->add_incoming(inedge);
            for (auto edge : node->get_out_edges()) {
                edge->set_src(ntt_phase2);
                ntt_phase2->add_outgoing(edge);
            }
            auto new_edge =
                std::make_shared<polyfhe::core::Edge>(ntt_phase1, ntt_phase2);
            new_edge->set_start_limb(inedge->get_start_limb());
            new_edge->set_end_limb(inedge->get_end_limb());
            new_edge->set_limb(inedge->get_limb());
            new_edge->update_name();
            new_edge->set_level(polyfhe::core::EdgeLevel::Global);
            ntt_phase1->add_outgoing(new_edge);
            ntt_phase2->add_incoming(new_edge);

            graph->remove_node(node);
            node = ntt_phase2;
        } else if (node->get_op_type() == core::OpType::iNTT) {
            // Partision iNTT Node into iNTTPhase1 and iNTTPhase2
            auto intt_phase1 = std::make_shared<polyfhe::core::Node>();
            auto intt_phase2 = std::make_shared<polyfhe::core::Node>();
            intt_phase1->set_op_type(core::OpType::iNTTPhase1);
            intt_phase2->set_op_type(core::OpType::iNTTPhase2);
            intt_phase1->set_access_pattern(
                polyfhe::core::MemoryAccessPattern::LimbWise);
            intt_phase2->set_access_pattern(
                polyfhe::core::MemoryAccessPattern::LimbWise);
            intt_phase1->set_block_phase(core::BlockPhase::NTTPhase1);
            intt_phase2->set_block_phase(core::BlockPhase::NTTPhase2);
            // Update graph
            graph->add_node(intt_phase1);
            graph->add_node(intt_phase2);
            assert(node->get_in_edges().size() == 1);
            auto inedge = node->get_in_edges()[0];
            inedge->set_dst(intt_phase2);
            intt_phase2->add_incoming(inedge);
            for (auto edge : node->get_out_edges()) {
                edge->set_src(intt_phase1);
                intt_phase1->add_outgoing(edge);
            }
            auto new_edge =
                std::make_shared<polyfhe::core::Edge>(intt_phase2, intt_phase1);
            new_edge->set_start_limb(inedge->get_start_limb());
            new_edge->set_end_limb(inedge->get_end_limb());
            new_edge->set_limb(inedge->get_limb());
            new_edge->update_name();
            new_edge->set_level(polyfhe::core::EdgeLevel::Global);
            intt_phase2->add_outgoing(new_edge);
            intt_phase1->add_incoming(new_edge);

            graph->remove_node(node);
            node = intt_phase1;
        }

        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    polyfhe::frontend::export_graph_to_dot(graph, "build/rewrite_ntt_pass.dot");

    return true;
}
} // namespace engine
} // namespace polyfhe