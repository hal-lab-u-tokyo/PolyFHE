#include "polyfhe/engine/pass/set_block_phase_pass.hpp"

#include "polyfhe/core/logger.hpp"
#include "polyfhe/frontend/exporter.hpp"
#include "set_block_phase_pass.hpp"

namespace polyfhe {
namespace engine {

bool SetBlockPhasePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running SetBlockPhasePass\n");

    // Topological sort using DFS
    const int n = graph->get_nodes().size();
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    stack.push_back(graph->get_init_node_id());
    graph->get_init_node()->set_block_phase(core::BlockPhase::NTTPhase2);

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

        polyfhe::core::BlockPhase next_block_phase = node->get_block_phase();
        if (node->get_op_type() == core::OpType::NTTPhase1 ||
            node->get_op_type() == core::OpType::iNTTPhase1) {
            node->set_block_phase(core::BlockPhase::NTTPhase1);
            next_block_phase = core::BlockPhase::NTTPhase1;
        } else if (node->get_op_type() == core::OpType::NTTPhase2 ||
                   node->get_op_type() == core::OpType::iNTTPhase2) {
            node->set_block_phase(core::BlockPhase::NTTPhase2);
            next_block_phase = core::BlockPhase::NTTPhase2;
        }

        for (auto edge : node->get_out_edges()) {
            edge->get_dst()->set_block_phase(next_block_phase);
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    return true;
}
} // namespace engine
} // namespace polyfhe