#include "hifive/engine/pass/set_block_phase_pass.hpp"

#include "hifive/core/logger.hpp"
#include "hifive/frontend/exporter.hpp"
#include "set_block_phase_pass.hpp"

namespace hifive {
namespace engine {

bool SetBlockPhasePass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running SetBlockPhasePass\n");

    // Topological sort using DFS
    const int n = graph->get_nodes().size();
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    stack.push_back(graph->get_init_node_id());
    graph->get_init_node()->set_block_phase(core::BlockPhase::NTTPhase1);

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

        hifive::core::BlockPhase next_block_phase = node->get_block_phase();
        if (node->get_op_type() == "NTTPhase1") {
            next_block_phase = core::BlockPhase::NTTPhase1;
        } else if (node->get_op_type() == "NTTPhase2") {
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
} // namespace hifive