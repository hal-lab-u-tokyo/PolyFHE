#include "extract_l2reuse_pass.hpp"

#include <optional>

#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/engine/pass/data_reuse_pass.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

bool ExtractL2ReusePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running ExtractL2ReusePass\n");

    std::vector<std::vector<std::shared_ptr<polyfhe::core::Node>>>
        l2reuse_nodes;

    for (auto node : graph->get_nodes()) {
        if (node->get_op_type() == polyfhe::core::OpType::MultConst) {
            for (auto outedge : node->get_out_edges()) {
                auto outnode = outedge->get_dst();
            }
        }
    }

    polyfhe::frontend::export_graph_to_dot(
        graph, "build/graph_extract_l2reuse_pass.dot");
    return true;
}
} // namespace engine
} // namespace polyfhe