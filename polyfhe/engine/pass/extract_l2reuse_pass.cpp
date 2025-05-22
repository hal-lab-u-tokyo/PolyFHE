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
        if (node == nullptr) {
            continue;
        }
        if (node->get_op_type() == polyfhe::core::OpType::MultConst) {
            for (auto outedge : node->get_out_edges()) {
                std::shared_ptr<polyfhe::core::Node> nextnode =
                    outedge->get_dst();
                std::vector<std::shared_ptr<polyfhe::core::Node>> reuse_nodes;
                while (nextnode->get_op_type() !=
                       polyfhe::core::OpType::MultKeyAccum) {
                    reuse_nodes.push_back(nextnode);
                    assert(nextnode->get_out_edges().size() == 1);
                    nextnode = nextnode->get_out_edges()[0]->get_dst();
                }
                l2reuse_nodes.push_back(reuse_nodes);
            }
        }
    }

    std::cout << "L2 reuse nodes: " << std::endl;
    for (auto reuse_nodes : l2reuse_nodes) {
        std::cout << "  ";
        for (auto node : reuse_nodes) {
            std::cout << node->get_op_name() << " ";
        }
        std::cout << std::endl;
    }

    polyfhe::core::SubGraph subgraph;
    assert(l2reuse_nodes.size() > 0);
    for (int i = 0; i < l2reuse_nodes[0].size(); i++) {
        for (int j = 0; j < l2reuse_nodes.size(); j++) {
            auto node = l2reuse_nodes[j][i];
            subgraph.add_node(node);
        }
    }
    subgraph.set_subgraph_type(polyfhe::core::SubgraphType::L2);

    for (auto subgraph : graph->get_subgraphs()) {
        std::cout << "subgraph: " << subgraph->get_name() << std::endl;
        if (subgraph->get_nodes().size() == 1) {
        }
    }

    graph->add_subgraph(std::make_shared<polyfhe::core::SubGraph>(subgraph));

    // remove reuse node from subgraph
    std::vector<std::shared_ptr<core::SubGraph>> erase_target;
    for (auto reuse_nodes : l2reuse_nodes) {
        for (auto node : reuse_nodes) {
            bool found = false;
            for (auto sgraph : graph->get_subgraphs()) {
                if (sgraph->get_nodes().size() == 1) {
                    if (sgraph->get_nodes()[0] == node) {
                        erase_target.push_back(sgraph);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                LOG_ERROR("Cannot find subgraph for reuse node %s\n",
                          node->get_op_name().c_str());
            }
        }
    }
    for (auto sgraph : erase_target) {
        graph->get_subgraphs().erase(
            std::remove(graph->get_subgraphs().begin(),
                        graph->get_subgraphs().end(), sgraph),
            graph->get_subgraphs().end());
    }

    // polyfhe::frontend::export_graph_to_dot(
    //     new_graph, "build/graph_extract_l2reuse_pass.dot");
    return true;
}
} // namespace engine
} // namespace polyfhe