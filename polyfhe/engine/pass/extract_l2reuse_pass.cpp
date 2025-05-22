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
    std::shared_ptr<polyfhe::core::Node> accum_node;

    for (auto node : graph->get_nodes()) {
        if (node == nullptr) {
            continue;
        }
        if (node->get_op_type() == polyfhe::core::OpType::MultConst) {
            for (auto outedge : node->get_out_edges()) {
                std::shared_ptr<polyfhe::core::Node> nextnode =
                    outedge->get_dst();
                std::vector<std::shared_ptr<polyfhe::core::Node>> reuse_nodes;
                do {
                    reuse_nodes.push_back(nextnode);
                    assert(nextnode->get_out_edges().size() == 1);
                    nextnode = nextnode->get_out_edges()[0]->get_dst();
                } while (nextnode->get_op_type() !=
                         polyfhe::core::OpType::MultKeyAccum);
                assert(nextnode->get_op_type() ==
                       polyfhe::core::OpType::MultKeyAccum);
                accum_node = nextnode;
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

    polyfhe::core::SubGraph new_subgraph;

    assert(l2reuse_nodes.size() > 0);
    int n_beta = l2reuse_nodes.size();
    int n_ops = l2reuse_nodes[0].size();
    std::vector<std::shared_ptr<polyfhe::core::Node>> sorted_nodes(
        n_beta * n_ops, nullptr);
    std::cout << "n_beta: " << n_beta << ", n_ops: " << n_ops << std::endl;
    for (int i = 0; i < n_beta; i++) {
        auto nodes_i = l2reuse_nodes[i];
        assert(nodes_i[0]->get_op_type() == polyfhe::core::OpType::BConv);
        int beta_idx = nodes_i[0]->get_beta_idx();
        for (int j = 0; j < n_ops; j++) {
            sorted_nodes[beta_idx * n_ops + j] = nodes_i[j];
        }
    }
    for (auto node : sorted_nodes) {
        std::cout << node->get_op_name() << std::endl;
        new_subgraph.add_node(node);
    }
    new_subgraph.add_node(accum_node);
    new_subgraph.set_subgraph_type(polyfhe::core::SubgraphType::L2);
    new_subgraph.set_beta(n_beta);

    graph->add_subgraph(
        std::make_shared<polyfhe::core::SubGraph>(new_subgraph));

    // remove reuse node from subgraph
    std::vector<std::shared_ptr<core::SubGraph>> erase_target;
    for (auto node : new_subgraph.get_nodes()) {
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