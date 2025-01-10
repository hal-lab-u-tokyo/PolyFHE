#include "hifive/engine/pass/lowering_ckks_to_poly_pass.hpp"

#include "lowering_ckks_to_poly_pass.hpp"

namespace hifive {
namespace engine {

std::pair<std::vector<std::shared_ptr<core::Node>>,
          std::vector<std::shared_ptr<core::Node>>>
lower_fhe_node_to_poly(std::shared_ptr<hifive::core::Graph>& graph_poly,
                       std::shared_ptr<hifive::core::Node>& node) {
    std::vector<std::shared_ptr<core::Node>> tops;
    std::vector<std::shared_ptr<core::Node>> bottoms;

    // TODO: parameters
    const std::string edge_label = "20";
    const int d = 3;

    if (node->get_op_type() == core::OpType::Init) {
        auto init = std::make_shared<hifive::core::Node>(core::OpType::Init);
        graph_poly->add_node(init);
        graph_poly->set_init_node(init);
        bottoms.push_back(init);
    } else if (node->get_op_type() == core::OpType::End) {
        auto end = std::make_shared<hifive::core::Node>(core::OpType::End);
        graph_poly->add_node(end);
        graph_poly->set_exit_node(end);
        tops.push_back(end);
    } else if (node->get_op_type() == core::OpType::HAdd) {
        auto add_ax = std::make_shared<hifive::core::Node>(core::OpType::Add);
        auto add_bx = std::make_shared<hifive::core::Node>(core::OpType::Add);
        graph_poly->add_node(add_ax);
        graph_poly->add_node(add_bx);
        tops.push_back(add_ax);
        tops.push_back(add_bx);
        bottoms.push_back(add_ax);
        bottoms.push_back(add_bx);
    } else if (node->get_op_type() == core::OpType::HMult) {
        std::vector<std::shared_ptr<hifive::core::Node>> list_accum_ax;
        std::vector<std::shared_ptr<hifive::core::Node>> list_accum_bx;

        auto mult_axax =
            std::make_shared<hifive::core::Node>(core::OpType::Mult);
        auto mult_bxbx =
            std::make_shared<hifive::core::Node>(core::OpType::Mult);
        auto mult_axbx =
            std::make_shared<hifive::core::Node>(core::OpType::Mult);
        auto mult_bxax =
            std::make_shared<hifive::core::Node>(core::OpType::Mult);
        auto add_axbx = std::make_shared<hifive::core::Node>(core::OpType::Add);
        auto intt_axax =
            std::make_shared<hifive::core::Node>(core::OpType::iNTT);
        auto mult_decomp =
            std::make_shared<hifive::core::Node>(core::OpType::Mult);
        graph_poly->add_node(mult_axax);
        graph_poly->add_node(mult_bxbx);
        graph_poly->add_node(mult_axbx);
        graph_poly->add_node(mult_bxax);
        graph_poly->add_node(add_axbx);
        graph_poly->add_node(intt_axax);
        graph_poly->add_node(mult_decomp);
        graph_poly->add_edge(mult_axbx, add_axbx, edge_label);
        graph_poly->add_edge(mult_bxax, add_axbx, edge_label);
        graph_poly->add_edge(mult_axax, intt_axax, edge_label);
        graph_poly->add_edge(intt_axax, mult_decomp, edge_label);

        std::shared_ptr<hifive::core::Node> accum = nullptr;
        for (int i = 0; i < d; i++) {
            auto modup =
                std::make_shared<hifive::core::Node>(core::OpType::ModUp);
            auto ntt = std::make_shared<hifive::core::Node>(core::OpType::NTT);
            auto multkey_ax =
                std::make_shared<hifive::core::Node>(core::OpType::Mult);
            auto multkey_bx =
                std::make_shared<hifive::core::Node>(core::OpType::Mult);
            graph_poly->add_node(modup);
            graph_poly->add_node(ntt);
            graph_poly->add_node(multkey_ax);
            graph_poly->add_node(multkey_bx);
            graph_poly->add_edge(mult_decomp, modup, edge_label);
            graph_poly->add_edge(modup, ntt, edge_label);
            graph_poly->add_edge(ntt, multkey_ax, edge_label);
            graph_poly->add_edge(ntt, multkey_bx, edge_label);

            if (i == 0) {
                list_accum_ax.push_back(multkey_ax);
                list_accum_bx.push_back(multkey_bx);
            } else {
                auto accum_ax =
                    std::make_shared<hifive::core::Node>(core::OpType::Add);
                auto accum_bx =
                    std::make_shared<hifive::core::Node>(core::OpType::Add);
                list_accum_ax.push_back(accum_ax);
                list_accum_bx.push_back(accum_bx);
                graph_poly->add_node(accum_ax);
                graph_poly->add_node(accum_bx);
                graph_poly->add_edge(list_accum_ax[i - 1], accum_ax,
                                     edge_label);
                graph_poly->add_edge(list_accum_bx[i - 1], accum_bx,
                                     edge_label);
                graph_poly->add_edge(multkey_ax, accum_ax, edge_label);
                graph_poly->add_edge(multkey_bx, accum_bx, edge_label);
            }
        }
        auto intt_ax = std::make_shared<hifive::core::Node>(core::OpType::iNTT);
        auto intt_bx = std::make_shared<hifive::core::Node>(core::OpType::iNTT);
        auto moddown_ax =
            std::make_shared<hifive::core::Node>(core::OpType::ModDown);
        auto moddown_bx =
            std::make_shared<hifive::core::Node>(core::OpType::ModDown);
        auto ntt_ax = std::make_shared<hifive::core::Node>(core::OpType::NTT);
        auto ntt_bx = std::make_shared<hifive::core::Node>(core::OpType::NTT);
        auto add_final_ax =
            std::make_shared<hifive::core::Node>(core::OpType::Add);
        auto add_final_bx =
            std::make_shared<hifive::core::Node>(core::OpType::Add);
        graph_poly->add_node(intt_ax);
        graph_poly->add_node(intt_bx);
        graph_poly->add_node(moddown_ax);
        graph_poly->add_node(moddown_bx);
        graph_poly->add_node(ntt_ax);
        graph_poly->add_node(ntt_bx);
        graph_poly->add_node(add_final_ax);
        graph_poly->add_node(add_final_bx);
        graph_poly->add_edge(list_accum_ax[list_accum_ax.size() - 1], intt_ax,
                             edge_label);
        graph_poly->add_edge(list_accum_bx[list_accum_bx.size() - 1], intt_bx,
                             edge_label);
        graph_poly->add_edge(intt_ax, moddown_ax, edge_label);
        graph_poly->add_edge(intt_bx, moddown_bx, edge_label);
        graph_poly->add_edge(moddown_ax, ntt_ax, edge_label);
        graph_poly->add_edge(moddown_bx, ntt_bx, edge_label);
        graph_poly->add_edge(ntt_ax, add_final_ax, edge_label);
        graph_poly->add_edge(ntt_bx, add_final_bx, edge_label);
        graph_poly->add_edge(add_axbx, add_final_ax, edge_label);
        graph_poly->add_edge(mult_bxbx, add_final_bx, edge_label);

        tops.push_back(mult_axax);
        tops.push_back(mult_bxbx);
        tops.push_back(mult_axbx);
        tops.push_back(mult_bxax);
        bottoms.push_back(add_final_ax);
        bottoms.push_back(add_final_bx);
    }
    return std::make_pair(tops, bottoms);
}

bool LoweringCKKSToPolyPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running LoweringCKKSToPolyPass\n");

    std::shared_ptr<hifive::core::Graph> graph_poly =
        std::make_shared<hifive::core::Graph>(graph->m_config);

    if (graph->get_graph_type() != hifive::core::GraphType::FHE) {
        LOG_ERROR("Graph type is not FHE\n");
        return false;
    }

    // Lowering nodes
    // DFS
    const int n = graph->get_nodes_size();
    std::vector<bool> visited(n, false);
    std::vector<int> stack;
    stack.push_back(graph->get_init_node_id());
    while (!stack.empty()) {
        const int node_idx = stack.back();
        stack.pop_back();

        if (visited[node_idx]) {
            continue;
        }

        visited[node_idx] = true;
        auto node = graph->get_nodes()[node_idx];
        if (node == nullptr) {
            // node shouldn't be nullptr at this point
            LOG_ERROR("Node is nullptr\n");
            return false;
        }

        // Lowering
        auto [tops, bottoms] = lower_fhe_node_to_poly(graph_poly, node);
        for (auto top : tops) {
            node->add_top_poly_op(top);
        }
        for (auto bottom : bottoms) {
            node->add_bottom_poly_op(bottom);
        }

        // Update
        for (auto edge : node->get_out_edges()) {
            auto dst = edge->get_dst();
            stack.push_back(dst->get_id());
        }
    }

    // Lowering edges
    const std::string edge_label = "32768_20";
    for (auto node : graph->get_nodes()) {
        for (auto edge : node->get_out_edges()) {
            auto src = edge->get_src();
            auto dst = edge->get_dst();

            std::vector<std::shared_ptr<hifive::core::Node>> src_outputs =
                src->get_bottom_poly_ops();
            std::vector<std::shared_ptr<hifive::core::Node>> dst_inputs =
                dst->get_top_poly_ops();

            if ((src->get_op_type() == core::OpType::Init) ||
                (dst->get_op_type() == core::OpType::End)) {
                if (src->get_op_type() == core::OpType::Init) {
                    if (src_outputs.size() != 1) {
                        LOG_ERROR("Init node should have only one output\n");
                        return false;
                    }
                    for (auto d : dst_inputs) {
                        graph_poly->add_edge(src_outputs[0], d, edge_label);
                    }
                }
                if (dst->get_op_type() == core::OpType::End) {
                    if (dst_inputs.size() != 1) {
                        LOG_ERROR("End node should have only one input\n");
                        return false;
                    }
                    for (auto s : src_outputs) {
                        graph_poly->add_edge(s, dst_inputs[0], edge_label);
                    }
                }
            } else {
                if (src_outputs.size() != 2) {
                    LOG_ERROR("Source node should have two outputs\n");
                    return false;
                }
                if (dst_inputs.size() == 2) {
                    graph_poly->add_edge(src_outputs[0], dst_inputs[0],
                                         edge_label);
                    graph_poly->add_edge(src_outputs[1], dst_inputs[1],
                                         edge_label);
                } else if (dst_inputs.size() == 4) {
                    if (dst_inputs[0]->get_in_edges().size() == 0) {
                        graph_poly->add_edge(src_outputs[0], dst_inputs[0],
                                             edge_label);
                        graph_poly->add_edge(src_outputs[1], dst_inputs[1],
                                             edge_label);
                    } else {
                        if (dst_inputs[2]->get_in_edges().size() != 0) {
                            LOG_ERROR(
                                "Destination node should have one or two "
                                "inputs\n");
                            return false;
                        }
                        graph_poly->add_edge(src_outputs[0], dst_inputs[2],
                                             edge_label);
                        graph_poly->add_edge(src_outputs[1], dst_inputs[3],
                                             edge_label);
                    }
                } else {
                    LOG_ERROR("Unexpected dst node input size\n");
                    return false;
                }
            }
        }
    }
    graph = graph_poly;
    return true;
}
} // namespace engine
} // namespace hifive