#include "extract_subgraph_pass.hpp"

#include <optional>

#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

void SortSubgraphNodes(
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph) {
    // Check number of inedges which level is Shared
    std::vector<std::shared_ptr<hifive::core::Node>> sorted_subgraph;
    std::map<std::shared_ptr<hifive::core::Node>, bool> sorted;

    // Sort nodes which have no dependent inedges
    for (size_t i = 0; i < subgraph.size(); i++) {
        bool has_dependent_inedges = false;
        for (auto inedge : subgraph[i]->get_in_edges()) {
            if (inedge->get_level() == hifive::core::EdgeLevel::Shared) {
                has_dependent_inedges = true;
            }
        }
        if (!has_dependent_inedges) {
            sorted_subgraph.push_back(subgraph[i]);
            sorted[subgraph[i]] = true;
        }
    }

    // Sort other nodes
    while (sorted_subgraph.size() < subgraph.size()) {
        for (size_t i = 0; i < subgraph.size(); i++) {
            if (sorted[subgraph[i]]) {
                continue;
            }
            auto node = subgraph[i];
            bool all_inedges_sorted = true;
            for (auto inedge : node->get_in_edges()) {
                if (inedge->get_level() != hifive::core::EdgeLevel::Shared) {
                    continue;
                }
                if (inedge->get_src()->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (!sorted[inedge->get_src()]) {
                    LOG_INFO("Wait for %s\n",
                             inedge->get_src()->get_op_name().c_str());
                    all_inedges_sorted = false;
                    break;
                }
            }
            if (all_inedges_sorted) {
                sorted_subgraph.push_back(node);
                sorted[node] = true;
            } else {
                LOG_INFO("Cannot Sort %s\n", node->get_op_name().c_str());
            }
        }
    }
    subgraph.erase(subgraph.begin(), subgraph.end());
    for (auto node : sorted_subgraph) {
        subgraph.push_back(node);
    }
}

void ExtractSubgraph(
    std::shared_ptr<hifive::core::Node> node,
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph) {
    if (node->get_op_type() == core::OpType::Init ||
        node->get_op_type() == core::OpType::End) {
        return;
    }
    subgraph.push_back(node);
    for (auto edge : node->get_out_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_dst());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            ExtractSubgraph(edge->get_dst(), subgraph);
        }
    }
    for (auto edge : node->get_in_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_src());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == hifive::core::EdgeLevel::Shared) {
            ExtractSubgraph(edge->get_src(), subgraph);
        }
    }
}

std::optional<std::vector<std::shared_ptr<hifive::core::Node>>>
GetSortedSubgraph(std::shared_ptr<hifive::core::Node> node,
                  std::vector<bool>& visited) {
    std::vector<std::shared_ptr<hifive::core::Node>> subgraph;
    ExtractSubgraph(node, subgraph);
    if (subgraph.size() == 0) {
        return std::nullopt;
    }
    SortSubgraphNodes(subgraph);
    for (auto subnode : subgraph) {
        if (!visited[subnode->get_id()]) {
            return std::nullopt;
        }
    }
    return std::make_optional(subgraph);
}

void SetSharedMemOffset(hifive::core::SubGraph& subgraph,
                        std::shared_ptr<hifive::Config>& config) {
    int sPoly_size = core::GetsPolySize(subgraph.get_nodes(), config);
    int n_spoly = 0;
    for (auto node : subgraph.get_nodes()) {
        bool can_overwrite = false;
        bool all_global = true;
        for (auto inedge : node->get_in_edges()) {
            if (inedge->get_level() == hifive::core::EdgeLevel::Shared) {
                if (inedge->can_overwrite()) {
                    can_overwrite = true;
                    for (auto outedge : node->get_out_edges()) {
                        // Set offset_smem if dst node is in subgraph
                        auto found = std::find(subgraph.get_nodes().begin(),
                                               subgraph.get_nodes().end(),
                                               outedge->get_dst());
                        if (found != subgraph.get_nodes().end()) {
                            outedge->set_offset_smem(inedge->get_offset_smem());
                        }
                    }
                }
            }
        }

        for (auto outedge : node->get_out_edges()) {
            if (outedge->get_level() != hifive::core::EdgeLevel::Global) {
                all_global = false;
            }
        }

        if (can_overwrite | all_global) {
            continue;
        } else {
            for (auto outedge : node->get_out_edges()) {
                // Set offset_smem if dst node is in subgraph
                auto found =
                    std::find(subgraph.get_nodes().begin(),
                              subgraph.get_nodes().end(), outedge->get_dst());
                if (found != subgraph.get_nodes().end()) {
                    outedge->set_offset_smem(sPoly_size * n_spoly /
                                             sizeof(uint64_t));
                }
            }
            n_spoly++;
        }
        for (auto outedge : node->get_out_edges()) {
            LOG_INFO("Offset of %s <-> %s: %d\n", node->get_op_name().c_str(),
                     outedge->get_dst()->get_op_name().c_str(),
                     outedge->get_offset_smem());
        }
    }
}

bool ExtractSubgraphPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running ExtractSubgraphPass\n");

    int idx_subgraph = 0;
    const int n = graph->get_nodes().size();
    std::vector<int> indegree(n, 0);
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    for (int i = 0; i < n; i++) {
        auto node = graph->get_nodes()[i];
        if (node == nullptr) {
            // Some node can be nullptr, i.e., NTT node is replaced with
            // NTTPhase1 and NTTPhase2 nodes
            continue;
        }
        indegree[i] = node->get_in_edges().size();
    }

    // Init node
    // Visit init node first because we don't include it in subgraph
    auto init_node = graph->get_init_node();
    visited[init_node->get_id()] = true;
    for (auto edge : init_node->get_out_edges()) {
        int dst_id = edge->get_dst()->get_id();
        indegree[dst_id] -= 1;
        if (indegree[dst_id] == 0) {
            stack.push_back(dst_id);
        }
    }

    while (!stack.empty()) {
        int node_idx = stack.back();
        stack.pop_back();
        if (visited[node_idx]) {
            continue;
        }
        visited[node_idx] = true;
        auto node = graph->get_nodes()[node_idx];
        if (node == nullptr) {
            continue;
        }

        // Check if all of subgraph nodes are visited
        std::optional<std::vector<std::shared_ptr<hifive::core::Node>>>
            has_visited_subnodes = GetSortedSubgraph(node, visited);
        if (has_visited_subnodes) {
            hifive::core::SubGraph subgraph;
            for (auto subnode : *has_visited_subnodes) {
                subnode->set_idx_subgraph(idx_subgraph);
                subgraph.add_node(subnode);
            }
            // Set block_phase
            subgraph.set_block_phase(node->get_block_phase());

            // Set Subgraph Type
            subgraph.set_subgraph_type(GetSubgraphType(subgraph.get_nodes()));
            subgraph.set_smem_size(GetSubgraphSmemFoorprint(
                subgraph.get_nodes(), graph->m_config));

            // Set SharedMemOffset
            SetSharedMemOffset(subgraph, graph->m_config);

            // TODO: search for optimal ny_batch
            subgraph.set_nx_batch(1);
            subgraph.set_ny_batch(1);
            graph->add_subgraph(
                std::make_shared<hifive::core::SubGraph>(subgraph));
            idx_subgraph += 1;
        }

        for (auto edge : node->get_out_edges()) {
            int dst_id = edge->get_dst()->get_id();
            indegree[dst_id] -= 1;
            if (indegree[dst_id] == 0) {
                stack.push_back(dst_id);
            }
        }
    }

    for (auto subgraph : graph->get_subgraphs()) {
        LOG_INFO("Subgraph[%d]:\n", subgraph->get_idx());
        for (auto node : subgraph->get_nodes()) {
            LOG_INFO("  %s\n", node->get_op_name().c_str());
        }
    }
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_extract_subgraph_pass.dot");
    return true;
}
} // namespace engine
} // namespace hifive