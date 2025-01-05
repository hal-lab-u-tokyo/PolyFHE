#include "extract_subgraph_pass.hpp"

#include <optional>

#include "hifive/core/logger.hpp"
#include "hifive/core/param.hpp"
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
CheckIfSubgraphNodesVisited(std::shared_ptr<hifive::core::Node> node,
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

core::SubgraphType GetSubgraphType(core::SubGraph& s) {
    if (s.get_nodes().size() == 0) {
        LOG_ERROR("Subgraph is empty\n");
        assert(false);
    }

    bool contains_elemwise = false;
    bool contains_limbwise = false;
    bool contains_ntt1 = false;
    bool contains_ntt2 = false;
    bool contains_slotwise = false;
    for (auto node : s.get_nodes()) {
        core::MemoryAccessPattern pattern = node->get_access_pattern();
        if (pattern == core::MemoryAccessPattern::ElementWise) {
            contains_elemwise = true;
        } else if (pattern == core::MemoryAccessPattern::LimbWise) {
            contains_limbwise = true;
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::NTTPhase1 ||
                op_type == core::OpType::iNTTPhase1) {
                contains_ntt1 = true;
            } else if (op_type == core::OpType::NTTPhase2 ||
                       op_type == core::OpType::iNTTPhase2) {
                contains_ntt2 = true;
            }
        } else if (pattern == core::MemoryAccessPattern::SlotWise) {
            contains_slotwise = true;
        }
    }

    // Check if Phase1 or Phase2 in advance
    bool if_phase1 = false;
    if (contains_limbwise) {
        if (contains_ntt1) {
            if (contains_ntt2) {
                LOG_ERROR(
                    "Unknown SubgraphType: Both NTTPhase1 and 2 is "
                    "contained to one subgraph.\n");
                assert(false);
            }
            if_phase1 = true;
        } else if (contains_ntt2) {
            if_phase1 = false;
        } else {
            LOG_ERROR(
                "Unknown SubgraphType: None of NTTPhase1 and 2 is "
                "contained to one subgraph.\n");
        }
    }

    //
    if (contains_slotwise) {
        if (contains_limbwise) {
            return if_phase1 ? core::SubgraphType::ElemLimb1Slot
                             : core::SubgraphType::ElemLimb2Slot;
        } else {
            return core::SubgraphType::ElemSlot;
        }
    } else if (contains_limbwise) {
        return if_phase1 ? core::SubgraphType::ElemLimb1
                         : core::SubgraphType::ElemLimb2;
    } else if (contains_elemwise) {
        return core::SubgraphType::Elem;
    } else {
        LOG_ERROR("Unknown SubgraphType\n");
        assert(false);
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
            has_visited_subnodes = CheckIfSubgraphNodesVisited(node, visited);
        if (has_visited_subnodes) {
            hifive::core::SubGraph subgraph;
            for (auto subnode : *has_visited_subnodes) {
                subnode->set_idx_subgraph(idx_subgraph);
                subgraph.add_node(subnode);
            }
            // Set block_phase
            subgraph.set_block_phase(node->get_block_phase());

            // Set Subgraph Type
            subgraph.set_subgraph_type(GetSubgraphType(subgraph));

            // TODO: search for optimal ny_batch
            subgraph.set_nx_batch(1);
            subgraph.set_ny_batch(L);
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