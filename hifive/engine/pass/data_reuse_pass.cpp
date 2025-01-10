#include "hifive/engine/pass/data_reuse_pass.hpp"

#include <shared_mutex>

#include "hifive/core/config.hpp"
#include "hifive/core/logger.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

bool CanReuse(std::shared_ptr<hifive::core::Node> src,
              std::shared_ptr<hifive::core::Node> dst) {
    if (src->get_op_type() == core::OpType::Init) {
        return false;
    }
    if (dst->get_op_type() == core::OpType::Init) {
        return false;
    }
    switch (dst->get_access_pattern()) {
    case hifive::core::MemoryAccessPattern::ElementWise:
        return true;
    case hifive::core::MemoryAccessPattern::SlotWise:
        return true;
    case hifive::core::MemoryAccessPattern::LimbWise:
        return src->get_access_pattern() !=
               hifive::core::MemoryAccessPattern::LimbWise;
    case hifive::core::MemoryAccessPattern::NotDefined:
        return false;
    case hifive::core::MemoryAccessPattern::YetSet:
        LOG_ERROR("Yetset access pattern\n");
        return false;
    default:
        LOG_ERROR("Unknown access pattern\n");
        return false;
    }
}

std::shared_ptr<hifive::core::Node> GenSeed(
    std::set<std::shared_ptr<hifive::core::Node>>& unreused) {
    if (unreused.empty()) {
        return nullptr;
    }
    for (auto node : unreused) {
        if (node->get_access_pattern() ==
            core::MemoryAccessPattern::ElementWise) {
            unreused.erase(node);
            return node;
        }
    }
    for (auto node : unreused) {
        if (node->get_access_pattern() == core::MemoryAccessPattern::LimbWise) {
            unreused.erase(node);
            return node;
        }
    }
    for (auto node : unreused) {
        if (node->get_access_pattern() == core::MemoryAccessPattern::SlotWise) {
            unreused.erase(node);
            return node;
        }
    }
    return nullptr;
}

// TODO: merge with GetSubgraphType in extract_subgraph_pass.cpp
core::SubgraphType GetSubgraphType(
    std::vector<std::shared_ptr<hifive::core::Node>>& s) {
    if (s.size() == 0) {
        LOG_ERROR("Subgraph is empty\n");
        assert(false);
    }

    bool contains_elemwise = false;
    bool contains_limbwise = false;
    bool contains_ntt1 = false;
    bool contains_ntt2 = false;
    bool contains_slotwise = false;
    for (auto node : s) {
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

int GetSubgraphSmemFoorprint(
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph,
    core::SubgraphType s_type, std::shared_ptr<Config> config) {
    LOG_INFO("Calculate Footprint for: ");
    for (auto node : subgraph) {
        std::cout << node->get_op_name() << ", ";
    }
    std::cout << std::endl;

    // Get sPoly Size
    int spoly_size = core::GetsPolySize(s_type, config);

    // Analyze each outedge can be overwritten or not
    for (auto node : subgraph) {
        const int n_outedges = node->get_out_edges().size();
        for (int i = 0; i < n_outedges - 1; i++) {
            auto edge = node->get_out_edges()[n_outedges - i - 1];
            edge->set_can_overwrite(true);
        }
    }

    // Get number of sPoly
    int n_spoly = 1;
    int smem_size = 0;
    for (auto node : subgraph) {
        for (auto outedge : node->get_out_edges()) {
            // If dst is not in subgraph, skip
            if (std::find(subgraph.begin(), subgraph.end(),
                          outedge->get_dst()) == subgraph.end()) {
                continue;
            }
            if (outedge->get_level() == hifive::core::EdgeLevel::Shared) {
                if (!outedge->can_overwrite()) {
                    LOG_INFO("%s <-> %s cannot be overwritten\n",
                             outedge->get_src()->get_op_name().c_str(),
                             outedge->get_dst()->get_op_name().c_str());
                    n_spoly++;
                }
            }
        }
    }

    const int smem_footprint = spoly_size * n_spoly;
    LOG_INFO("-> sPoly %.2f KB * %d = %.2f KB\n", spoly_size / 1000.0, n_spoly,
             smem_footprint / 1000.0);
    return smem_footprint;
}

void ReuseWithSuccessor(
    std::shared_ptr<hifive::core::Graph> graph,
    std::shared_ptr<hifive::core::Node> seed,
    std::shared_ptr<hifive::core::Node> successor,
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph,
    std::set<std::shared_ptr<hifive::core::Node>>& unreused) {
    // Set Global level in default
    auto edge = core::get_edge(seed, successor);
    edge->set_level(hifive::core::EdgeLevel::Global);

    // Check if seed and successor can be reused
    if (!CanReuse(seed, successor)) {
        return;
    }

    // Add successor to subgraph temporarily
    subgraph.push_back(successor);

    // Get subgraph type
    core::SubgraphType s_type = GetSubgraphType(subgraph);

    // Get subgraph memory footprint
    // TODO: consider limb
    int footprint_kb =
        GetSubgraphSmemFoorprint(subgraph, s_type, graph->m_config) / 1000;

    // Check if footprint exceeds the limit
    if (footprint_kb > graph->m_config->SharedMemKB / 3) {
        subgraph.pop_back();
        return;
    }

    // Reuse!
    edge->set_level(hifive::core::EdgeLevel::Shared);
    unreused.erase(successor);

    for (auto edge : successor->get_out_edges()) {
        if (unreused.find(edge->get_dst()) == unreused.end()) {
            continue;
        }
        ReuseWithSuccessor(graph, successor, edge->get_dst(), subgraph,
                           unreused);
    }
}

void ReuseWithPredecessor(
    std::shared_ptr<hifive::core::Graph> graph,
    std::shared_ptr<hifive::core::Node> seed,
    std::shared_ptr<hifive::core::Node> pred,
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph,
    std::set<std::shared_ptr<hifive::core::Node>>& unreused) {
    // Set Global level in default
    auto edge = core::get_edge(pred, seed);
    edge->set_level(hifive::core::EdgeLevel::Global);

    // Check if seed and successor can be reused
    if (!CanReuse(pred, seed)) {
        return;
    }

    // Add successor to subgraph temporarily
    subgraph.push_back(pred);

    // Get subgraph type
    core::SubgraphType s_type = GetSubgraphType(subgraph);

    // Get subgraph memory footprint
    // TODO: consider limb
    int footprint_kb =
        GetSubgraphSmemFoorprint(subgraph, s_type, graph->m_config) / 1000;

    // Check if footprint exceeds the limit
    if (footprint_kb > graph->m_config->SharedMemKB / 3) {
        subgraph.pop_back();
        return;
    }

    // Reuse!
    edge->set_level(hifive::core::EdgeLevel::Shared);
    unreused.erase(pred);

    for (auto edge : pred->get_in_edges()) {
        if (unreused.find(edge->get_src()) == unreused.end()) {
            continue;
        }
        ReuseWithPredecessor(graph, pred, edge->get_src(), subgraph, unreused);
    }
}

bool DataReusePass::run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running DataReusePass\n");
    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_before.dot");

    // Put all node into unreused
    std::set<std::shared_ptr<hifive::core::Node>> unreused;
    for (auto node : graph->get_nodes()) {
        if (node == nullptr) {
            continue;
        }
        unreused.insert(node);
    }

    // Loop while unreused is not empty
    while (auto seed = GenSeed(unreused)) {
        LOG_INFO("Seed: %s\n", seed->get_op_name().c_str());
        // Subgraph to reuse data
        std::vector<std::shared_ptr<hifive::core::Node>> subgraph;

        // Add seed to subgraph
        subgraph.push_back(seed);

        // Check if append successors to subgraph
        for (auto edge : seed->get_out_edges()) {
            // if successor is not in unreused, skip
            if (unreused.find(edge->get_dst()) == unreused.end()) {
                continue;
            }
            ReuseWithSuccessor(graph, seed, edge->get_dst(), subgraph,
                               unreused);
        }

        for (auto edge : seed->get_in_edges()) {
            // if predecessor is not in unreused, skip
            if (unreused.find(edge->get_src()) == unreused.end()) {
                continue;
            }
            ReuseWithPredecessor(graph, seed, edge->get_src(), subgraph,
                                 unreused);
        }

        LOG_INFO("Extracted Subgraph: ");
        for (auto node : subgraph) {
            std::cout << node->get_op_name() << ", ";
        }
        std::cout << std::endl;

        // Remove subgraph from unreused
        for (auto node : subgraph) {
            unreused.erase(node);
        }
    }

    hifive::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_after.dot");
    return true;
}
} // namespace engine
} // namespace hifive