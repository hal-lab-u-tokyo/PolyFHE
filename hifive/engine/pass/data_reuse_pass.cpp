#include "hifive/engine/pass/data_reuse_pass.hpp"

#include <shared_mutex>

#include "hifive/core/config.hpp"
#include "hifive/core/graph/graph.hpp"
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
void WithdrawReuse(std::shared_ptr<hifive::core::Node> seed) {
    for (auto edge : seed->get_out_edges()) {
        edge->set_level(hifive::core::EdgeLevel::Global);
    }
    for (auto edge : seed->get_in_edges()) {
        edge->set_level(hifive::core::EdgeLevel::Global);
    }
}

void ReuseWithSuccessor(
    std::shared_ptr<hifive::core::Graph> graph,
    std::shared_ptr<hifive::core::Node> seed,
    std::vector<std::shared_ptr<hifive::core::Node>>& subgraph) {
    // Check if seed and successor can be reused
    for (auto edge : seed->get_out_edges()) {
        if (!CanReuse(seed, edge->get_dst())) {
            WithdrawReuse(seed);
            return;
        }
    }

    // Extract subgraph
    std::vector<std::shared_ptr<hifive::core::Node>> new_subgraph;
    ExtractSubgraph(seed, new_subgraph);

    // Get subgraph memory footprint
    // TODO: consider limb
    int footprint_kb =
        GetSubgraphSmemFoorprint(new_subgraph, graph->m_config) / 1000;

    // Check if footprint exceeds the limit
    if (footprint_kb > graph->m_config->SharedMemKB) {
        WithdrawReuse(seed);
        return;
    }

    // Reuse!
    for (auto edge : seed->get_out_edges()) {
        edge->set_level(hifive::core::EdgeLevel::Shared);
        subgraph.push_back(edge->get_dst());
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
            edge->set_level(hifive::core::EdgeLevel::Shared);
        }
        ReuseWithSuccessor(graph, seed, subgraph);

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