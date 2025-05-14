#include "polyfhe/engine/pass/data_reuse_pass.hpp"

#include <shared_mutex>

#include "polyfhe/core/config.hpp"
#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

bool CanReuse(std::shared_ptr<polyfhe::core::Node> src,
              std::shared_ptr<polyfhe::core::Node> dst) {
    if (src->get_op_type() == core::OpType::Init ||
        src->get_op_type() == core::OpType::End) {
        return false;
    }
    if (dst->get_op_type() == core::OpType::Init ||
        dst->get_op_type() == core::OpType::End) {
        return false;
    }
    if (dst->get_block_phase() != src->get_block_phase()) {
        std::cout << "dst->get_block_phase() != src->get_block_phase()\n";
        std::cout << "src phase: " << src->get_block_phase() << "\n";
        std::cout << "dst phase: " << dst->get_block_phase() << "\n";
        return false;
    }
    // TODO: udpate
    if (src->get_op_type() == core::OpType::BConv ||
        dst->get_op_type() == core::OpType::BConv) {
        return false;
    }
    if (dst->get_op_type() == core::OpType::MultKeyAccum ||
        src->get_op_type() == core::OpType::MultKeyAccum) {
        return false;
    }

    if (src->get_access_pattern() == core::MemoryAccessPattern::ElementWise ||
        dst->get_access_pattern() == core::MemoryAccessPattern::ElementWise) {
        return true;
    } else if (src->get_access_pattern() != dst->get_access_pattern()) {
        return false;
    }
    return true;
}

std::shared_ptr<polyfhe::core::Node> GenSeed(
    std::set<std::shared_ptr<polyfhe::core::Node>>& unreused) {
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
void WithdrawReuse(std::shared_ptr<polyfhe::core::Node> seed) {
    for (auto edge : seed->get_out_edges()) {
        edge->set_level(polyfhe::core::EdgeLevel::Global);
    }
    for (auto edge : seed->get_in_edges()) {
        edge->set_level(polyfhe::core::EdgeLevel::Global);
    }
}

void ReuseWithSuccessor(
    std::shared_ptr<polyfhe::core::Graph> graph,
    std::shared_ptr<polyfhe::core::Node> seed,
    std::vector<std::shared_ptr<polyfhe::core::Node>>& subgraph) {
    // Check if seed and successor can be reused
    for (auto edge : seed->get_out_edges()) {
        std::cout << "Reuse?" << edge->get_src()->get_op_name() << " -> "
                  << edge->get_dst()->get_op_name() << "\n";
        if (!CanReuse(seed, edge->get_dst())) {
            std::cout << " no" << std::endl;
            LOG_INFO("Cannot reuse %s -> %s\n", seed->get_op_name().c_str(),
                     edge->get_dst()->get_op_name().c_str());
            edge->set_level(polyfhe::core::EdgeLevel::Global);
        }
    }

    // Extract subgraph
    std::vector<std::shared_ptr<polyfhe::core::Node>> new_subgraph;
    ExtractSubgraph(seed, new_subgraph);

    // Get subgraph memory footprint
    // TODO: consider limb
    int footprint_kb =
        GetSubgraphSmemFoorprint(new_subgraph, graph->m_config) / 1000;

    // Check if footprint exceeds the limit
    if (footprint_kb > graph->m_config->SharedMemKB) {
        LOG_INFO("Footprint %d KB exceeds the limit %d KB\n", footprint_kb,
                 graph->m_config->SharedMemKB);
        for (auto edge : seed->get_out_edges()) {
            edge->set_level(polyfhe::core::EdgeLevel::Global);
        }
        return;
    }

    // Reuse!
    subgraph = new_subgraph;
}

void ReuseWithPredecessor(
    std::shared_ptr<polyfhe::core::Graph> graph,
    std::shared_ptr<polyfhe::core::Node> seed,
    std::vector<std::shared_ptr<polyfhe::core::Node>>& subgraph) {
    // Check if seed and successor can be reused
    for (auto edge : seed->get_in_edges()) {
        if (!CanReuse(seed, edge->get_src())) {
            LOG_INFO("Cannot reuse with predecessor %s <-> %s\n",
                     seed->get_op_name().c_str(),
                     edge->get_src()->get_op_name().c_str());
            edge->set_level(polyfhe::core::EdgeLevel::Global);
        }
    }

    // Extract subgraph
    std::vector<std::shared_ptr<polyfhe::core::Node>> new_subgraph;
    ExtractSubgraph(seed, new_subgraph);

    // Get subgraph memory footprint
    // TODO: consider limb
    int footprint_kb =
        GetSubgraphSmemFoorprint(new_subgraph, graph->m_config) / 1000;

    // Check if footprint exceeds the limit
    if (footprint_kb > graph->m_config->SharedMemKB) {
        LOG_INFO("Footprint %d KB exceeds the limit %d KB\n", footprint_kb,
                 graph->m_config->SharedMemKB);
        for (auto edge : seed->get_in_edges()) {
            edge->set_level(polyfhe::core::EdgeLevel::Global);
        }
        return;
    }

    // Reuse!
    subgraph = new_subgraph;
}

bool DataReusePass::run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running DataReusePass\n");
    polyfhe::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_before.dot");

    // Put all node into unreused
    std::set<std::shared_ptr<polyfhe::core::Node>> unreused;
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
        std::vector<std::shared_ptr<polyfhe::core::Node>> subgraph;

        // Add seed to subgraph
        subgraph.push_back(seed);

        // Check if append successors to subgraph
        for (auto edge : seed->get_out_edges()) {
            edge->set_level(polyfhe::core::EdgeLevel::Shared);
        }
        ReuseWithSuccessor(graph, seed, subgraph);

        for (auto edge : seed->get_in_edges()) {
            edge->set_level(polyfhe::core::EdgeLevel::Shared);
        }
        ReuseWithPredecessor(graph, seed, subgraph);

        // Remove subgraph from unreused
        for (auto node : subgraph) {
            unreused.erase(node);
        }
    }

    polyfhe::frontend::export_graph_to_dot(
        graph, "build/graph_data_reuse_pass_after.dot");
    return true;
}
} // namespace engine
} // namespace polyfhe