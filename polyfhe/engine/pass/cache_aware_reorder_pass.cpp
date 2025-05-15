
#include "polyfhe/engine/pass/cache_aware_reorder_pass.hpp"

#include <shared_mutex>

#include "polyfhe/core/config.hpp"
#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

bool CacheAwareReorderpass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CacheAwareReorderpass\n");

    /*
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
    */

    return true;
}
} // namespace engine
} // namespace polyfhe