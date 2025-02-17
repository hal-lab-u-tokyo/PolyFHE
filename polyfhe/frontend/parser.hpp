#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <memory>
#include <string>

#include "polyfhe/core/config.hpp"
#include "polyfhe/core/graph/graph.hpp"

namespace polyfhe {
namespace frontend {

struct DotNode {
    std::string name;
    std::string label;
    int peripheries;
};

struct DotEdge {
    std::string label;
    std::string color;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              DotNode, DotEdge>
    graph_t;

std::shared_ptr<polyfhe::core::Graph> ParseDotToGraph(
    const std::string& dot, polyfhe::core::GraphType graph_type,
    std::shared_ptr<polyfhe::Config> config);
} // namespace frontend
} // namespace polyfhe