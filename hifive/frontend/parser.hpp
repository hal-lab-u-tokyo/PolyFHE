#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <memory>
#include <string>

#include "hifive/core/graph/graph.hpp"

namespace hifive {
namespace frontend {

struct DotNode {
    std::string name;
    std::string label;
    int peripheries;
};

struct DotEdge {
    std::string label;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              DotNode, DotEdge>
    graph_t;

std::shared_ptr<hifive::core::Graph> ParseDotToGraph(
    const std::string& dot, hifive::core::GraphType graph_type);
} // namespace frontend
} // namespace hifive