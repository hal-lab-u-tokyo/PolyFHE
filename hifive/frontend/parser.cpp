#include "hifive/frontend/parser.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "hifive/core/logger.h"

namespace hifive {
namespace frontend {

struct DotVertex {
    std::string name;
    std::string label;
    int peripheries;
};

struct DotEdge {
    std::string label;
};

void ParseDotToGraph(const std::string& dot,
                     std::shared_ptr<hifive::core::Graph>& graph_hifive) {
    LOG_INFO("Parsing %s\n", dot.c_str());

    // Boost graph
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                  DotVertex, DotEdge>
        graph_t;
    graph_t g_dot(0);
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("node_id", boost::get(&DotVertex::name, g_dot));
    dp.property("label", boost::get(&DotVertex::label, g_dot));
    dp.property("peripheries", boost::get(&DotVertex::peripheries, g_dot));
    dp.property("label", boost::get(&DotEdge::label, g_dot));

    // Read dot file
    std::ifstream gvgraph(dot);
    if (!boost::read_graphviz(gvgraph, g_dot, dp)) {
        LOG_ERROR("Failed to parse graphviz dot file");
        exit(1);
    }
    LOG_INFO("Successfully read dot file\n");
}
} // namespace frontend
} // namespace hifive