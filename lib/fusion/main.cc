#include <fstream>
#include <utility>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>

#include "graph.h"

int main()
{
    Graph g;

    define_hmult(g);

    boost::print_graph(g, get(&PolyOp::name, g));
    boost::dynamic_properties dp;
    dp.property("node_id", get(boost::vertex_index, g));
    dp.property("label", get(&PolyOp::name, g));
    dp.property("color", get(&PolyOp::color, g));

    std::ofstream file("graph.dot");
    boost::write_graphviz_dp(file, g, dp);
}
