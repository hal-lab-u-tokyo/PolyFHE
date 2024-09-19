#include <fstream>
#include <utility>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>

#include "graph.h"

int main()
{
    // Define FHE graph
    GraphFHE g_fhe;
    dummy_fhe_graph(g_fhe);
    boost::dynamic_properties dp_fhe;
    dp_fhe.property("node_id", get(boost::vertex_index, g_fhe));
    dp_fhe.property("label", get(&FHEOp::name, g_fhe));
    std::ofstream file_fhe("./data/graph_fhe.dot");
    boost::write_graphviz_dp(file_fhe, g_fhe, dp_fhe);
    
    // Lower to Poly graph
    GraphPoly g_poly;
    lower_fhe_to_poly(g_fhe, g_poly);
    //boost::print_graph(g_poly, get(&PolyOp::name, g_poly));
    boost::dynamic_properties dp_poly;
    dp_poly.property("node_id", get(boost::vertex_index, g_poly));
    dp_poly.property("label", get(&PolyOp::name, g_poly));
    dp_poly.property("color", get(&PolyOp::color, g_poly));
    std::ofstream file("./data/graph_poly.dot");
    boost::write_graphviz_dp(file, g_poly, dp_poly);

    // Fuse Poly graph
    GraphPoly g_poly_fused;
    fuse_poly(g_poly, g_poly_fused);
}
