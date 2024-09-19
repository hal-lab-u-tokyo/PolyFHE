#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <string>
#include <utility>

#include "graph.h"

void export_poly_graph(GraphPoly &g_poly, std::string filename) {
    boost::dynamic_properties dp_poly;
    dp_poly.property("node_id", get(boost::vertex_index, g_poly));
    dp_poly.property("label", get(&PolyOp::name, g_poly));
    dp_poly.property("color", get(&PolyOp::color, g_poly));
    std::ofstream file(filename);
    boost::write_graphviz_dp(file, g_poly, dp_poly);
}

int main() {
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
    export_poly_graph(g_poly, "./data/graph_poly.dot");

    // Fuse Poly graph
    fuse_poly(g_poly);
    export_poly_graph(g_poly, "./data/graph_poly_fused_elemwise.dot");

    // AlphaWise fusion
    const bool alpha_wise = false;
    if (alpha_wise) {
        fuse_poly_alpha(g_poly);
        export_poly_graph(g_poly, "./data/graph_poly_fused_alpha.dot");
    } else {
        fuse_poly_beta(g_poly);
        export_poly_graph(g_poly, "./data/graph_poly_fused_beta.dot");
    }
}
