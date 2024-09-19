#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

enum AccessPattern { ElemWise, PolyWise, AlphaWise, BetaWise, Other };

struct PolyOp {
public:
    std::string name;
    AccessPattern access_pattern;
    std::string color;
};

struct FHEOp {
    std::string name;
    std::vector<int> inputs;
    std::vector<int> outputs;
};

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS,
                              PolyOp>
    GraphPoly;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS,
                              FHEOp>
    GraphFHE;

std::pair<int, int> lower_hadd(GraphPoly &g, int ct0_ax, int ct0_bx, int ct1_ax,
                               int ct1_bx);
std::pair<int, int> lower_hmult(GraphPoly &g, int ct0_ax, int ct0_bx,
                                int ct1_ax, int ct1_bx);

void dummy_fhe_graph(GraphFHE &g);
void lower_fhe_to_poly(GraphFHE &g_fhe, GraphPoly &g_poly);
void fuse_poly(GraphPoly &g_poly);
void fuse_poly_alpha(GraphPoly &g_poly);
void fuse_poly_beta(GraphPoly &g_poly);