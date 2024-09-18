#pragma once

#include <fstream>
#include <utility>
#include <string>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>

enum AccessPattern {
    ElemWise,
    PolyWise,
    CoeffWise
};

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

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, PolyOp> GraphPoly;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, FHEOp> GraphFHE;

std::pair<int, int> lower_hadd(GraphPoly &g, int in1, int in2);
std::pair<int, int> lower_hmult(GraphPoly &g, int in1, int in2);

void dummy_fhe_graph(GraphFHE &g);
void lower_fhe_to_poly(GraphFHE &g_fhe, GraphPoly &g_poly);