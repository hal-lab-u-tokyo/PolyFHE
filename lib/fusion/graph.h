#pragma once

#include <fstream>
#include <utility>
#include <string>
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
};

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, PolyOp> GraphPoly;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, FHEOp> GraphFHE;

void dummy_fhe_graph(GraphFHE &g);
void define_hmult(GraphPoly &g);