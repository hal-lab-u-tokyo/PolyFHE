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

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, PolyOp> Graph;

void define_hmult(Graph &g);