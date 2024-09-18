#include "graph.h"

FHEOp NewHAdd(){
    return {"HAdd"};
}

FHEOp NewHMult(){
    return {"HMult"};
}

void dummy_fhe_graph(GraphFHE &g){
    GraphFHE::vertex_descriptor add1 = boost::add_vertex(g);
    GraphFHE::vertex_descriptor add2 = boost::add_vertex(g);
    GraphFHE::vertex_descriptor mult1 = boost::add_vertex(g);
    g[add1] = NewHAdd();
    g[add2] = NewHAdd();
    g[mult1] = NewHMult();
    boost::add_edge(add1, mult1, g);
    boost::add_edge(add2, mult1, g);
}