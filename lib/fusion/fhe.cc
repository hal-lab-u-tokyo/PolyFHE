#include "graph.h"

#include <iostream>

FHEOp NewHAdd(){
    return {"HAdd", {-1, -1}, {-1, -1}};
}

FHEOp NewHMult(){
    return {"HMult", {-1, -1}, {-1, -1}};
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

void lower_fhe_to_poly(GraphFHE &g_fhe, GraphPoly &g_poly){
    auto vertex_range = boost::vertices(g_fhe);
    for(auto it = vertex_range.first; it != vertex_range.second; ++it){
        GraphFHE::vertex_descriptor v = *it;
        if (g_fhe[v].name == "HAdd"){
            if (g_fhe[v].inputs.size() != 2){
                std::cerr << "HAdd should have 2 inputs" << std::endl;
                exit(1);
            }
            auto [out1, out2] = lower_hadd(g_poly, g_fhe[v].inputs[0], g_fhe[v].inputs[1]);

            // Search child nodes
            auto children = boost::adjacent_vertices(v, g_fhe);
            for(auto it = children.first; it != children.second; ++it){
                GraphPoly::vertex_descriptor child = *it;
                g_fhe[child].inputs[0] = out1;
                g_fhe[child].inputs[1] = out2;
            }
        }else if (g_fhe[v].name == "HMult"){
            if (g_fhe[v].inputs.size() != 2){
                std::cerr << "HMult should have 2 inputs" << std::endl;
                exit(1);
            }
            auto [out1, out2] = lower_hmult(g_poly, g_fhe[v].inputs[0], g_fhe[v].inputs[1]);
            
            // Search child nodes
            auto children = boost::adjacent_vertices(v, g_fhe);
            for(auto it = children.first; it != children.second; ++it){
                GraphPoly::vertex_descriptor child = *it;
                g_fhe[child].inputs[0] = out1;
                g_fhe[child].inputs[1] = out2;
            }

        }else{
            std::cerr << "Unknown operation: " << g_fhe[v].name << std::endl;
            exit(1);
        }
    } 
}