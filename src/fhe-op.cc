#include "graph.h"

#include <iostream>

FHEOp NewHAdd(){
    return {"HAdd", {-1, -1, -1, -1}, {-1, -1}};
}

FHEOp NewHMult(){
    return {"HMult", {-1, -1, -1, -1}, {-1, -1}};
}

void dummy_fhe_graph(GraphFHE &g){
    GraphFHE::vertex_descriptor add0 = boost::add_vertex(g);
    GraphFHE::vertex_descriptor add1 = boost::add_vertex(g);
    GraphFHE::vertex_descriptor mult0 = boost::add_vertex(g);
    g[add0] = NewHAdd();
    g[add1] = NewHAdd();
    g[mult0] = NewHMult();
    boost::add_edge(add0, mult0, g);
    boost::add_edge(add1, mult0, g);
}

void update_children(GraphFHE &g, GraphFHE::vertex_descriptor v, int out1, int out2){
    auto children = boost::adjacent_vertices(v, g);
    for(auto it = children.first; it != children.second; ++it){
        GraphFHE::vertex_descriptor child = *it;
        if (g[child].inputs.size() == 2){
            g[child].inputs[0] = out1;
            g[child].inputs[1] = out2;
        }else if (g[child].inputs.size() == 4){
            if (g[child].inputs[0] == -1){
                g[child].inputs[0] = out1;
                g[child].inputs[1] = out2;
            }else{
                g[child].inputs[2] = out1;
                g[child].inputs[3] = out2;
            }
        }else {
            std::cerr << "HAdd child should have 2 or 4 inputs" << std::endl;
            exit(1);
        }
    }
}

void lower_fhe_to_poly(GraphFHE &g_fhe, GraphPoly &g_poly){
    auto vertex_range = boost::vertices(g_fhe);
    for(auto it = vertex_range.first; it != vertex_range.second; ++it){
        GraphFHE::vertex_descriptor v = *it;
        if (g_fhe[v].name == "HAdd"){
            // Check inputs
            if (g_fhe[v].inputs.size() != 4){
                std::cerr << "HAdd should have 4 inputs" << std::endl;
                exit(1);
            }
            
            // Lower HAdd to Poly
            auto [out1, out2] = lower_hadd(g_poly, g_fhe[v].inputs[0], g_fhe[v].inputs[1], g_fhe[v].inputs[2], g_fhe[v].inputs[3]);

            // Update children's inputs
            update_children(g_fhe, v, out1, out2);
            
        }else if (g_fhe[v].name == "HMult"){
            // Check inputs
            if (g_fhe[v].inputs.size() != 4){
                std::cerr << "HMult should have 4 inputs" << std::endl;
                exit(1);
            }

            // Lower HMult to Poly
            auto [out1, out2] = lower_hmult(g_poly, g_fhe[v].inputs[0], g_fhe[v].inputs[1], g_fhe[v].inputs[2], g_fhe[v].inputs[3]);
            
            // Update children's inputs
            update_children(g_fhe, v, out1, out2);
        }else{
            std::cerr << "Unknown operation: " << g_fhe[v].name << std::endl;
            exit(1);
        }
    } 
}