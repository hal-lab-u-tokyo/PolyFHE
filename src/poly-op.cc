#include "graph.h"

std::string getColor(AccessPattern ap) {
    switch (ap) {
        case ElemWise:
            return "dodgerblue2";
        case PolyWise:
            return "coral1";
        case CoeffWise:
            return "green4";
        case Other:
            return "black";
    }
    return "black";
}

PolyOp NewAdd(){
    return {"Add", ElemWise, getColor(ElemWise)};
}

PolyOp NewMult(){
    return {"Mult", ElemWise, getColor(ElemWise)};
}

PolyOp NewKeySwitch(){
    return {"KeySwitch", ElemWise, getColor(ElemWise)};
}

PolyOp NewModDown(){
    return {"ModDown", CoeffWise, getColor(CoeffWise)};
}

PolyOp NewModUp(){
    return {"ModUp", CoeffWise, getColor(CoeffWise)};
}

PolyOp NewReduce(){
    return {"Reduce", CoeffWise, getColor(CoeffWise)};
}

PolyOp NewNTT(){
    return {"NTT", PolyWise, getColor(PolyWise)};
}

PolyOp NewINTT(){
    return {"INTT", PolyWise, getColor(PolyWise)};
}

PolyOp NewMalloc(){
    return {"Malloc", Other, getColor(Other)};
}

void connect_edge(int in, GraphPoly::vertex_descriptor to, GraphPoly &g) {
    if (in != -1) {
        GraphPoly::vertex_descriptor in_v = boost::vertex(in, g);
        boost::add_edge(in_v, to, g);
    }else {
        GraphPoly::vertex_descriptor malloc = boost::add_vertex(g);
        g[malloc] = NewMalloc();
        boost::add_edge(malloc, to, g);
    }
}

std::pair<int, int> lower_hadd(GraphPoly &g, int ct0_ax, int ct0_bx, int ct1_ax, int ct1_bx) {
    GraphPoly::vertex_descriptor add_ax = boost::add_vertex(g);
    GraphPoly::vertex_descriptor add_bx = boost::add_vertex(g);
    g[add_ax] = NewAdd();
    g[add_bx] = NewAdd();
    connect_edge(ct0_ax, add_ax, g);
    connect_edge(ct0_bx, add_bx, g);
    connect_edge(ct1_ax, add_ax, g);
    connect_edge(ct1_bx, add_bx, g);
    const int out1 = boost::get(boost::vertex_index, g, add_ax);
    const int out2 = boost::get(boost::vertex_index, g, add_bx);
    return std::make_pair(out1, out2);
}

std::pair<int, int> lower_hmult(GraphPoly &g, int ct0_ax, int ct0_bx, int ct1_ax, int ct1_bx) {
    // Mult
    GraphPoly::vertex_descriptor mult_axax = boost::add_vertex(g);
    GraphPoly::vertex_descriptor mult_axbx = boost::add_vertex(g);
    GraphPoly::vertex_descriptor mult_bxax = boost::add_vertex(g);
    GraphPoly::vertex_descriptor mult_bxbx = boost::add_vertex(g);
    GraphPoly::vertex_descriptor add_axbx_bxax = boost::add_vertex(g);
    g[mult_axax] = NewMult();
    g[mult_axbx] = NewMult();
    g[mult_bxax] = NewMult();
    g[mult_bxbx] = NewMult();
    g[add_axbx_bxax] = NewAdd();
    boost::add_edge(mult_axbx, add_axbx_bxax, g);
    boost::add_edge(mult_bxax, add_axbx_bxax, g);
    connect_edge(ct0_ax, mult_axax, g);
    connect_edge(ct1_ax, mult_axax, g);
    connect_edge(ct0_bx, mult_bxbx, g);
    connect_edge(ct1_bx, mult_bxbx, g);
    connect_edge(ct0_ax, mult_axbx, g);
    connect_edge(ct1_bx, mult_axbx, g);
    connect_edge(ct0_bx, mult_bxax, g);
    connect_edge(ct1_ax, mult_bxax, g);
    
    // KeySwitch
    GraphPoly::vertex_descriptor intt = boost::add_vertex(g);
    GraphPoly::vertex_descriptor modup = boost::add_vertex(g);
    GraphPoly::vertex_descriptor ntt = boost::add_vertex(g);
    GraphPoly::vertex_descriptor mult = boost::add_vertex(g);
    GraphPoly::vertex_descriptor reduce = boost::add_vertex(g);
    GraphPoly::vertex_descriptor intt_after_ksw = boost::add_vertex(g);
    GraphPoly::vertex_descriptor moddown = boost::add_vertex(g);
    GraphPoly::vertex_descriptor ntt_after_moddown = boost::add_vertex(g);
    g[intt] = NewINTT();
    g[modup] = NewModUp();
    g[ntt] = NewNTT();
    g[mult] = NewMult();
    g[reduce] = NewReduce();
    g[intt_after_ksw] = NewINTT();
    g[moddown] = NewModDown();
    g[ntt_after_moddown] = NewNTT();
    boost::add_edge(add_axbx_bxax, intt, g);
    boost::add_edge(intt, modup, g);
    boost::add_edge(modup, ntt, g);
    boost::add_edge(ntt, mult, g);
    boost::add_edge(mult, reduce, g);
    boost::add_edge(reduce, intt_after_ksw, g);
    boost::add_edge(intt_after_ksw, moddown, g);
    boost::add_edge(moddown, ntt_after_moddown, g);
    
    // Sum
    GraphPoly::vertex_descriptor add_c0c2 = boost::add_vertex(g);
    GraphPoly::vertex_descriptor add_c1c2 = boost::add_vertex(g);
    g[add_c0c2] = NewAdd();
    g[add_c1c2] = NewAdd();
    boost::add_edge(mult_axax, add_c0c2, g);
    boost::add_edge(mult_bxbx, add_c1c2, g);
    boost::add_edge(ntt_after_moddown, add_c0c2, g);
    boost::add_edge(ntt_after_moddown, add_c1c2, g);

    const int out1 = boost::get(boost::vertex_index, g, add_c0c2);
    const int out2 = boost::get(boost::vertex_index, g, add_c1c2);
    return std::make_pair(out1, out2);
}

void fuse_poly(GraphPoly &g_poly){
    // Depth first search
    const int n = boost::num_vertices(g_poly);
    std::cout << "Number of vertices: " << n << std::endl;

    std::vector<int> visited(n, 0);
    std::vector<int> stack;
    for (int i = 0; i < n; i++) {
        const GraphPoly::vertex_descriptor v = boost::vertex(i, g_poly);
        if (g_poly[v].name == "Malloc") {
            stack.push_back(i);
        }
    }

    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        if (visited[v] == 1) {
            continue;
        }
        visited[v] = 1;
        // fuse Elem-wise operations with adjacent PolyOp
        if (g_poly[v].access_pattern == ElemWise) {
            const int num_adjacent = boost::out_degree(v, g_poly);
            if (num_adjacent == 1) {
                const GraphPoly::vertex_descriptor adj = *boost::adjacent_vertices(v, g_poly).first;
                g_poly[adj].name = g_poly[v].name + "\n" + g_poly[adj].name;

                // connect v's parent to adj
                for (auto it = boost::in_edges(v, g_poly); it.first != it.second; ++it.first) {
                    const GraphPoly::vertex_descriptor parent = boost::source(*it.first, g_poly);
                    boost::add_edge(parent, adj, g_poly);
                }

                // remove v
                boost::clear_vertex(v, g_poly);
                boost::remove_vertex(v, g_poly);
            }
        }
        stack.push_back(v);
        for (auto it = boost::adjacent_vertices(v, g_poly); it.first != it.second; ++it.first) {
            stack.push_back(*it.first);
        }
    }
}