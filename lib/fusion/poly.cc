#include "graph.h"

std::string getColor(AccessPattern ap) {
    switch (ap) {
        case ElemWise:
            return "dodgerblue2";
        case PolyWise:
            return "coral1";
        case CoeffWise:
            return "green4";
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

void define_hmult(GraphPoly &g) {
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
}
