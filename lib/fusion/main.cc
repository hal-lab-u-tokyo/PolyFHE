#include <fstream>
#include <utility>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>

struct PolyOp {
    std::string name;
};

typedef std::pair<int, int> Edge;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, PolyOp> Graph;

void define_hmult(Graph &g) {
    // Mult
    Graph::vertex_descriptor mult_axax = boost::add_vertex(g);
    Graph::vertex_descriptor mult_axbx = boost::add_vertex(g);
    Graph::vertex_descriptor mult_bxax = boost::add_vertex(g);
    Graph::vertex_descriptor mult_bxbx = boost::add_vertex(g);
    Graph::vertex_descriptor add_axbx_bxax = boost::add_vertex(g);
    g[mult_axax].name = "Mult";
    g[mult_axbx].name = "Mult";
    g[mult_bxax].name = "Mult";
    g[mult_bxbx].name = "Mult";
    g[add_axbx_bxax].name = "Add";
    boost::add_edge(mult_axbx, add_axbx_bxax, g);
    boost::add_edge(mult_bxax, add_axbx_bxax, g);

    // KeySwitch
    Graph::vertex_descriptor intt = boost::add_vertex(g);
    Graph::vertex_descriptor modup = boost::add_vertex(g);
    Graph::vertex_descriptor ntt = boost::add_vertex(g);
    Graph::vertex_descriptor mult = boost::add_vertex(g);
    Graph::vertex_descriptor reduce = boost::add_vertex(g);
    Graph::vertex_descriptor intt_after_ksw = boost::add_vertex(g);
    Graph::vertex_descriptor moddown = boost::add_vertex(g);
    Graph::vertex_descriptor ntt_after_moddown = boost::add_vertex(g);
    g[intt].name = "INTT";
    g[modup].name = "ModUp";
    g[ntt].name = "NTT";
    g[mult].name = "Mult";
    g[reduce].name = "Reduce";
    g[intt_after_ksw].name = "INTT";
    g[moddown].name = "ModDown";
    g[ntt_after_moddown].name = "NTT";
    boost::add_edge(add_axbx_bxax, intt, g);
    boost::add_edge(intt, modup, g);
    boost::add_edge(modup, ntt, g);
    boost::add_edge(ntt, mult, g);
    boost::add_edge(mult, reduce, g);
    boost::add_edge(reduce, intt_after_ksw, g);
    boost::add_edge(intt_after_ksw, moddown, g);
    boost::add_edge(moddown, ntt_after_moddown, g);
    
    // Sum
    Graph::vertex_descriptor add_c0c2 = boost::add_vertex(g);
    Graph::vertex_descriptor add_c1c2 = boost::add_vertex(g);
    g[add_c0c2].name = "Add";
    g[add_c1c2].name = "Add";
    boost::add_edge(mult_axax, add_c0c2, g);
    boost::add_edge(mult_bxbx, add_c1c2, g);
    boost::add_edge(ntt_after_moddown, add_c0c2, g);
    boost::add_edge(ntt_after_moddown, add_c1c2, g);
}

int main()
{
    Graph g;

    define_hmult(g);

    boost::print_graph(g, get(&PolyOp::name, g));

    std::ofstream file("graph.dot");
    boost::write_graphviz(file, g, make_label_writer(get(&PolyOp::name, g)));
}
