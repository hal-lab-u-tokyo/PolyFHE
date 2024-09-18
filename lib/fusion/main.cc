#include <fstream>
#include <utility>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>

typedef std::pair<int, int> Edge;

struct PolyOp {
    std::string name;
};

int main()
{
    typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, PolyOp> Graph;
   
    Graph g;

    Graph::vertex_descriptor v0 = boost::add_vertex(g);
    g[v0].name = "Add";

    Graph::vertex_descriptor v1 = boost::add_vertex(g);
    g[v1].name = "Sub";

    boost::add_edge(v0, v1, g);

    boost::print_graph(g, get(&PolyOp::name, g));

    std::ofstream file("graph.dot");
    boost::write_graphviz(file, g, make_label_writer(get(&PolyOp::name, g)));
}
