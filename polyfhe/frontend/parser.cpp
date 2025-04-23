#include "polyfhe/frontend/parser.hpp"

#include "polyfhe/core/logger.hpp"

namespace polyfhe {
namespace frontend {

graph_t ParseDot(const std::string& dot) {
    LOG_INFO("Parsing %s\n", dot.c_str());

    // Boost graph
    graph_t g_dot(0);
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("node_id", boost::get(&DotNode::name, g_dot));
    dp.property("label", boost::get(&DotNode::label, g_dot));
    dp.property("peripheries", boost::get(&DotNode::peripheries, g_dot));
    dp.property("label", boost::get(&DotEdge::label, g_dot));

    // Read dot file
    std::ifstream gvgraph(dot);
    if (!boost::read_graphviz(gvgraph, g_dot, dp)) {
        LOG_ERROR("Failed to parse graphviz dot file");
        exit(1);
    }
    LOG_INFO("Successfully read dot file\n");
    return g_dot;
}

std::shared_ptr<polyfhe::core::Graph> ConvertDotToGraph(
    const graph_t& g_dot, polyfhe::core::GraphType graph_type,
    std::shared_ptr<polyfhe::Config> config) {
    std::shared_ptr<polyfhe::core::Graph> graph_polyfhe =
        std::make_shared<polyfhe::core::Graph>(config);

    // Add nodes in DFS order from the init node
    const int n = boost::num_vertices(g_dot);
    std::vector<std::shared_ptr<polyfhe::core::Node>> visited(n, nullptr);
    std::vector<int> stack;
    LOG_INFO("Number of nodes: %d\n", n);

    // Find the node labled "Init"
    for (int i = 0; i < n; i++) {
        if (g_dot[i].label == "Init") {
            stack.push_back(i);
            break;
        }
    }
    if (stack.size() != 1) {
        LOG_ERROR("%ld nodes labeled 'Init' found\n", stack.size());
        exit(1);
    }

    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        if (visited[v]) {
            continue;
        }
        std::shared_ptr<polyfhe::core::Node> node =
            std::make_shared<polyfhe::core::Node>(g_dot[v].label);
        visited[v] = node;
        if (node->get_op_type() == core::OpType::Init) {
            if (graph_polyfhe->get_init_node()) {
                LOG_ERROR("Multiple init nodes found\n");
                exit(1);
            }
            graph_polyfhe->set_init_node(node);
        } else if (node->get_op_type() == core::OpType::End) {
            if (graph_polyfhe->get_exit_node()) {
                LOG_ERROR("Multiple exit nodes found\n");
                exit(1);
            }
            graph_polyfhe->set_exit_node(node);
        }

        // Add Node
        graph_polyfhe->add_node(node);

        for (auto it = boost::adjacent_vertices(v, g_dot);
             it.first != it.second; ++it.first) {
            stack.push_back(*it.first);
        }
    }

    // Add Edges
    for (int i = 0; i < n; i++) {
        std::shared_ptr<polyfhe::core::Node> src = visited[i];
        if (!src) {
            LOG_ERROR("Node %d has not visited\n", i);
            exit(1);
        }

        std::vector<bool> visited_edges(n, false);
        for (auto it = boost::adjacent_vertices(i, g_dot);
             it.first != it.second; ++it.first) {
            // Get destination node
            int j = *it.first;
            if (visited_edges[j]) {
                continue;
            }
            visited_edges[j] = true;
            std::shared_ptr<polyfhe::core::Node> dst = visited[j];
            if (!dst) {
                LOG_ERROR("Node %d not visited\n", j);
                exit(1);
            }
            std::cout << "src: " << src->get_op_name()
                      << ", dst: " << dst->get_op_name() << std::endl;

            // Get DotEdge
            auto i_node = boost::vertex(i, g_dot);
            auto j_node = boost::vertex(j, g_dot);
            auto [ei, ei_end] = boost::out_edges(i_node, g_dot);
            for (; ei != ei_end; ++ei) {
                if (boost::target(*ei, g_dot) != j_node) {
                    continue;
                }
                DotEdge edge = g_dot[*ei];
                graph_polyfhe->add_edge(src, dst, edge.label);
            }
        }
    }

    graph_polyfhe->set_graph_type(graph_type);
    return graph_polyfhe;
}

std::shared_ptr<polyfhe::core::Graph> ParseDotToGraph(
    const std::string& dot, polyfhe::core::GraphType graph_type,
    std::shared_ptr<polyfhe::Config> config) {
    graph_t g_dot = ParseDot(dot);
    std::shared_ptr<polyfhe::core::Graph> graph =
        ConvertDotToGraph(g_dot, graph_type, config);
    LOG_INFO("Successfully converted dot to graph, %ld nodes\n",
             graph->get_nodes().size());
    return graph;
}

} // namespace frontend
} // namespace polyfhe