#include "hifive/frontend/parser.hpp"

#include "hifive/core/logger.hpp"

namespace hifive {
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

std::shared_ptr<hifive::core::Graph> ConvertDotToGraph(
    const graph_t& g_dot, hifive::core::GraphType graph_type) {
    std::shared_ptr<hifive::core::Graph> graph_hifive =
        std::make_shared<hifive::core::Graph>();

    // Add nodes in DFS order from the init node
    const int n = boost::num_vertices(g_dot);
    std::vector<std::shared_ptr<hifive::core::Node>> visited(n, nullptr);
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
        std::shared_ptr<hifive::core::Node> node =
            std::make_shared<hifive::core::Node>(g_dot[v].label);
        visited[v] = node;
        if (node->get_op_type() == "Init") {
            if (graph_hifive->get_init_node()) {
                LOG_ERROR("Multiple init nodes found\n");
                exit(1);
            }
            graph_hifive->set_init_node(node);
        } else if (node->get_op_type() == "End") {
            if (graph_hifive->get_exit_node()) {
                LOG_ERROR("Multiple exit nodes found\n");
                exit(1);
            }
            graph_hifive->set_exit_node(node);
        }

        // Add Node
        graph_hifive->add_node(node);

        for (auto it = boost::adjacent_vertices(v, g_dot);
             it.first != it.second; ++it.first) {
            stack.push_back(*it.first);
        }
    }

    // Add Edges
    for (int i = 0; i < n; i++) {
        std::shared_ptr<hifive::core::Node> src = visited[i];
        if (!src) {
            LOG_ERROR("Node %d not visited\n", i);
            exit(1);
        }
        for (auto it = boost::adjacent_vertices(i, g_dot);
             it.first != it.second; ++it.first) {
            // Get destination node
            int j = *it.first;
            std::shared_ptr<hifive::core::Node> dst = visited[j];
            if (!dst) {
                LOG_ERROR("Node %d not visited\n", j);
                exit(1);
            }

            // Get DotEdge
            graph_t::edge_descriptor e = boost::edge(i, *it.first, g_dot).first;
            DotEdge edge = g_dot[e];
            graph_hifive->add_edge(src, dst, edge.label);
        }
    }

    graph_hifive->set_graph_type(graph_type);
    return graph_hifive;
}

std::shared_ptr<hifive::core::Graph> ParseDotToGraph(
    const std::string& dot, hifive::core::GraphType graph_type) {
    graph_t g_dot = ParseDot(dot);
    std::shared_ptr<hifive::core::Graph> graph =
        ConvertDotToGraph(g_dot, graph_type);
    LOG_INFO("Successfully converted dot to graph, %ld nodes\n",
             graph->get_nodes().size());
    return graph;
}

} // namespace frontend
} // namespace hifive