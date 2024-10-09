#include "hifive/frontend/exporter.hpp"

#include "hifive/core/logger.hpp"

namespace hifive {
namespace frontend {

void export_graph_to_dot(std::shared_ptr<hifive::core::Graph>& graph,
                         std::string filename) {
    LOG_INFO("Exporting graph to %s\n", filename.c_str());

    graph_t g_dot;

    // Add nodes
    const int n = graph->get_nodes().size();
    std::vector<graph_t::vertex_descriptor> v_descs(n);
    for (auto node : graph->get_nodes()) {
        if (node == nullptr) {
            continue;
        }
        DotNode dn;
        dn.name = node->get_op_name();
        dn.label = node->get_op_type();
        dn.peripheries = 1;
        v_descs[node->get_id()] = boost::add_vertex(dn, g_dot);
    }

    // Add edges
    for (auto node : graph->get_nodes()) {
        if (node == nullptr) {
            continue;
        }
        const int node_id = node->get_id();
        for (auto edge : node->get_out_edges()) {
            DotEdge de;
            de.label = "";
            auto src = v_descs[node_id];
            auto dst = v_descs[edge->get_dst()->get_id()];
            boost::add_edge(src, dst, de, g_dot);
        }
    }

    // Write to file
    boost::dynamic_properties dp;
    dp.property("node_id", get(&DotNode::name, g_dot));
    dp.property("label", get(&DotNode::label, g_dot));
    dp.property("peripheries", get(&DotNode::peripheries, g_dot));
    std::ofstream file(filename);
    boost::write_graphviz_dp(file, g_dot, dp);
}

} // namespace frontend
} // namespace hifive