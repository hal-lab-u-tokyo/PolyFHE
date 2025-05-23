#include "polyfhe/frontend/exporter.hpp"

#include "polyfhe/core/graph/edge.hpp"
#include "polyfhe/core/logger.hpp"

namespace polyfhe {
namespace frontend {

std::string EdgeLevelColor(polyfhe::core::EdgeLevel level) {
    switch (level) {
    case polyfhe::core::EdgeLevel::Register:
        return "skyblue";
    case polyfhe::core::EdgeLevel::Shared:
        return "blue";
    case polyfhe::core::EdgeLevel::L2:
        return "green";
    case polyfhe::core::EdgeLevel::Global:
        return "red";
    case polyfhe::core::EdgeLevel::YetToDetermine:
        return "black";
    default:
        LOG_ERROR("Unknown edge level\n");
        return "black";
    }
}

void export_graph_to_dot(std::shared_ptr<polyfhe::core::Graph>& graph,
                         std::string filename) {
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
        dn.label = node->get_op_name();
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
            std::string limb_range =
                "[" + std::to_string(edge->get_start_limb()) + "," +
                std::to_string(edge->get_end_limb()) + "]";
            de.label = limb_range;
            de.color = EdgeLevelColor(edge->get_level());
            auto src = v_descs[node_id];
            auto dst = v_descs[edge->get_dst()->get_id()];
            boost::add_edge(src, dst, de, g_dot);
        }
    }

    // Write to file
    boost::dynamic_properties dp;
    dp.property("node_id", get(&DotNode::name, g_dot));
    dp.property("label", get(&DotNode::label, g_dot));
    dp.property("color", get(&DotEdge::color, g_dot));
    dp.property("label", get(&DotEdge::label, g_dot));
    dp.property("peripheries", get(&DotNode::peripheries, g_dot));
    std::ofstream file(filename);
    boost::write_graphviz_dp(file, g_dot, dp);
}

} // namespace frontend
} // namespace polyfhe