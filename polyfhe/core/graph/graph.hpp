#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "polyfhe/core/config.hpp"
#include "polyfhe/core/graph/edge.hpp"
#include "polyfhe/core/graph/node.hpp"
namespace polyfhe {
namespace core {

enum class GraphType { FHE, Poly, Other };

enum class sPolyType {
    sPolyP1,
    sPolyP2,
    sPolyLimbStrip,
    sPolySlotStrip,
};

enum class SubgraphType {
    Elem,
    ElemLimb1,
    ElemLimb2,
    ElemSlot,
    ElemLimb1Slot,
    ElemLimb2Slot,
    NoAccess,
};

struct KernelLaunchConfig {
    std::string grid_size;
    std::string block_size;
    std::string shared_mem_size;
};

std::ostream &operator<<(std::ostream &os, const SubgraphType &subgraph_type);
std::string to_string(SubgraphType subgraph_type);

class SubGraph {
public:
    void add_node(std::shared_ptr<Node> node) { m_nodes.push_back(node); }
    std::vector<std::shared_ptr<Node>> &get_nodes() { return m_nodes; }

    // idx
    void set_idx(int idx) { m_idx = idx; }
    int get_idx() { return m_idx; }

    // name
    std::string get_name() {
        if (m_name.empty()) {
            for (auto node : m_nodes) {
                m_name += node->get_op_name() + "_";
            }
            m_name.pop_back();
        }
        return m_name;
    }

    // block phase
    void set_block_phase(BlockPhase block_phase) {
        m_block_phase = block_phase;
    }
    BlockPhase get_block_phase() { return m_block_phase; }

    // batch
    int get_nx_batch() { return m_nx_batch; }
    int get_ny_batch() { return m_ny_batch; }
    void set_nx_batch(int nx_batch) { m_nx_batch = nx_batch; }
    void set_ny_batch(int ny_batch) { m_ny_batch = ny_batch; }

    // subgraph type
    void set_subgraph_type(SubgraphType subgraph_type) {
        m_subgraph_type = subgraph_type;
    }
    SubgraphType get_subgraph_type() { return m_subgraph_type; }

    // Shared memory size
    void set_smem_size(int smem_size) { m_smem_size = smem_size; }
    int get_smem_size() { return m_smem_size; }

    // max limb
    int get_max_limb();

    // Kernel launch config
    void set_kernel_launch_config(KernelLaunchConfig c) {
        m_kernel_launch_config = c;
    }
    KernelLaunchConfig get_kernel_launch_config() {
        return m_kernel_launch_config;
    }

    // Search node of specified OpType and return the first one
    // If the number of found node is different with `n_found`,
    // raise assertion
    std::shared_ptr<Node> search_op(core::OpType op_type, int n_found);

private:
    std::vector<std::shared_ptr<Node>> m_nodes;
    int m_idx;
    std::string m_name;

    SubgraphType m_subgraph_type;
    sPolyType m_sPoly_type;
    BlockPhase m_block_phase;
    int m_nx_batch;
    int m_ny_batch;
    int m_gridX;
    int m_blockX;
    int m_smem_size;

    KernelLaunchConfig m_kernel_launch_config;
};

class Graph {
public:
    Graph(std::shared_ptr<Config>(config)) : m_config(config) {}
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst);
    void add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst,
                  std::string label);
    void add_node(std::shared_ptr<Node> node);
    void remove_node(std::shared_ptr<Node> node);

    std::vector<std::shared_ptr<Node>> &get_nodes() { return m_nodes; }

    // Get the number of nodes in the graph.
    // Don't use get_nodes().size() to get number of nodes
    // because get_nodes().size() returns the number of nodes including nullptr
    int get_nodes_size() {
        int count = 0;
        for (auto node : m_nodes) {
            if (node != nullptr) {
                count++;
            }
        }
        return count;
    }

    void set_init_node(std::shared_ptr<Node> node) { m_init_node = node; }
    void set_exit_node(std::shared_ptr<Node> node) { m_exit_node = node; }
    std::shared_ptr<Node> get_init_node() { return m_init_node; }
    std::shared_ptr<Node> get_exit_node() { return m_exit_node; }
    int get_init_node_id() {
        assert(m_init_node->get_id() == 0);
        return m_init_node->get_id();
    }

    GraphType get_graph_type() { return m_graph_type; }
    void set_graph_type(GraphType graph_type) { m_graph_type = graph_type; }

    // Subgraph
    void add_subgraph(std::shared_ptr<SubGraph> subgraph) {
        subgraph->set_idx(m_subgraphs.size());
        m_subgraphs.push_back(subgraph);
    }
    std::vector<std::shared_ptr<SubGraph>> &get_subgraphs() {
        return m_subgraphs;
    }

    // PolyFHE Config
    std::shared_ptr<Config> get_m_config() { return m_config; }
    std::shared_ptr<Config> m_config;

private:
    std::vector<std::shared_ptr<Node>> m_nodes;
    std::shared_ptr<Node> m_init_node;
    std::shared_ptr<Node> m_exit_node;
    GraphType m_graph_type;

    std::vector<std::shared_ptr<SubGraph>> m_subgraphs;
};

std::shared_ptr<Edge> get_edge(std::shared_ptr<Node> src,
                               std::shared_ptr<Node> dst);

core::SubgraphType GetSubgraphType(
    std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph);

int GetsPolySize(std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph,
                 std::shared_ptr<Config> config);

int GetSubgraphSmemFoorprint(
    std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph,
    std::shared_ptr<Config> config);

void ExtractSubgraph(
    std::shared_ptr<polyfhe::core::Node> node,
    std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph);

} // namespace core
} // namespace polyfhe