#pragma once

#include <memory>
#include <vector>

#include "polyfhe/core/graph/node.hpp"

namespace polyfhe {
namespace core {

enum class EdgeLevel {
    Register,
    Shared,
    Global,
    YetToDetermine,
};

class Edge : public std::enable_shared_from_this<Edge> {
public:
    Edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst)
        : m_src(src), m_dst(dst) {}

    // src and dst nodes
    std::shared_ptr<Node> get_src() { return m_src; }
    std::shared_ptr<Node> get_dst() { return m_dst; }
    void set_src(std::shared_ptr<Node> src) { m_src = src; }
    void set_dst(std::shared_ptr<Node> dst) { m_dst = dst; }

    // limb
    void set_limb(int limb) { m_limb = limb; }
    void set_start_limb(int limb) { m_start_limb = limb; }
    void set_end_limb(int limb) { m_end_limb = limb; }
    int get_limb() { return m_limb; }
    int get_start_limb() { return m_start_limb; }
    int get_end_limb() { return m_end_limb; }

    // name
    void update_name() {
        const int src_id = m_src->get_idx_in_outedge(shared_from_this());
        const int dst_id = m_dst->get_idx_in_inedge(shared_from_this());
        m_name = "edge_" + m_src->get_op_name() + "_" + std::to_string(src_id) +
                 "_" + m_dst->get_op_name() + "_" + std::to_string(dst_id);
    }
    std::string get_name() {
        update_name();
        return m_name;
    }

    // level
    void set_level(EdgeLevel level) { m_level = level; }
    EdgeLevel get_level() { return m_level; }

    // Check if memory of this edge can be overwritten
    bool can_overwrite() {
        if (m_src->get_out_edges().size() > 1) {
            return false;
        }
        return true;
    }

    // offset
    void set_offset_smem(int offset) { m_offset_smem = offset; }
    int get_offset_smem() { return m_offset_smem; }

    // overwrite
    void set_can_overwrite(bool can_overwrite) {
        m_can_overwrite = can_overwrite;
    }
    bool get_can_overwrite() { return m_can_overwrite; }

    // init/end node
    void set_idx_argc(int idx_argc) {
        if (m_src->get_op_type() != core::OpType::Init &&
            m_dst->get_op_type() != core::OpType::End) {
            LOG_ERROR("set_idx_argc: not special node\n");
            exit(1);
        }
        m_idx_argc = idx_argc;
    }
    int get_idx_argc() {
        if (m_src->get_op_type() != core::OpType::Init &&
            m_dst->get_op_type() != core::OpType::End) {
            LOG_ERROR("get_idx_argc: not special node\n");
            exit(1);
        }
        return m_idx_argc;
    }
    void set_offset(int offset) { m_offset = offset; }
    int get_offset() { return m_offset; }

    // For codegen
    void set_has_defined(bool has_defined) { m_has_defined = has_defined; }
    bool get_has_defined() { return m_has_defined; }

    void set_same_edge(std::shared_ptr<Edge> edge) { same_edge = edge; }
    std::shared_ptr<Edge> get_same_edge() { return same_edge; }

    void set_overwrite_edge(std::shared_ptr<Edge> edge) {
        overwrite_edge = edge;
    }
    std::shared_ptr<Edge> get_overwrite_edge() { return overwrite_edge; }
    std::shared_ptr<Edge> get_overwrite_edge_final() {
        if (overwrite_edge) {
            if (overwrite_edge->get_overwrite_edge()) {
                return overwrite_edge->get_overwrite_edge_final();
            } else {
                return overwrite_edge;
            }
        } else {
            return nullptr;
        }
    }

private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;

    // Note that (current limb) != (end - start)
    // range of limb which dst Node uses
    int m_start_limb = 0;
    int m_end_limb = 0;
    // current limb
    int m_limb = 0;

    std::string m_name;
    EdgeLevel m_level = EdgeLevel::Global;
    int m_offset_smem = 0;
    bool m_can_overwrite = false;

    int m_offset = 0;
    // Only for init/end node
    int m_idx_argc = 0;

    // For codegen
    bool m_has_defined = false;

    std::shared_ptr<Edge> same_edge;
    std::shared_ptr<Edge> overwrite_edge;
};

} // namespace core
} // namespace polyfhe