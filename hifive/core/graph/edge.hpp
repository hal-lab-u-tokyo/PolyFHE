#pragma once

#include <vector>

#include "hifive/core/graph/node.hpp"

namespace hifive {
namespace core {

enum class EdgeLevel {
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

    // shape
    void set_shape(std::vector<int> shape) { m_shape = shape; }
    std::vector<int> get_shape() { return m_shape; }
    int get_shape(size_t idx) {
        if (idx >= m_shape.size()) {
            LOG_ERROR("Index out of bound\n");
            return 1;
        }
        return m_shape[idx];
    }
    int get_size_in_byte() { return get_size() * sizeof(uint64_t); }

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

    // Returns the edge that contains the same result as this edge
    // i.e., the first global edge of the same src node
    std::shared_ptr<Edge> get_same_result_edge() {
        for (auto edge : m_src->get_out_edges()) {
            if (edge->get_level() == EdgeLevel::Global) {
                return edge;
            }
        }
        return nullptr;
    }

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

private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;
    std::vector<int> m_shape;
    std::string m_name;
    EdgeLevel m_level = EdgeLevel::Global;
    int m_offset_smem = 0;
    bool m_can_overwrite = false;

    int get_size() {
        int size = 1;
        for (auto s : m_shape) {
            size *= s;
        }
        return size;
    }
};

} // namespace core
} // namespace hifive