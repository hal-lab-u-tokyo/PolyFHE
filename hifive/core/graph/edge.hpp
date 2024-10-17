#pragma once

#include <vector>

#include "hifive/core/graph/node.hpp"

namespace hifive {
namespace core {

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
    int get_shape(int idx) { return m_shape[idx]; }
    int get_size_in_byte() { return get_size() * sizeof(uint64_t); }

    // name
    void update_name() {
        const int src_id = m_src->get_idx_in_outedge(shared_from_this());
        const int dst_id = m_dst->get_idx_in_inedge(shared_from_this());
        m_name = "edge_" + m_src->get_op_name() + "_" + std::to_string(src_id) +
                 "_" + m_dst->get_op_name() + "_" + std::to_string(dst_id);
    }
    std::string get_name() { return m_name; }

private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;
    std::vector<int> m_shape;
    std::string m_name;

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