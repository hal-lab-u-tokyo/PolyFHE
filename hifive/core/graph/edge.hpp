#pragma once

#include <vector>

#include "hifive/core/graph/node.hpp"

namespace hifive {
namespace core {

class Edge {
public:
    Edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst)
        : m_src(src), m_dst(dst) {}
    std::shared_ptr<Node> get_src() { return m_src; }
    std::shared_ptr<Node> get_dst() { return m_dst; }
    void set_src(std::shared_ptr<Node> src) { m_src = src; }
    void set_dst(std::shared_ptr<Node> dst) { m_dst = dst; }

    void set_shape(std::vector<int> shape) { m_shape = shape; }
    std::vector<int> get_shape() { return m_shape; }
    int get_shape(int idx) { return m_shape[idx]; }
    int get_size_in_byte() { return get_size() * sizeof(uint64_t); }

private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;
    std::vector<int> m_shape;

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