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
    int get_shape(size_t i) {
        assert(i < m_shape.size());
        return m_shape[i];
    }

private:
    std::shared_ptr<Node> m_src;
    std::shared_ptr<Node> m_dst;
    std::vector<int> m_shape;
};
} // namespace core
} // namespace hifive