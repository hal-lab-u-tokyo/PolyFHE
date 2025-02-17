#pragma once

#include <memory>

#include "polyfhe/core/graph/graph.hpp"

namespace polyfhe {
namespace engine {
class PassBase {
public:
    virtual bool run_on_graph(
        std::shared_ptr<polyfhe::core::Graph>& /*graph*/) {
        return true;
    }
    virtual std::string get_name() { return "PassBase"; }
};
} // namespace engine
} // namespace polyfhe