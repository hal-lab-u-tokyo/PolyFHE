#pragma once

#include <memory>

#include "polyfhe/core/graph/graph.hpp"

namespace polyfhe {
namespace engine {
class CodegenBase {
public:
    virtual bool run_on_graph(
        std::shared_ptr<polyfhe::core::Graph>& /*graph*/) {
        return true;
    }
};
} // namespace engine
} // namespace polyfhe