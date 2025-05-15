#pragma once

#include "polyfhe/engine/pass/pass_base.hpp"

namespace polyfhe {
namespace engine {
class CheckEdgeSamePass : public PassBase {
public:
    bool run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) override;
    std::string get_name() override { return "CheckEdgeSamePass"; }
};
} // namespace engine
} // namespace polyfhe