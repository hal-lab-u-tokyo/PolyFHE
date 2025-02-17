#pragma once

#include "polyfhe/engine/pass/pass_base.hpp"

namespace polyfhe {
namespace engine {
class LoweringCKKSToPolyPass : public PassBase {
public:
    bool run_on_graph(
        std::shared_ptr<polyfhe::core::Graph>& /*graph*/) override;
    std::string get_name() override { return "LoweringCKKSToPolyPass"; }
};
} // namespace engine
} // namespace polyfhe