#pragma once

#include "hifive/engine/pass/pass_base.hpp"

namespace hifive {
namespace engine {
class CalculateMemoryTrafficPass : public PassBase {
public:
    bool run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) override;
};
} // namespace engine
} // namespace hifive