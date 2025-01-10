#pragma once

#include "hifive/engine/pass/pass_base.hpp"

namespace hifive {
namespace engine {
class CalculateSmemSizePass : public PassBase {
public:
    bool run_on_graph(std::shared_ptr<hifive::core::Graph>& /*graph*/) override;
    std::string get_name() override { return "CalculateSmemSizePass"; }
};
} // namespace engine
} // namespace hifive