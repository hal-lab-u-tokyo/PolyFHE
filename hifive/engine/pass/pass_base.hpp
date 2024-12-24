#pragma once

#include <memory>

#include "hifive/core/graph/graph.hpp"

namespace hifive {
namespace engine {
class PassBase {
public:
    virtual bool run_on_graph(std::shared_ptr<hifive::core::Graph>& /*graph*/) {
        return true;
    }
    virtual std::string get_name() { return "PassBase"; }
};
} // namespace engine
} // namespace hifive