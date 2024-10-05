#pragma once

#include <hifive/engine/pass/pass_base.hpp>
#include <memory>
#include <vector>

namespace hifive {
namespace engine {
class PassManager : public std::vector<std::shared_ptr<PassBase>> {
public:
    bool run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
        for (auto& pass : *this) {
            if (!pass->run_on_graph(graph)) {
                return false;
            }
        }
        return true;
    }
};
} // namespace engine
} // namespace hifive