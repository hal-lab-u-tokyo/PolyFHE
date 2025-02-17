#pragma once

#include <iostream>
#include <polyfhe/engine/pass/pass_base.hpp>

namespace polyfhe {
namespace engine {
class PassManager : public std::vector<std::shared_ptr<PassBase>> {
public:
    bool run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) {
        for (auto& pass : *this) {
            if (!pass->run_on_graph(graph)) {
                return false;
            }
        }
        return true;
    }

    void display_passes() {
        std::cout << "Passes:" << std::endl;
        for (auto& pass : *this) {
            std::cout << "    - " << pass->get_name() << std::endl;
        }
    }
};
} // namespace engine
} // namespace polyfhe