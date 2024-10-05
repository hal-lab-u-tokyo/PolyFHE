#pragma once

#include <memory>
#include <vector>

#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.h"
#include "hifive/engine/codegen/codegen_base.hpp"

namespace hifive {
namespace engine {
class CodegenManager : public std::vector<std::shared_ptr<CodegenBase>> {
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