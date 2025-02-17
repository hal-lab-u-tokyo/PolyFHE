#pragma once

#include <memory>
#include <vector>

#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/engine/codegen/codegen_base.hpp"

namespace polyfhe {
namespace engine {
class CodegenManager : public std::shared_ptr<CodegenBase> {
public:
    void set(std::shared_ptr<CodegenBase> pass_codegen) {
        this->pass_codegen = pass_codegen;
    }

    bool run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) {
        if (pass_codegen == nullptr) {
            LOG_ERROR("Codegen is not set\n");
            return false;
        }
        if (pass_codegen->run_on_graph(graph) == false) {
            LOG_ERROR("Codegen failed\n");
            return false;
        }
        return true;
    }

private:
    std::shared_ptr<CodegenBase> pass_codegen;
};
} // namespace engine
} // namespace polyfhe