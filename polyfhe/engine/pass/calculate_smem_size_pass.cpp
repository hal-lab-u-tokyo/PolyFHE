#include "calculate_smem_size_pass.hpp"

#include <optional>

#include "polyfhe/core/logger.hpp"
#include "polyfhe/engine/pass/data_reuse_pass.hpp"
#include "polyfhe/frontend/exporter.hpp"

namespace polyfhe {
namespace engine {

bool CalculateSmemSizePass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CalculateSmemSizePass\n");

    return true;
}
} // namespace engine
} // namespace polyfhe