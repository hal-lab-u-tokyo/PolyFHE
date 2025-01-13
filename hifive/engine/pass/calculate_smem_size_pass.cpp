#include "calculate_smem_size_pass.hpp"

#include <optional>

#include "hifive/core/logger.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/frontend/exporter.hpp"

namespace hifive {
namespace engine {

bool CalculateSmemSizePass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running CalculateSmemSizePass\n");

    return true;
}
} // namespace engine
} // namespace hifive