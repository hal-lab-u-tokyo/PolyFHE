#include "hifive/core/logger.h"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"

namespace hifive {
namespace engine {
bool KernelFusionPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running KernelFusionPass\n");
    return true;
}
} // namespace engine
} // namespace hifive