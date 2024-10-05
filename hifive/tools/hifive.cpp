#include "hifive/common/logger.h"
#include "hifive/core/graph/graph.hpp"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"

int main() {
    std::shared_ptr<hifive::core::Graph> graph =
        std::make_shared<hifive::core::Graph>();

    // Prepare PassManager
    hifive::engine::PassManager pass_manager;
    pass_manager.push_back(
        std::make_shared<hifive::engine::KernelFusionPass>());

    // Run PassManager
    pass_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}