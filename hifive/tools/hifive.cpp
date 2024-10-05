#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.h"
#include "hifive/engine/codegen/codegen_manager.hpp"
#include "hifive/engine/codegen/cuda_codegen.hpp"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"

int main() {
    std::shared_ptr<hifive::core::Graph> graph =
        std::make_shared<hifive::core::Graph>();

    // Register Pass
    hifive::engine::PassManager pass_manager;
    pass_manager.push_back(
        std::make_shared<hifive::engine::KernelFusionPass>());

    // Run PassManager
    pass_manager.run_on_graph(graph);

    // Code Generation
    hifive::engine::CodegenManager codegen_manager;
    codegen_manager.push_back(
        std::make_shared<hifive::engine::CudaCodegen>());
    codegen_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}