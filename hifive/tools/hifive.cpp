#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.h"
#include "hifive/engine/codegen/codegen_manager.hpp"
#include "hifive/engine/codegen/cuda_codegen.hpp"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"
#include "hifive/frontend/parser.hpp"

int main(int argc, char** argv) {
    std::shared_ptr<hifive::core::Graph> graph =
        std::make_shared<hifive::core::Graph>();

    // Parse Input Dot
    if (argc < 2) {
        LOG_ERROR("Usage: %s <input_dot>\n", argv[0]);
        exit(1);
    }
    hifive::frontend::ParseDotToGraph(argv[1], graph);

    // Register Pass
    hifive::engine::PassManager pass_manager;
    pass_manager.push_back(
        std::make_shared<hifive::engine::KernelFusionPass>());

    // Run PassManager
    pass_manager.run_on_graph(graph);

    // Code Generation
    hifive::engine::CodegenManager codegen_manager;
    codegen_manager.set(std::make_shared<hifive::engine::CudaCodegen>());
    codegen_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}