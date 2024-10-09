#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.hpp"
#include "hifive/engine/codegen/codegen_manager.hpp"
#include "hifive/engine/codegen/cuda_codegen.hpp"
#include "hifive/engine/pass/calculate_memory_traffic_pass.hpp"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"
#include "hifive/frontend/parser.hpp"

int main(int argc, char** argv) {
    // Parse Input Dot
    if (argc < 2) {
        LOG_ERROR("Usage: %s <input_dot>\n", argv[0]);
        exit(1);
    }

    std::shared_ptr<hifive::core::Graph> graph =
        hifive::frontend::ParseDotToGraph(argv[1]);

    // Register Pass
    hifive::engine::PassManager pass_manager;

    // Memory Traffic of original graph
    pass_manager.push_back(
        std::make_shared<hifive::engine::CalculateMemoryTrafficPass>());
    // Kernel Fusion
    pass_manager.push_back(
        std::make_shared<hifive::engine::KernelFusionPass>());
    // Memory Traffic of optimized graph
    pass_manager.push_back(
        std::make_shared<hifive::engine::CalculateMemoryTrafficPass>());

    // Run PassManager
    pass_manager.run_on_graph(graph);

    // Code Generation
    hifive::engine::CodegenManager codegen_manager;
    codegen_manager.set(std::make_shared<hifive::engine::CudaCodegen>());
    codegen_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}