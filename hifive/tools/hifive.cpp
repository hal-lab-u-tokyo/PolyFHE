#include <boost/program_options.hpp>

#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.hpp"
#include "hifive/engine/codegen/codegen_manager.hpp"
#include "hifive/engine/codegen/cuda_codegen.hpp"
#include "hifive/engine/pass/calculate_memory_traffic_pass.hpp"
#include "hifive/engine/pass/kernel_fusion_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"
#include "hifive/frontend/parser.hpp"

struct Config {
    std::string input_file;
    hifive::core::GraphType type;
};

Config define_and_parse_arguments(int argc, char** argv) {
    Config config;
    boost::program_options::options_description desc("Hifive Options");
    desc.add_options()("help,h", "Print help message")(
        "input,i", boost::program_options::value<std::string>(),
        "Input dot file");

    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(1);
    }

    if (!vm.count("input")) {
        LOG_ERROR("Input file is required\n");
        exit(1);
    }
    config.input_file = vm["input"].as<std::string>();

    // Set Graph Type
    // If `input` argument contains `fhe`, then the graph is
    // core::GraphType::FHE Else if it contains `poly`, then the graph is
    // core::GraphType::Poly Otherwise, the graph is core::GraphType::Other
    if (config.input_file.find("fhe") != std::string::npos) {
        config.type = hifive::core::GraphType::FHE;
    } else if (config.input_file.find("poly") != std::string::npos) {
        config.type = hifive::core::GraphType::Poly;
    } else {
        config.type = hifive::core::GraphType::Other;
    }

    return config;
}

int main(int argc, char** argv) {
    Config config = define_and_parse_arguments(argc, argv);

    std::shared_ptr<hifive::core::Graph> graph =
        hifive::frontend::ParseDotToGraph(config.input_file, config.type);

    // Register Pass
    hifive::engine::PassManager pass_manager;

    // Memory Traffic of original graph
    pass_manager.push_back(
        std::make_shared<hifive::engine::CalculateMemoryTrafficPass>());
    // Kernel Fusion
    // pass_manager.push_back(
    //    std::make_shared<hifive::engine::KernelFusionPass>());
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