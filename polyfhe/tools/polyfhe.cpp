#include <boost/program_options.hpp>

#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/core/logger.hpp"
#include "polyfhe/engine/codegen/codegen_manager.hpp"
#include "polyfhe/engine/codegen/cuda_codegen.hpp"
#include "polyfhe/engine/pass/analyze_intra_node_pass.hpp"
#include "polyfhe/engine/pass/cache_aware_reorder_pass.hpp"
#include "polyfhe/engine/pass/calculate_memory_traffic_pass.hpp"
#include "polyfhe/engine/pass/calculate_smem_size_pass.hpp"
#include "polyfhe/engine/pass/check_edge_overwrite_pass.hpp"
#include "polyfhe/engine/pass/check_edge_same_pass.hpp"
#include "polyfhe/engine/pass/check_subgraph_dependencies_pass.hpp"
#include "polyfhe/engine/pass/data_reuse_pass.hpp"
#include "polyfhe/engine/pass/extract_l2reuse_pass.hpp"
#include "polyfhe/engine/pass/extract_subgraph_pass.hpp"
#include "polyfhe/engine/pass/kernel_launch_config_pass.hpp"
#include "polyfhe/engine/pass/lowering_ckks_to_poly_pass.hpp"
#include "polyfhe/engine/pass/pass_manager.hpp"
#include "polyfhe/engine/pass/rewrite_ntt_pass.hpp"
#include "polyfhe/engine/pass/set_block_phase_pass.hpp"
#include "polyfhe/engine/pass/sort_subgraph_pass.hpp"
#include "polyfhe/frontend/exporter.hpp"
#include "polyfhe/frontend/parser.hpp"

enum class OptLevel {
    None,
    Reg,
    L2,
};
struct Args {
    std::string input_file;
    std::string config_file;
    polyfhe::core::GraphType type;
    OptLevel opt_level;
};

Args define_and_parse_arguments(int argc, char** argv) {
    Args args;
    boost::program_options::options_description desc("Hifive Options");
    desc.add_options()("optlevel", boost::program_options::value<int>(),
                       "Optimization level (0: None, 1: Reg, 2: L2)")(
        "poly,p", "Input *.dot if poly graph")("help,h", "Print help message")(
        "input,i", boost::program_options::value<std::string>(),
        "Input dot file")("config,c",
                          boost::program_options::value<std::string>(),
                          "Config file");

    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(1);
    }

    if (!vm.count("config")) {
        LOG_ERROR("Config file is required\n");
        exit(1);
    }
    args.config_file = vm["config"].as<std::string>();

    if (!vm.count("input")) {
        LOG_ERROR("Input file is required\n");
        exit(1);
    }
    args.input_file = vm["input"].as<std::string>();

    if (vm.count("poly")) {
        args.type = polyfhe::core::GraphType::Poly;
    } else {
        args.type = polyfhe::core::GraphType::FHE;
    }

    switch (vm["optlevel"].as<int>()) {
    case 0:
        args.opt_level = OptLevel::None;
        break;
    case 1:
        args.opt_level = OptLevel::Reg;
        break;
    case 2:
        args.opt_level = OptLevel::L2;
        break;
    default:
        LOG_ERROR("Invalid optimization level\n");
        exit(1);
    }

    return args;
}

int main(int argc, char** argv) {
    // Read arguments
    Args args = define_and_parse_arguments(argc, argv);
    std::string input_file_tail = args.input_file.substr(
        args.input_file.find_last_of("/") + 1, args.input_file.size());
    LOG_INFO("Input file: %s\n", input_file_tail.c_str());

    // Read Hifive config file
    polyfhe::Config config(args.config_file);

    // Parse dot file
    std::shared_ptr<polyfhe::core::Graph> graph =
        polyfhe::frontend::ParseDotToGraph(
            args.input_file, args.type,
            std::make_shared<polyfhe::Config>(config));
    polyfhe::frontend::export_graph_to_dot(graph, "build/" + input_file_tail);

    // PassManager
    polyfhe::engine::PassManager pass_manager;

    // ==================================================
    //    Pass to FHE graph
    // ==================================================
    if (args.type == polyfhe::core::GraphType::FHE) {
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::LoweringCKKSToPolyPass>());
    }

    // ==================================================
    // Pass to Poly graph
    // ==================================================
    // Pass: Calculate nemory traffic of original graph
    pass_manager.push_back(
        std::make_shared<polyfhe::engine::CalculateMemoryTrafficPass>());
    pass_manager.push_back(std::make_shared<polyfhe::engine::RewriteNTTPass>());
    pass_manager.push_back(
        std::make_shared<polyfhe::engine::SetBlockPhasePass>());

    // Pass: Data reuse
    if (args.opt_level == OptLevel::None) {
        LOG_INFO("Do not optimize graph\n");
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::ExtractSubgraphPass>());
    } else if (args.opt_level == OptLevel::Reg) {
        LOG_INFO("Optimize graph\n");
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::AnalyzeIntraNodePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::DataReusePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CalculateMemoryTrafficPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::ExtractSubgraphPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CalculateSmemSizePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CacheAwareReorderpass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckSubgraphDependenciesPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckEdgeSamePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckEdgeOverwritePass>());
    } else if (args.opt_level == OptLevel::L2) {
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::AnalyzeIntraNodePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::DataReusePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CalculateMemoryTrafficPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::ExtractSubgraphPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckSubgraphDependenciesPass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckEdgeSamePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::CheckEdgeOverwritePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::ExtractL2ReusePass>());
        pass_manager.push_back(
            std::make_shared<polyfhe::engine::SortSubgraphPass>());
    }
    pass_manager.push_back(
        std::make_shared<polyfhe::engine::KernelLaunchConfigPass>());

    // Run PassManager
    std::cout << "==================================================\n";
    std::cout << "    Input file: " << input_file_tail << std::endl;
    std::cout << "    Config file: " << args.config_file << std::endl;
    std::cout << "    Graph type: "
              << (args.type == polyfhe::core::GraphType::FHE ? "FHE" : "Poly")
              << std::endl;
    std::cout << "    Optlevel(0: None, 1: Reg, 2: L2) : "
              << (int) args.opt_level << std::endl;
    std::cout << "    config.SharedMemKB: " << config.SharedMemKB << std::endl;
    pass_manager.display_passes();
    std::cout << "==================================================\n";
    pass_manager.run_on_graph(graph);

    // ==================================================
    // Code Generation
    // ==================================================
    polyfhe::frontend::export_graph_to_dot(graph,
                                           "build/final_" + input_file_tail);
    polyfhe::engine::CodegenManager codegen_manager;
    codegen_manager.set(std::make_shared<polyfhe::engine::CudaCodegen>());
    codegen_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}