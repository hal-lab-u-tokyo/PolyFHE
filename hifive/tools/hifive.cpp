#include <boost/program_options.hpp>

#include "hifive/core/graph/graph.hpp"
#include "hifive/core/logger.hpp"
#include "hifive/engine/codegen/codegen_manager.hpp"
#include "hifive/engine/codegen/cuda_codegen.hpp"
#include "hifive/engine/pass/analyze_intra_node_pass.hpp"
#include "hifive/engine/pass/calculate_memory_traffic_pass.hpp"
#include "hifive/engine/pass/data_reuse_pass.hpp"
#include "hifive/engine/pass/extract_subgraph_pass.hpp"
#include "hifive/engine/pass/lowering_ckks_to_poly_pass.hpp"
#include "hifive/engine/pass/pass_manager.hpp"
#include "hifive/engine/pass/rewrite_ntt_pass.hpp"
#include "hifive/engine/pass/set_block_phase_pass.hpp"
#include "hifive/frontend/exporter.hpp"
#include "hifive/frontend/parser.hpp"

struct Args {
    std::string input_file;
    hifive::core::GraphType type;
    bool if_not_optimize;
};

Args define_and_parse_arguments(int argc, char** argv) {
    Args args;
    boost::program_options::options_description desc("Hifive Options");
    desc.add_options()("noopt,n", "Not optimize graph")(
        "poly,p", "Input *.dot if poly graph")("help,h", "Print help message")(
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
    args.input_file = vm["input"].as<std::string>();

    args.if_not_optimize = vm.count("noopt");

    if (vm.count("poly")) {
        args.type = hifive::core::GraphType::Poly;
    } else {
        args.type = hifive::core::GraphType::FHE;
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
    hifive::Config config("config.csv");

    // Parse dot file
    std::shared_ptr<hifive::core::Graph> graph =
        hifive::frontend::ParseDotToGraph(
            args.input_file, args.type,
            std::make_shared<hifive::Config>(config));
    hifive::frontend::export_graph_to_dot(graph, "build/" + input_file_tail);

    // PassManager
    hifive::engine::PassManager pass_manager;

    // ==================================================
    //    Pass to FHE graph
    // ==================================================
    if (args.type == hifive::core::GraphType::FHE) {
        pass_manager.push_back(
            std::make_shared<hifive::engine::LoweringCKKSToPolyPass>());
    }

    // ==================================================
    // Pass to Poly graph
    // ==================================================
    // Pass: Calculate nemory traffic of original graph
    pass_manager.push_back(
        std::make_shared<hifive::engine::CalculateMemoryTrafficPass>());
    pass_manager.push_back(std::make_shared<hifive::engine::RewriteNTTPass>());
    pass_manager.push_back(
        std::make_shared<hifive::engine::SetBlockPhasePass>());

    // Pass: Data reuse
    if (args.if_not_optimize) {
        LOG_INFO("Do not optimize graph\n");
        pass_manager.push_back(
            std::make_shared<hifive::engine::ExtractSubgraphPass>());
    } else {
        LOG_INFO("Optimize graph\n");
        pass_manager.push_back(
            std::make_shared<hifive::engine::AnalyzeIntraNodePass>());
        pass_manager.push_back(
            std::make_shared<hifive::engine::DataReusePass>());
        pass_manager.push_back(
            std::make_shared<hifive::engine::CalculateMemoryTrafficPass>());
        pass_manager.push_back(
            std::make_shared<hifive::engine::ExtractSubgraphPass>());
    }

    // Run PassManager
    std::cout << "==================================================\n";
    std::cout << "    Input file: " << input_file_tail << std::endl;
    std::cout << "    Graph type: "
              << (args.type == hifive::core::GraphType::FHE ? "FHE" : "Poly")
              << std::endl;
    std::cout << "    Optimize: " << (args.if_not_optimize ? "No" : "Yes")
              << std::endl;
    pass_manager.display_passes();
    std::cout << "==================================================\n";
    pass_manager.run_on_graph(graph);

    // ==================================================
    // Code Generation
    // ==================================================
    hifive::frontend::export_graph_to_dot(graph,
                                          "build/final_" + input_file_tail);
    hifive::engine::CodegenManager codegen_manager;
    codegen_manager.set(std::make_shared<hifive::engine::CudaCodegen>());
    codegen_manager.run_on_graph(graph);

    LOG_INFO("Hifive succeeded\n");
}