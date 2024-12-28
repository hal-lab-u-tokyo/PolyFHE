#include "hifive/engine/codegen/cuda_codegen.hpp"

#include <string>

#include "hifive/core/logger.hpp"
#include "hifive/core/param.hpp"
#include "hifive/engine/codegen/codegen_writer.hpp"

namespace hifive {
namespace engine {

std::string generate_signature(std::vector<hifive::core::VariableType> types,
                               std::string suffix = "") {
    if (types.size() == 0) {
        return "";
    }

    std::string signature = "";
    for (size_t i = 0; i < types.size(); i++) {
        hifive::core::VariableType type = types[i];
        if (type == hifive::core::VariableType::U64) {
            signature += "uint64_t " + suffix + std::to_string(i);
        } else if (type == hifive::core::VariableType::U64_PTR) {
            signature += "uint64_t *" + suffix + std::to_string(i);
        }
        signature += ", ";
    }

    // Remove the last comma and space
    if (signature.size() > 0) {
        signature.pop_back();
        signature.pop_back();
    }
    return signature;
}

std::string GenerateArgs(std::vector<std::string> args) {
    std::string s = "";
    for (size_t i = 0; i < args.size(); i++) {
        s += args[i];
        s += ", ";
    }
    if (s.size() > 0) {
        s.pop_back();
        s.pop_back();
    }
    return s;
}

std::string GenerateNByLevel(std::shared_ptr<hifive::core::Edge> edge,
                             const hifive::core::BlockPhase phase) {
    std::string n;
    if (edge->get_level() == hifive::core::EdgeLevel::Global) {
        n = "params->N";
    } else {
        if (phase == hifive::core::BlockPhase::NTTPhase1) {
            n = "params->n1";
        } else if (phase == hifive::core::BlockPhase::NTTPhase2) {
            n = "params->n2";
        }
    }
    return n;
}

std::string GenerateN(const hifive::core::BlockPhase phase) {
    std::string n;
    if (phase == hifive::core::BlockPhase::NTTPhase1) {
        n = "params->n1";
    } else if (phase == hifive::core::BlockPhase::NTTPhase2) {
        n = "params->n2";
    }
    return n;
}

void CudaCodegen::generate_kernel_defs(
    std::shared_ptr<hifive::core::Graph>& graph, const std::string& filename,
    const bool if_append) {
    LOG_INFO("Start Generate kernel definitions\n");

    CodeWriter w;
    std::map<std::string, bool> kernel_defined;
    for (auto subgraph : graph->get_subgraphs()) {
        if (kernel_defined.contains(subgraph->get_name())) {
            continue;
        }
        kernel_defined[subgraph->get_name()] = true;
        w << "// Define kernel for subgraph[" << subgraph->get_idx() << "]\n";
        w << "__global__ void " << subgraph->get_name() << "(Params *params";
        w << ", int nxBatch, int nyBatch";
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                    // We need to global-output only once
                    break;
                }
            }
        }
        w << ")";
        w.block_begin();

        w << "extern __shared__ uint64_t shared[];\n";
        for (auto node : subgraph->get_nodes()) {
            w << "// " << node->get_op_name() << "\n";
            if (node->get_op_type() == core::OpType::Add) {
                // ==============================
                // Add
                // ==============================
                if (subgraph->get_block_phase() ==
                    hifive::core::BlockPhase::NTTPhase1) {
                    w << "Add_Phase1";
                } else {
                    w << "Add_Phase2";
                }
                std::vector<std::string> args;
                std::vector<std::string> args_if_gmem;
                args.push_back("params");
                assert(node->get_out_edges().size() == 1);
                assert(node->get_in_edges().size() == 2);
                for (auto edge : node->get_out_edges()) {
                    if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                        args.push_back(edge->get_name());
                        args_if_gmem.push_back("true");
                    } else {
                        args.push_back("shared");
                        args_if_gmem.push_back("false");
                    }
                }
                for (auto edge : node->get_in_edges()) {
                    if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                        args.push_back(edge->get_name());
                        args_if_gmem.push_back("true");
                    } else {
                        args.push_back("shared");
                        args_if_gmem.push_back("false");
                    }
                }
                args.push_back("nyBatch");
                args.insert(args.end(), args_if_gmem.begin(),
                            args_if_gmem.end());
                w << "(" << GenerateArgs(args) << ");\n";
            } else if (node->get_op_type() == core::OpType::Mult) {
                // ==============================
                // Mult
                // ==============================
                std::shared_ptr<hifive::core::Edge> global_output = nullptr;
                std::shared_ptr<hifive::core::Edge> shared_output = nullptr;
                std::vector<std::string> args;
                args.push_back("params");
                args.push_back(GenerateN(subgraph->get_block_phase()));
                args.push_back("params->L");
                // output
                if (node->get_out_edges().size() == 1) {
                    // When only one output
                    w << "// Mult";
                    auto outedge = node->get_out_edges()[0];
                    if (outedge->get_level() ==
                        hifive::core::EdgeLevel::Global) {
                        args.push_back(outedge->get_name());
                    } else {
                        args.push_back("shared");
                    }
                } else {
                    // When multiple outputs
                    // If there is no Global output, use Mult
                    // If there is no Shared output, use Mult
                    // If there is both Global and Shared outputs, use
                    // MultOutputTwo
                    for (auto edge : node->get_out_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            if (global_output != nullptr) {
                                continue;
                            }
                            global_output = edge;
                        } else if (edge->get_level() ==
                                   hifive::core::EdgeLevel::Shared) {
                            if (shared_output != nullptr) {
                                continue;
                            }
                            shared_output = edge;
                        }
                    }
                    if (global_output == nullptr) {
                        // Some Shared outputs but no Global output
                        w << "// Mult";
                        args.push_back("shared");
                    } else if (shared_output == nullptr) {
                        // Some Global outputs but no Shared output
                        w << "// Mult";
                        args.push_back(global_output->get_name());
                    } else {
                        // One Global output and some Shared outputs
                        // dst0: global_output
                        // dst1: shared
                        w << "// MultOutputTwo";
                        args.push_back(global_output->get_name());
                        args.push_back("shared");
                    }
                }
                // input
                assert(node->get_in_edges().size() == 2);
                for (auto edge : node->get_in_edges()) {
                    if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                        args.push_back(edge->get_name());
                    } else {
                        args.push_back("shared");
                    }
                }

                // N
                if (node->get_out_edges().size() == 1) {
                    if (node->get_out_edges()[0]->get_level() ==
                        hifive::core::EdgeLevel::Global) {
                        args.push_back("N");
                    } else {
                        args.push_back(GenerateN(subgraph->get_block_phase()));
                    }
                } else {
                    if (global_output == nullptr) {
                        // Mult with only Shared output
                        args.push_back(GenerateN(subgraph->get_block_phase()));
                    } else if (shared_output == nullptr) {
                        // Mult with only Global output
                        args.push_back("N");
                    } else {
                        // MultOutputTwo
                        // dst0: global_output
                        // dst1: shared
                        args.push_back("N");
                        args.push_back(GenerateN(subgraph->get_block_phase()));
                    }
                }
                args.push_back(GenerateNByLevel(node->get_in_edges()[0],
                                                node->get_block_phase()));
                args.push_back(GenerateNByLevel(node->get_in_edges()[1],
                                                node->get_block_phase()));
                w << "(" << GenerateArgs(args) << ");\n";
            } else if (node->get_op_type() == core::OpType::NTTPhase1) {
                // ==============================
                // NTTPhase1
                // ==============================
                std::vector<std::string> args;
                args.push_back("params");
                args.push_back("L");
                args.push_back("shared");
                // Inedge.size() and Outedge.size() should be 1
                // Outedge must be Global
                assert(node->get_in_edges().size() == 1);
                assert(node->get_out_edges().size() == 1);
                assert(node->get_out_edges()[0]->get_level() ==
                       hifive::core::EdgeLevel::Global);

                // If inedge is Global, load from global at first
                if (node->get_in_edges()[0]->get_level() ==
                    hifive::core::EdgeLevel::Global) {
                    std::vector<std::string> args_load;
                    args_load.push_back(node->get_in_edges()[0]->get_name());
                    args_load.push_back("shared");
                    args_load.push_back("params->N");
                    args_load.push_back("params->n1");
                    args_load.push_back("params->n2");
                    w << "// load_g2s_phase1(" << GenerateArgs(args_load)
                      << ");\n";
                }
                // Call NTTPhase1
                w << "// NTTPhase1Batched(" << GenerateArgs(args) << ");\n";

                // Store to global
                std::vector<std::string> args_store;
                args_store.push_back(node->get_out_edges()[0]->get_name());
                args_store.push_back("shared");
                args_store.push_back("params->N");
                args_store.push_back("params->n1");
                args_store.push_back("params->n2");
                w << "// store_s2g_phase1(" << GenerateArgs(args_store)
                  << ");\n";
            } else if (node->get_op_type() == core::OpType::NTTPhase2) {
                // ==============================
                // NTTPhase2
                // ==============================
                // Inedge.size() must be 1 and Global
                assert(node->get_in_edges().size() == 1);
                assert(node->get_in_edges()[0]->get_level() ==
                       hifive::core::EdgeLevel::Global);

                std::vector<std::string> args_load;
                args_load.push_back(node->get_in_edges()[0]->get_name());
                args_load.push_back("shared");
                args_load.push_back("params->N");
                args_load.push_back("params->n1");
                args_load.push_back("params->n2");
                w << "// load_g2s_phase2(" << GenerateArgs(args_load) << ");\n";

                // Call NTTPhase2
                std::vector<std::string> args;
                args.push_back("params");
                args.push_back("L");
                args.push_back("shared");
                w << "// NTTPhase2Batched(" << GenerateArgs(args) << ");\n";

                // Store to global if required
                for (auto edge : node->get_out_edges()) {
                    if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                        std::vector<std::string> args_store;
                        args_store.push_back(edge->get_name());
                        args_store.push_back("shared");
                        args_store.push_back("params->N");
                        args_store.push_back("params->n1");
                        args_store.push_back("params->n2");
                        w << "// store_s2g_phase2(" << GenerateArgs(args_store)
                          << ");\n";

                        // We need to global-output only once
                        break;
                    }
                }
            }
        }

        w.block_end();
        w << "\n\n";
    }

    w.write_to_file(filename, if_append);
}

void CudaCodegen::generate_entry(std::shared_ptr<hifive::core::Graph>& graph,
                                 const std::string& filename,
                                 const bool if_append) {
    LOG_INFO("Start Generate entry kernel\n");
    CodeWriter w;

    w << "void entry_kernel(FHEContext &context)";
    w.block_begin();

    w << "Params *params_d = context.get_d_params();\n";
    w << "Params *params_h = context.get_h_params().get();\n";
    w << "const long N = params_h->N;\n";
    w << "const long L = params_h->L;\n";

    w << "\n";
    w << "// =====================================\n";
    w << "// Input arguments\n";
    w << "// =====================================\n";
    std::shared_ptr<hifive::core::Node> init_node = graph->get_init_node();
    if (init_node == nullptr) {
        LOG_ERROR("No init node\n");
        assert(false);
    }
    int i = 0;
    for (auto edge : init_node->get_out_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_dst();
        w << "// Edge: " << init_node->get_op_name() << " -> "
          << e->get_op_name() << "\n";
        w << "uint64_t *" << edge->get_name() << "_h;\n";
        w << "uint64_t *" << edge->get_name() << "_d;\n";
        w << "cudaMallocHost((void **)&" << edge->get_name()
          << "_h, N * L * sizeof(uint64_t));\n";
        w << "cudaMalloc((void **)&" << edge->get_name()
          << "_d, N * L * sizeof(uint64_t));\n";
        w << "for (int i = 0; i < N * L; i++) {" << edge->get_name()
          << "_h[i] = 1;}\n";
        w << "cudaMemcpy(" << edge->get_name() << "_d, " << edge->get_name()
          << "_h, N * L * sizeof(uint64_t), cudaMemcpyHostToDevice);\n";
        i++;
    }

    w << "\n";
    w << "// =====================================\n";
    w << "// Output arguments\n";
    w << "// =====================================\n";
    std::shared_ptr<hifive::core::Node> output_node = graph->get_exit_node();
    if (output_node == nullptr) {
        LOG_ERROR("No output node\n");
        assert(false);
    }
    for (auto edge : output_node->get_in_edges()) {
        w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
          << output_node->get_op_name() << "\n";
        w << "uint64_t *" << edge->get_name() << "_d;\n";
        w << "uint64_t *" << edge->get_name() << "_h_from_d;\n";
        w << "uint64_t *" << edge->get_name() << "_h;\n";
        w << "cudaMalloc((void **)&" << edge->get_name()
          << "_d, N * L * sizeof(uint64_t));\n";
        w << "cudaMallocHost((void **)&" << edge->get_name()
          << "_h_from_d, N * L * sizeof(uint64_t));\n";
        w << "cudaMallocHost((void **)&" << edge->get_name()
          << "_h, N * L * sizeof(uint64_t));\n";
    }

    // Define Global edge of each subgraph
    w << "\n";
    w << "// =====================================\n";
    w << "// Define edges\n";
    w << "// Define global edges for GPU and CPU test\n";
    w << "// Define shared edges for CPU test\n";
    w << "// =====================================\n";
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            bool has_global_edge = false;
            for (auto edge : node->get_out_edges()) {
                // TODO: treat Init as a alone subgraph
                if (edge->get_src()->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (edge->get_dst()->get_op_type() == core::OpType::End) {
                    continue;
                }
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    if (has_global_edge) {
                        continue;
                    }
                    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
                      << edge->get_dst()->get_op_name() << "\n";
                    w << "uint64_t *" << edge->get_name() << "_d;\n";
                    w << "uint64_t *" << edge->get_name() << "_h;\n";
                    w << "cudaMalloc((void **)&" << edge->get_name()
                      << "_d, N * L * sizeof(uint64_t));\n";
                    w << "cudaMallocHost((void **)&" << edge->get_name()
                      << "_h, N * L * sizeof(uint64_t));\n";
                    has_global_edge = true;
                } else if (edge->get_level() ==
                           hifive::core::EdgeLevel::Shared) {
                    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
                      << edge->get_dst()->get_op_name() << "\n";
                    w << "uint64_t *" << edge->get_name() << "_h;\n";
                    w << "cudaMallocHost((void **)&" << edge->get_name()
                      << "_h, N * L * sizeof(uint64_t));\n";
                }
            }
        }
    }
    w << "\n";

    // Warm up and Test
    w << "// =====================================\n";
    w << "// Test\n";
    w << "// =====================================\n";
    w << "std::cout << \"### Warm up and Test\" << std::endl;\n";
    w.block_begin();
    w << "// Call kernel\n";
    w << "dim3 gridPhase1(params_h->n2);\n";
    w << "dim3 gridPhase2(params_h->n1);\n";
    w << "dim3 blockPhase1(params_h->n1 / 8);\n";
    w << "dim3 blockPhase2(params_h->n2 / 8);\n";
    w << "dim3 block256(256);\n";
    w << "const int shared_size_phase1 = params_h->n1 * params_h->L * "
         "sizeof(uint64_t);\n";
    w << "const int shared_size_phase2 = params_h->n2 * params_h->L * "
         "sizeof(uint64_t);\n";
    for (auto subgraph : graph->get_subgraphs()) {
        if (subgraph->get_block_phase() ==
            hifive::core::BlockPhase::NTTPhase1) {
            w << subgraph->get_name()
              << "<<<gridPhase1, blockPhase1, shared_size_phase1>>>";
        } else {
            w << subgraph->get_name()
              << "<<<gridPhase2, blockPhase2, shared_size_phase2>>>";
        }
        w << "(params_d";
        w << ", " << subgraph->get_nx_batch();
        w << ", " << subgraph->get_ny_batch();
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    auto same_result_edge = edge->get_same_result_edge();
                    if (same_result_edge == nullptr) {
                        LOG_ERROR("No same result global edge\n");
                        assert(false);
                    }
                    w << ", " << same_result_edge->get_name() << "_d";
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", " << edge->get_name() << "_d";
                    // We need to global-output only once
                    break;
                }
            }
        }
        w << ");\n";
    }
    w << "cudaDeviceSynchronize();\n";

    w << "\n";
    w << "// Call CPU\n";
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            w << node->get_op_type_str() << "_h";
            w << "(params_h";
            for (auto edge : node->get_out_edges()) {
                w << ", " << edge->get_name() << "_h";
            }
            for (auto edge : node->get_in_edges()) {
                w << ", " << edge->get_name() << "_h";
            }
            w << ");\n";
        }
    }

    w << "\n";
    w << "// Copy back to host and check\n";
    for (auto edge : output_node->get_in_edges()) {
        w << "cudaMemcpy(" << edge->get_name() << "_h_from_d, "
          << edge->get_name()
          << "_d, N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost);\n";
        w << "for (int i = 0; i < N * L; i++)";
        w.block_begin();
        w << "if (";
        w << edge->get_name() << "_h[i] != ";
        w << edge->get_name() << "_h_from_d[i])";
        w.block_begin();
        w << "std::cout << \"Error[\" << i << \"] ";
        w << "result: \" << " << edge->get_name() << "_h_from_d[i] << \" vs ";
        w << "expected: \" << " << edge->get_name() << "_h[i] << std::endl;\n";
        w << "std::cout << \"Check failed\" << std::endl;\n";
        w << "return;\n";
        w.block_end();
        w.block_end();
        w << "std::cout << \"Check passed\" << std::endl;\n";
    }
    w.block_end(); // warm up

    // Timer
    w << "\n";
    w << "// =====================================\n";
    w << "// Benchmark\n";
    w << "// =====================================\n";
    w << "std::cout << \"### Benchmark\" << std::endl;\n";
    w << "std::vector<double> elapsed_times;\n";
    w << "for (int i = 0; i < 5; i++)\n";
    w.block_begin();
    w << "// Timer start\n";
    w << "auto start = std::chrono::high_resolution_clock::now();\n";

    w << "\n";
    w << "// Call kernels\n";
    w << "dim3 gridPhase1(params_h->n2);\n";
    w << "dim3 gridPhase2(params_h->n1);\n";
    w << "dim3 blockPhase1(params_h->n2 / 8);\n";
    w << "dim3 blockPhase2(params_h->n2 / 8);\n";
    w << "const int shared_size_phase1 = params_h->n1 * params_h->L * "
         "sizeof(uint64_t);\n";
    w << "const int shared_size_phase2 = params_h->n2 * params_h->L * "
         "sizeof(uint64_t);\n";
    for (auto subgraph : graph->get_subgraphs()) {
        if (subgraph->get_block_phase() ==
            hifive::core::BlockPhase::NTTPhase1) {
            w << subgraph->get_name()
              << "<<<gridPhase1, blockPhase1, shared_size_phase1>>>";
        } else {
            w << subgraph->get_name()
              << "<<<gridPhase2, blockPhase2, shared_size_phase2>>>";
        }
        w << "(params_d";
        w << ", " << subgraph->get_nx_batch();
        w << ", " << subgraph->get_ny_batch();
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    auto same_result_edge = edge->get_same_result_edge();
                    if (same_result_edge == nullptr) {
                        LOG_ERROR("No same result global edge\n");
                        assert(false);
                    }
                    w << ", " << same_result_edge->get_name() << "_d";
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", " << edge->get_name() << "_d";
                    // We need to global-output only once
                    break;
                }
            }
        }
        w << ");\n";
    }

    // Timer
    w << "\n";
    w << "// Timer Stop\n";
    w << "checkCudaErrors(cudaDeviceSynchronize());\n";
    w << "auto end = std::chrono::high_resolution_clock::now();\n";
    w << "auto elapsed_usec = "
         "std::chrono::duration_cast<std::chrono::microseconds>(end - "
         "start);\n";
    w << "std::cout << \"Elapsed time: \" << elapsed_usec.count() << "
         "\"us\" << std::endl;\n";
    w << "if (i != 0) {elapsed_times.push_back(elapsed_usec.count());}\n";
    w.block_end(); // for end

    w << "std::cout << \"Average time: \" << "
         "std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) "
         "/ elapsed_times.size() << \"us\" << std::endl;\n";

    w.block_end(); // funcion end
    w.write_to_file(filename, if_append);
}

void CudaCodegen::generate_include(
    std::shared_ptr<hifive::core::Graph>& /*graph*/,
    const std::string& filename, const bool if_append) {
    LOG_INFO("Start Generate include\n");
    CodeWriter w;
    w << "// This file is generated by HiFive\n";
    w << "#include <cuda.h>\n";
    w << "#include <cuda_runtime.h>\n";
    w << "#include <chrono>\n";
    w << "#include <iostream>\n\n";
    w << "#include <numeric>\n";
    w << "#include \"hifive/kernel/device_context.hpp\"\n\n";
    w << "#include \"hifive/kernel/polynomial.hpp\"\n\n";
    w << "#include \"hifive/kernel/ntt.hpp\"\n\n";
    w.write_to_file(filename, if_append);
}

bool CudaCodegen::run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running CudaCodegen\n");

    std::string output_filename = "build/generated.cu";

    generate_include(graph, output_filename, /*append=*/false);
    generate_kernel_defs(graph, output_filename, /*append=*/true);
    generate_entry(graph, output_filename, /*append=*/true);

    CodeWriter w;
    w << "int main(int argc, char *argv[])";
    w.block_begin();
    w << "std::cout << \"Starting Benchmarking...\" << std::endl;\n";
    w << "FHEContext context;\n";

    w << "\n// Run the graph\n";
    w << "entry_kernel(context);\n";
    w << "\n";
    w << "std::cout << \"Finished Benchmarking...\" << std::endl;\n";

    w.block_end();
    w.write_to_file(output_filename, true);
    return true;
}
} // namespace engine
} // namespace hifive