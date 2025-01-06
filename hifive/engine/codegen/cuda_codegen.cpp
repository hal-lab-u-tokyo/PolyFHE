#include "hifive/engine/codegen/cuda_codegen.hpp"

#include <iostream>
#include <string>

#include "hifive/core/logger.hpp"
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

        if (subgraph->get_block_phase() !=
            hifive::core::BlockPhase::NTTPhase0) {
            w << ", int nxBatch, int nyBatch";
        }

        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == hifive::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                    // TODO: why?
                    // We need to global-output only once
                    // break;
                }
            }
        }
        w << ")";
        w.block_begin();

        core::SubgraphType s_type = subgraph->get_subgraph_type();
        std::cout << "Subgraph type: " << s_type << std::endl;
        if (s_type == core::SubgraphType::Elem) {
            // ==============================
            // Elem
            // ==============================
            if (subgraph->get_nodes().size() > 1) {
                w << "extern __shared__ uint64_t shared[];\n";
            }
            w << "for (int idx = threadIdx.x + blockIdx.x * blockDim.x;";
            w << "idx < params->N * params->limb;";
            w << "idx += blockDim.x * gridDim.x)";
            w.block_begin();
            w << "const int l_idx = idx / params->N;\n";
            w << "const int n_idx = idx % params->N;\n";
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    std::vector<std::string> args;
                    std::vector<std::string> args_if_gmem;
                    if (op_type == core::OpType::Add) {
                        args.push_back("ElemWiseOp::Add");
                    } else if (op_type == core::OpType::Sub) {
                        args.push_back("ElemWiseOp::Sub");
                    } else if (op_type == core::OpType::Mult) {
                        args.push_back("ElemWiseOp::Mult");
                    }
                    args.push_back("params");
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    for (auto edge : node->get_out_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    for (auto edge : node->get_in_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    args.insert(args.end(), args_if_gmem.begin(),
                                args_if_gmem.end());
                    args.push_back("blockDim.x");
                    args.push_back("l_idx");
                    args.push_back("n_idx");
                    args.push_back("threadIdx.x");
                    w << "ElemWiseOp_Elem(" << GenerateArgs(args) << ");\n";
                } else {
                    LOG_ERROR(
                        "Only Add, Sub, Mult are supported for "
                        "SubgraphType::Elem\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
            }
            w.block_end();
        } else if (s_type == core::SubgraphType::ElemLimb1) {
            // ==============================
            // ElemLimb1
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "for (int idx = blockIdx.x;";
            w << "idx < params->n2 * params->limb;";
            w << "idx += gridDim.x)";
            w.block_begin();
            bool defined_l_idx = false;
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    std::vector<std::string> args;
                    std::vector<std::string> args_if_gmem;
                    if (op_type == core::OpType::Add) {
                        args.push_back("ElemWiseOp::Add");
                    } else if (op_type == core::OpType::Sub) {
                        args.push_back("ElemWiseOp::Sub");
                    } else if (op_type == core::OpType::Mult) {
                        args.push_back("ElemWiseOp::Mult");
                    }
                    args.push_back("params");
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    for (auto edge : node->get_out_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    for (auto edge : node->get_in_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    args.insert(args.end(), args_if_gmem.begin(),
                                args_if_gmem.end());
                    args.push_back("params->n2");
                    args.push_back("l_idx");
                    args.push_back("n_gidx");
                    args.push_back("threadIdx.x + i * params->n1 / 8");

                    if (!defined_l_idx) {
                        w << "const int l_idx = idx / params->n2;\n";
                        w << "const int n_idx = idx % params->n2;\n";
                        w << "int n_gidx = n_idx + threadIdx.x * params->n1;\n";
                        defined_l_idx = true;
                    }
                    w << "for (int i = 0; i < 8; i++)";
                    w.block_begin();
                    w << "ElemWiseOp_Elem(" << GenerateArgs(args) << ");\n";
                    w << "n_gidx += params->n1 * blockDim.x;\n";
                    w.block_end();
                } else if (op_type == core::OpType::NTTPhase1) {
                    // If inedge is Global, load from global at first
                    if (node->get_in_edges()[0]->get_level() ==
                        hifive::core::EdgeLevel::Global) {
                        std::vector<std::string> args_load;
                        args_load.push_back(
                            node->get_in_edges()[0]->get_name());
                        args_load.push_back("shared");
                        args_load.push_back("params->N");
                        args_load.push_back("params->n1");
                        args_load.push_back("params->n2");
                        w << "load_g2s_phase1(" << GenerateArgs(args_load)
                          << ");\n";
                    }
                    // Call NTTPhase1
                    std::vector<std::string> args;
                    args.push_back("shared");
                    args.push_back("params->ntt_params");
                    args.push_back("idx/params->n2");
                    args.push_back("threadIdx.x");
                    w << "NTTPhase1Op(" << GenerateArgs(args) << ");\n";
                    // Outedge must be Global
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_out_edges()[0]->get_level() ==
                           hifive::core::EdgeLevel::Global);
                    // Store to global
                    std::vector<std::string> args_store;
                    args_store.push_back(node->get_out_edges()[0]->get_name());
                    args_store.push_back("shared");
                    args_store.push_back("params->N");
                    args_store.push_back("params->n1");
                    args_store.push_back("params->n2");
                    w << "store_s2g_phase1(" << GenerateArgs(args_store)
                      << ");\n";

                } else if (op_type == core::OpType::iNTTPhase1) {
                    LOG_ERROR("Not implemented\n");
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase1 and iNTTPhase1 are "
                        "supported for SubgraphType::ElemLimb1\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
            }
            w.block_end();
        } else if (s_type == core::SubgraphType::ElemLimb2) {
            // ==============================
            // ElemLimb2
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "for (int idx = blockIdx.x;";
            w << "idx < params->n1 * params->limb;";
            w << "idx += gridDim.x)";
            w.block_begin();
            bool defined_l_idx = false;
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    std::vector<std::string> args;
                    std::vector<std::string> args_if_gmem;
                    if (op_type == core::OpType::Add) {
                        args.push_back("ElemWiseOp::Add");
                    } else if (op_type == core::OpType::Sub) {
                        args.push_back("ElemWiseOp::Sub");
                    } else if (op_type == core::OpType::Mult) {
                        args.push_back("ElemWiseOp::Mult");
                    }
                    args.push_back("params");
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    for (auto edge : node->get_out_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    for (auto edge : node->get_in_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            args.push_back(edge->get_name());
                            args_if_gmem.push_back("1");
                        } else {
                            args.push_back("shared");
                            args_if_gmem.push_back("0");
                        }
                    }
                    args.insert(args.end(), args_if_gmem.begin(),
                                args_if_gmem.end());
                    args.push_back("params->n2");
                    args.push_back("l_idx");
                    args.push_back("n_idx");
                    args.push_back("threadIdx.x + i * params->n2 / 8");

                    if (!defined_l_idx) {
                        w << "const int l_idx = idx / params->n1;\n";
                        w << "int n_idx = (idx % params->n1) * params->n2 + "
                             "threadIdx.x;\n";
                        defined_l_idx = true;
                    }
                    w << "for (int i = 0; i < 8; i++)";
                    w.block_begin();
                    w << "ElemWiseOp_Elem(" << GenerateArgs(args) << ");\n";
                    w << "n_idx += params->n2/8;\n";
                    w.block_end();
                } else if (op_type == core::OpType::NTTPhase2) {
                    // Inedge must be Global
                    assert(node->get_in_edges().size() == 1);
                    assert(node->get_in_edges()[0]->get_level() ==
                           hifive::core::EdgeLevel::Global);
                    std::vector<std::string> args_load;
                    args_load.push_back(node->get_in_edges()[0]->get_name());
                    args_load.push_back("shared");
                    args_load.push_back("params->N");
                    args_load.push_back("params->n1");
                    args_load.push_back("params->n2");
                    w << "load_g2s_phase2(" << GenerateArgs(args_load)
                      << ");\n";

                    // Call NTTPhase2
                    std::vector<std::string> args;
                    args.push_back("shared");
                    args.push_back("params->ntt_params");
                    args.push_back("idx/params->n1");
                    args.push_back("threadIdx.x");
                    args.push_back("tid % (params->N / 8)");
                    w << "size_t tid = blockIdx.x * blockDim.x + "
                         "threadIdx.x;\n";
                    w << "NTTPhase2Op(" << GenerateArgs(args) << ");\n";

                    // Store to global if required
                    for (auto edge : node->get_out_edges()) {
                        if (edge->get_level() ==
                            hifive::core::EdgeLevel::Global) {
                            std::vector<std::string> args_store;
                            args_store.push_back(edge->get_name());
                            args_store.push_back("shared");
                            args_store.push_back("params->N");
                            args_store.push_back("params->n1");
                            args_store.push_back("params->n2");
                            w << "store_s2g_phase2(" << GenerateArgs(args_store)
                              << ");\n";
                        }
                    }
                } else if (op_type == core::OpType::iNTTPhase2) {
                    LOG_ERROR("Not implemented\n");
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase2 and iNTTPhase2 are "
                        "supported for SubgraphType::ElemLimb1\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
            }
            w.block_end();
        } else {
            LOG_ERROR("Not implemented\n");
        }
        /*
        for (auto node : subgraph->get_nodes()) {
            w << "// " << node->get_op_name() << "\n";
            if (node->get_op_type() == core::OpType::Add) {
            } else if (node->get_op_type() == core::OpType::Mult) {
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
        */

        w.block_end();
        w << "\n\n";
    }

    w.write_to_file(filename, if_append);
}

void CudaCodegen::generate_call_kernels(
    std::shared_ptr<hifive::core::Graph>& graph, CodeWriter& w) {
    w << "// Call kernel\n";
    // w << "dim3 gridPhase1(params_h->n2);\n";
    // w << "dim3 gridPhase2(params_h->n1);\n";
    // w << "dim3 gridCommon(2048);\n";
    // w << "dim3 blockPhase1(params_h->n1 / 8);\n";
    // w << "dim3 blockPhase2(params_h->n2 / 8);\n";
    // w << "dim3 blockCommon(128);\n";
    // w << "const int shared_size_phase1 = params_h->n1 * params_h->L * "
    //      "sizeof(uint64_t);\n";
    // w << "const int shared_size_phase2 = params_h->n2 * params_h->L * "
    //      "sizeof(uint64_t);\n";
    // w << "const int shared_size_common = 128 * sizeof(uint64_t);\n";

    w << "// Timer start\n";
    w << "auto start = std::chrono::high_resolution_clock::now();\n";
    for (auto subgraph : graph->get_subgraphs()) {
        core::SubgraphType s_type = subgraph->get_subgraph_type();
        if (s_type == core::SubgraphType::Elem) {
            // If subgraph contains only ONE node, we don't need to malloc
            // shared memory
            if (subgraph->get_nodes().size() == 1) {
                w << subgraph->get_name() << "<<<2048, 128>>>";
            } else {
                w << subgraph->get_name()
                  << "<<<2048, 128, 128 * sizeof(uint64_t)>>>";
            }
        } else if (s_type == core::SubgraphType::ElemLimb1) {
            // NTTPhase1 uses shared memory even if it contains only one node
            w << subgraph->get_name()
              << "<<<params_h->n2 * params_h->limb, params_h->n1/8, "
                 "params_h->n1 * sizeof(uint64_t)>>>";

        } else if (s_type == core::SubgraphType::ElemLimb2) {
            // NTTPhase2 uses shared memory even if it contains only one node
            w << subgraph->get_name()
              << "<<<params_h->n1 * params_h->limb, params_h->n2/8, "
                 "params_h->n2 * sizeof(uint64_t)>>>";
        } else {
            LOG_ERROR("Not implemented\n");
        }
        w << "(params_d";
        if (subgraph->get_block_phase() !=
            hifive::core::BlockPhase::NTTPhase0) {
            w << ", " << subgraph->get_nx_batch();
            w << ", " << subgraph->get_ny_batch();
        }
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
                    // TODO: why?
                    // We need to global-output only once
                    // break;
                }
            }
        }
        w << ");\n";
        w << "checkCudaErrors(cudaDeviceSynchronize());\n";
    }
    w << "// Timer Stop\n";
    w << "auto end = std::chrono::high_resolution_clock::now();\n";
    w << "\n";
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
          << "_h[i] = i % 10;}\n";
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
                    /*
                    // TODO: Why we need to define global edge only once?
                    if (has_global_edge) {
                        continue;
                    }
                    */
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
    w << "std::cout << \"N : \" << params_h->N << std::endl;\n";
    w << "std::cout << \"n1: \" << params_h->n1 << std::endl;\n";
    w << "std::cout << \"n2: \" << params_h->n2 << std::endl;\n";
    w << "std::cout << \"L : \" << params_h->L << std::endl;\n";
    w << "std::cout << \"q[0] : \" << params_h->ntt_params->q[0] << "
         "std::endl;\n";
    w << "std::cout << \"root[0] : \" << params_h->ntt_params->root[0] << "
         "std::endl;\n";
    w << "std::cout << \"------------------------------\" << std::endl;\n";
    w.block_begin();
    generate_call_kernels(graph, w);

    w << "\n";
    w << "// Call CPU\n";
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::NTTPhase1) {
                assert(node->get_out_edges().size() == 1);
                auto phase2_node = node->get_out_edges()[0]->get_dst();
                for (auto outedge : phase2_node->get_out_edges()) {
                    w << "NTT_h(params_h,";
                    w << outedge->get_name() << "_h, ";
                    w << node->get_in_edges()[0]->get_name() << "_h);\n";
                }
                continue;
            } else if (op_type == core::OpType::NTTPhase2) {
                continue;
            } else if (op_type == core::OpType::iNTTPhase2) {
                assert(node->get_out_edges().size() == 1);
                auto phase1_node = node->get_out_edges()[0]->get_dst();
                for (auto outedge : phase1_node->get_out_edges()) {
                    w << "iNTT_h(params_h,";
                    w << outedge->get_name() << "_h, ";
                    w << node->get_in_edges()[0]->get_name() << "_h);\n";
                }
                continue;
            } else if (op_type == core::OpType::iNTTPhase1) {
                continue;
            }
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
    w << "bool if_fail = false;\n";
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
        w << "if_fail = true;\n";
        w << "break;\n";
        w.block_end();
        w.block_end();
        w << "if (!if_fail) {std::cout << \"Check passed\" << std::endl;}\n";
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

    w << "\n";
    generate_call_kernels(graph, w);

    // Timer
    w << "\n";
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
    w << "#include <vector>\n";
    w << "#include <stdio.h>\n";
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
    w << "FHEContext context(";
    w << graph->m_config->logN << ", ";
    w << graph->m_config->L << ");\n";

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