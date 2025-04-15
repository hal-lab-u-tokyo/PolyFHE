#include "polyfhe/engine/codegen/cuda_codegen.hpp"

#include <iostream>
#include <string>

#include "polyfhe/core/logger.hpp"
#include "polyfhe/engine/codegen/codegen_writer.hpp"

namespace polyfhe {
namespace engine {

std::string generate_signature(std::vector<polyfhe::core::VariableType> types,
                               std::string suffix = "") {
    if (types.size() == 0) {
        return "";
    }

    std::string signature = "";
    for (size_t i = 0; i < types.size(); i++) {
        polyfhe::core::VariableType type = types[i];
        if (type == polyfhe::core::VariableType::U64) {
            signature += "uint64_t " + suffix + std::to_string(i);
        } else if (type == polyfhe::core::VariableType::U64_PTR) {
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

std::string GenerateNByLevel(std::shared_ptr<polyfhe::core::Edge> edge,
                             const polyfhe::core::BlockPhase phase) {
    std::string n;
    if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
        n = "params->N";
    } else {
        if (phase == polyfhe::core::BlockPhase::NTTPhase1) {
            n = "params->n1";
        } else if (phase == polyfhe::core::BlockPhase::NTTPhase2) {
            n = "params->n2";
        }
    }
    return n;
}

std::string GenerateN(const polyfhe::core::BlockPhase phase) {
    std::string n;
    if (phase == polyfhe::core::BlockPhase::NTTPhase1) {
        n = "params->n1";
    } else if (phase == polyfhe::core::BlockPhase::NTTPhase2) {
        n = "params->n2";
    }
    return n;
}

std::string gen_edge_access(std::shared_ptr<polyfhe::core::Edge> edge,
                            std::string g_idx, std::string s_idx) {
    std::string m = "";
    if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
        m = edge->get_name() + "[" + g_idx + "]";
    } else {
        m = "shared[" + s_idx + "]";
    }
    return m;
}

std::string gen_ElemWiseOp_internal(polyfhe::core::OpType op_type,
                                    std::string out, std::string in0,
                                    std::string in1) {
    std::string op = "";
    if (op_type == polyfhe::core::OpType::Add) {
        op = out + " = (" + in0 + " + " + in1 + ") % q;\n";
    } else if (op_type == polyfhe::core::OpType::Sub) {
        op = out + " = (" + in0 + " + q - " + in1 + ") % q;\n";
    } else if (op_type == polyfhe::core::OpType::Mult) {
        op = out + " = (" + in0 + " * " + in1 + ") % q;\n";
    }
    return op;
}

void CudaCodegen::generate_ElemWiseOp(
    std::shared_ptr<polyfhe::core::Node>& node, CodeWriter& w,
    std::shared_ptr<polyfhe::core::Edge> out,
    std::shared_ptr<polyfhe::core::Edge> in0,
    std::shared_ptr<polyfhe::core::Edge> in1,
    polyfhe::core::SubgraphType s_type) {
    auto op_type = node->get_op_type();
    if (op_type == core::OpType::Add) {
        w << "// Add\n";
    } else if (op_type == core::OpType::Sub) {
        w << "// Sub\n";
    } else if (op_type == core::OpType::Mult) {
        w << "// Mult\n";
    }

    if (s_type == polyfhe::core::SubgraphType::ElemLimb1) {
        std::vector<std::string> args;
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge, "batch_idx * params->N + n_idx",
                std::to_string(edge->get_offset_smem()) + " + threadIdx.x"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);

        args.clear();
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge, "batch_idx * params->N + n_idx + blockDim.x",
                std::to_string(edge->get_offset_smem()) + " + threadIdx.x + "
                                                          "blockDim.x"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);
    } else if (s_type == polyfhe::core::SubgraphType::ElemLimb2) {
        std::vector<std::string> args;
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge,
                "batch_idx * params->N + block_idx + threadIdx.x * params->n1",
                std::to_string(edge->get_offset_smem()) + " + threadIdx.x"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);

        args.clear();
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge,
                "batch_idx * params->N + block_idx + (threadIdx.x + "
                "blockDim.x) * params->n1",
                std::to_string(edge->get_offset_smem()) + " + threadIdx.x + "
                                                          "blockDim.x"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);
    } else if (s_type == polyfhe::core::SubgraphType::ElemLimb2Slot) {
        std::vector<std::string> args;
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge,
                "batch_idx * params->N + blockIdx.x + thread_idx * params->n1",
                "batch_idx * params->n2 + " +
                    std::to_string(edge->get_offset_smem()) + " + thread_idx"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);

        args.clear();
        for (auto edge : {out, in0, in1}) {
            args.push_back(gen_edge_access(
                edge,
                "batch_idx * params->N + blockIdx.x + (thread_idx + "
                "n_threads) * params->n1",
                "batch_idx * params->n2 + " +
                    std::to_string(edge->get_offset_smem()) +
                    " + thread_idx + n_threads"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);
    } else {
        LOG_ERROR("Not supported yet\n");
    }
}

void CudaCodegen::generate_NTT(std::shared_ptr<polyfhe::core::Node>& node,
                               CodeWriter& w, bool if_ntt, bool if_phase1,
                               bool has_defined) {
    // TODO: add limb range to Node?

    // IN-edge must be only one
    // OUT-edge can be multiple but not supported yet
    assert(node->get_in_edges().size() == 1);
    assert(node->get_out_edges().size() == 1);

    auto inedge = node->get_in_edges()[0];
    if (!has_defined) {
        if (if_phase1) {
            w << "const int n_threads = params->n1 / 2;\n";
        } else {
            w << "const int n_threads = params->n2 / 2;\n";
        }
        w << "const int n_group = blockDim.x / n_threads;\n";
        w << "const uint64_t thread_idx = threadIdx.x % n_threads;\n";
    }
    w << "#pragma unroll\n";
    w << "for (int batch_idx = " << inedge->get_start_limb()
      << " + threadIdx.x / n_threads; ";
    w << "batch_idx < " << inedge->get_end_limb() << "; ";
    w << "batch_idx += n_group)";
    w.block_begin();
    if (inedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        w << "uint64_t *in_i = " << inedge->get_name()
          << " + batch_idx * params->N;\n";
        if (if_phase1) {
            w << "uint64_t *shared_i = shared + batch_idx * params->n1;\n";
            w << "const uint64_t idx_base = blockIdx.x * params->n1 + "
                 "thread_idx;\n";
            w << "shared_i[thread_idx] ="
              << "in_i[idx_base];\n";
            w << "shared_i[thread_idx + n_threads] ="
              << "in_i[idx_base + n_threads];\n";
        } else {
            w << "uint64_t *shared_i = shared + batch_idx * params->n2;\n";
            w << "shared_i[thread_idx] ="
              << "in_i[blockIdx.x + thread_idx * params->n1];\n";
            w << "shared_i[thread_idx + n_threads] ="
              << "in_i[blockIdx.x + (thread_idx + n_threads) * params->n1];\n";
        }
    } else {
        if (if_phase1) {
            w << "uint64_t *shared_i = shared + batch_idx * params->n1;\n";
        } else {
            w << "uint64_t *shared_i = shared + batch_idx * params->n2;\n";
        }
    }

    if (if_ntt) {
        if (if_phase1) {
            w << "NTTPhase1BlockedInternal(shared_i, params->ntt_params, "
                 "batch_idx, thread_idx);\n";
        } else {
            w << "NTTPhase2BlockedInternal(shared_i, params->ntt_params, "
                 "batch_idx, "
                 "thread_idx);\n";
        }
    } else {
        if (if_phase1) {
            w << "iNTTPhase1BlockedInternal(shared_i, params->ntt_params, "
                 "batch_idx, thread_idx);\n";
        } else {
            w << "iNTTPhase2BlockedInternal(shared_i, params->ntt_params, "
                 "batch_idx, "
                 "thread_idx);\n";
        }
    }

    // TODO: multiple outedges
    auto outedge = node->get_out_edges()[0];
    if (outedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        w << "uint64_t *out_i = " << outedge->get_name()
          << " + batch_idx * params->N;\n";
        if (if_phase1) {
            w << "out_i[idx_base] = "
                 "shared_i[thread_idx];\n";
            w << "out_i[idx_base + n_threads] = "
                 "shared_i[thread_idx + n_threads];\n";
        } else {
            w << "out_i[blockIdx.x + thread_idx * params->n1] = "
                 "shared_i[thread_idx];\n";
            w << "out_i[blockIdx.x + (thread_idx + n_threads) * params->n1] "
                 "= "
                 "shared_i[thread_idx + n_threads];\n";
        }
    } else {
        // Do nothing for shared memory
    }
    w.block_end();
}

void CudaCodegen::generate_NTT_ElemLimb(
    std::shared_ptr<polyfhe::core::Node>& node, CodeWriter& w, bool if_ntt,
    bool if_phase1) {
    // TODO: add limb range to Node?

    // IN-edge must be only one
    // OUT-edge can be multiple but not supported yet
    assert(node->get_in_edges().size() == 1);
    assert(node->get_out_edges().size() == 1);

    auto inedge = node->get_in_edges()[0];
    w.block_begin();
    if (inedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        w << "uint64_t *in_i = " << inedge->get_name()
          << " + batch_idx * params->N;\n";
        if (if_phase1) {
            w << "shared[threadIdx.x] = in_i[n_idx];\n";
            w << "shared[threadIdx.x + blockDim.x] ="
              << "in_i[n_idx + blockDim.x];\n";
        } else {
            w << "shared[threadIdx.x] ="
              << "in_i[block_idx + threadIdx.x * params->n1];\n";
            w << "shared[threadIdx.x + blockDim.x] ="
              << "in_i[block_idx + (threadIdx.x + blockDim.x) * "
                 "params->n1];\n";
        }
    }

    if (if_ntt) {
        if (if_phase1) {
            w << "NTTPhase1Internal(shared, params->ntt_params, batch_idx, "
                 "threadIdx.x);\n";
        } else {
            w << "NTTPhase2Internal(shared, params->ntt_params, batch_idx, "
                 "threadIdx.x);\n";
        }
    } else {
        if (if_phase1) {
            w << "iNTTPhase1Internal(shared, params->ntt_params, batch_idx, "
                 "threadIdx.x);\n";
        } else {
            w << "iNTTPhase2Internal(shared, params->ntt_params, batch_idx, "
                 "threadIdx.x);\n";
        }
    }

    // TODO: multiple outedges
    auto outedge = node->get_out_edges()[0];
    if (outedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        w << "uint64_t *out_i = " << outedge->get_name()
          << " + batch_idx * params->N;\n";
        if (if_phase1) {
            w << "out_i[n_idx] = shared[threadIdx.x];\n";
            w << "out_i[n_idx + blockDim.x] = "
                 "shared[threadIdx.x + blockDim.x];\n";
        } else {
            w << "out_i[block_idx + threadIdx.x * params->n1] = "
                 "shared[threadIdx.x];\n";
            w << "out_i[block_idx + (threadIdx.x + blockDim.x) * params->n1] "
                 "= "
                 "shared[threadIdx.x + blockDim.x];\n";
        }
    } else {
        // Do nothing for shared memory
    }
    w.block_end();
}

void CudaCodegen::generate_modup(std::shared_ptr<polyfhe::core::Node>& node,
                                 CodeWriter& w, std::string sPoly_x,
                                 std::string n_gidx, std::string n_sidx) {
    assert(node->get_out_edges().size() == 1);
    assert(node->get_in_edges().size() == 1);
    auto outedge = node->get_out_edges()[0];
    auto inedge = node->get_in_edges()[0];
    std::vector<std::string> args;
    std::vector<std::string> args_if_gmem;
    args.push_back("params");
    if (outedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        args.push_back(outedge->get_name());
        args_if_gmem.push_back("1");
    } else {
        args.push_back("shared + " +
                       std::to_string(outedge->get_offset_smem()));
        args_if_gmem.push_back("0");
    }
    if (inedge->get_level() == polyfhe::core::EdgeLevel::Global) {
        args.push_back(inedge->get_name());
        args_if_gmem.push_back("1");
    } else {
        args.push_back("shared + " + std::to_string(inedge->get_offset_smem()));
        args_if_gmem.push_back("0");
    }
    args.insert(args.end(), args_if_gmem.begin(), args_if_gmem.end());
    args.push_back(sPoly_x);
    args.push_back(n_gidx);
    args.push_back(n_sidx);
    args.push_back(std::to_string(inedge->get_start_limb()));
    args.push_back(std::to_string(inedge->get_end_limb()));
    w << "ModUpOp(" << GenerateArgs(args) << ");\n";
}

void CudaCodegen::generate_kernel_defs(
    std::shared_ptr<polyfhe::core::Graph>& graph, const std::string& filename,
    const bool if_append) {
    LOG_INFO("Start Generate kernel definitions\n");

    CodeWriter w;
    for (auto subgraph : graph->get_subgraphs()) {
        w << "// Define kernel for subgraph[" << subgraph->get_idx() << "]\n";
        w << "__global__ void " << subgraph->get_name() << "(Params *params";

        if (subgraph->get_block_phase() !=
            polyfhe::core::BlockPhase::NTTPhase0) {
            w << ", int nxBatch, int nyBatch";
        }

        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
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
            std::cout << "number of nodes: " << subgraph->get_nodes().size()
                      << std::endl;
            auto node = subgraph->get_nodes()[0];

            // Check if limb range is the same for all nodes
            int start_limb = node->get_in_edges()[0]->get_start_limb();
            int end_limb = node->get_in_edges()[0]->get_end_limb();
            for (auto node : subgraph->get_nodes()) {
                if (node->get_in_edges()[0]->get_start_limb() != start_limb ||
                    node->get_in_edges()[0]->get_end_limb() != end_limb) {
                    LOG_ERROR("Limb range must be the same for all nodes\n");
                    exit(1);
                }
            }

            w << "const int start_limb = "
              << node->get_in_edges()[0]->get_start_limb() << ";\n";
            w << "const int end_limb = "
              << node->get_in_edges()[0]->get_end_limb() << ";\n";
            w << "for (int idx = threadIdx.x + blockIdx.x * "
                 "blockDim.x;";
            w << "idx < params->N * (end_limb - start_limb);";
            w << "idx += blockDim.x * gridDim.x)";
            w.block_begin();

            w << "const int l_idx = idx / params->N + start_limb;\n";
            w << "const int n_idx = l_idx * params->N + idx % params->N;\n";
            w << "const uint64_t q = params->ntt_params->q[l_idx];\n";
            w << "uint64_t res;\n";

            for (auto node : subgraph->get_nodes()) {
                std::cout << "node: " << node->get_op_name() << std::endl;
                core::OpType op_type = node->get_op_type();
                if (op_type != core::OpType::Add &&
                    op_type != core::OpType::Sub &&
                    op_type != core::OpType::Mult) {
                    LOG_ERROR(
                        "Only Add, Sub, Mult are supported for "
                        "SubgraphType::Elem\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }

                w << "// " << node->get_op_name() << "\n";
                assert(node->get_in_edges().size() == 2);

                // Make sure all levels of outedges are the same
                for (auto edge : node->get_out_edges()) {
                    assert(edge->get_level() ==
                           node->get_out_edges()[0]->get_level());
                }

                std::string op = "";
                if (op_type == core::OpType::Add) {
                    op = " + ";
                } else if (op_type == core::OpType::Sub) {
                    op = " + q - ";
                } else if (op_type == core::OpType::Mult) {
                    op = " * ";
                }

                // Output to only first outedge
                w << "res = ";

                // in0
                auto edge = node->get_in_edges()[0];
                w << gen_edge_access(
                    edge, "n_idx",
                    std::to_string(edge->get_offset_smem()) + " + threadIdx.x");

                w << op;

                // in1
                edge = node->get_in_edges()[1];
                w << gen_edge_access(
                    edge, "n_idx",
                    std::to_string(edge->get_offset_smem()) + " + threadIdx.x");
                w << ";\n";

                // out
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub) {
                    w << "if (res >= q) res -= q;\n";
                } else if (op_type == core::OpType::Mult) {
                    w << "res = (res % q);\n";
                }

                edge = node->get_out_edges()[0];
                w << gen_edge_access(
                    edge, "n_idx",
                    std::to_string(edge->get_offset_smem()) + " + threadIdx.x");
                w << " = res;\n";
            }
            w.block_end();
        } else if (s_type == core::SubgraphType::ElemLimb1) {
            // ==============================
            // ElemLimb1
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "if (blockIdx.x < params->n2 * params->L)";
            w.block_begin();
            w << "const int batch_idx = blockIdx.x / params->n2;\n";
            w << "const int block_idx = blockIdx.x % params->n2;\n";
            w << "const int n_idx = block_idx * params->n1 + threadIdx.x;\n";
            bool if_defined_q = false;
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    if (!if_defined_q) {
                        w << "const uint64_t q = params->ntt_params->q["
                             "batch_idx];\n";
                        if_defined_q = true;
                    }
                    w.block_begin();
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    generate_ElemWiseOp(node, w, node->get_out_edges()[0],
                                        node->get_in_edges()[0],
                                        node->get_in_edges()[1], s_type);
                    w.block_end();
                } else if (op_type == core::OpType::NTTPhase1) {
                    generate_NTT_ElemLimb(node, w, true, true);
                } else if (op_type == core::OpType::iNTTPhase1) {
                    generate_NTT_ElemLimb(node, w, false, true);
                } else {
                    LOG_ERROR(
                        "Invalid Op: Only ElementWiseOp, NTTPhase1 and "
                        "iNTTPhase1 are "
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
            w << "if (blockIdx.x < params->n1 * params->L)";
            w.block_begin();
            w << "const int batch_idx = blockIdx.x / params->n1;\n";
            w << "const int block_idx = blockIdx.x % params->n1;\n";
            bool if_defined_q = false;
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    if (!if_defined_q) {
                        w << "const uint64_t q = params->ntt_params->q["
                             "batch_idx];\n";
                        if_defined_q = true;
                    }
                    w.block_begin();
                    generate_ElemWiseOp(node, w, node->get_out_edges()[0],
                                        node->get_in_edges()[0],
                                        node->get_in_edges()[1], s_type);
                    w.block_end();
                } else if (op_type == core::OpType::NTTPhase2) {
                    generate_NTT_ElemLimb(node, w, true, false);
                } else if (op_type == core::OpType::iNTTPhase2) {
                    generate_NTT_ElemLimb(node, w, false, false);
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase2 and iNTTPhase2 are "
                        "supported for SubgraphType::ElemLimb2\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
            }
            w.block_end();
        } else if (s_type == polyfhe::core::SubgraphType::ElemSlot) {
            // ==============================
            // ElemSlot
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "for (int idx = threadIdx.x + blockIdx.x * blockDim.x; ";
            w << "idx < params->N; ";
            w << "idx += blockDim.x * gridDim.x)";
            w.block_begin();
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    LOG_ERROR("Not implemented\n");
                } else if (op_type == core::OpType::ModUp) {
                    // TODO: impl
                    w << "for (int batch_idx = 0; ";
                    w << "batch_idx < params->L; ";
                    w << "batch_idx++)";
                    w.block_begin();
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 1);
                    auto outedge = node->get_out_edges()[0];
                    auto inedge = node->get_in_edges()[0];
                    w << "const int idx = blockIdx.x * blockDim.x + "
                         "threadIdx.x;\n";
                    w << outedge->get_name()
                      << "[batch_idx * params->N + idx] = "
                      << inedge->get_name()
                      << "[batch_idx * params->N + idx];\n";
                    w.block_end();
                    // generate_modup(node, w, "blockDim.x", "idx",
                    // "threadIdx.x");
                } else if (op_type == core::OpType::ModDown) {
                } else {
                    LOG_ERROR(
                        "Only Add, Sub, Mult and ModUp/Down are supported for "
                        "SubgraphType::ElemSlot\n");
                }
            }
            w.block_end();
        } else if (s_type == polyfhe::core::SubgraphType::ElemLimb1Slot) {
            // ==============================
            // ElemLimb1Slot
            // ==============================
            LOG_ERROR("Not implemented\n");
        } else if (s_type == polyfhe::core::SubgraphType::ElemLimb2Slot) {
            // ==============================
            // ElemLimb2Slot
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            bool has_defined_ntt = false;
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    // TODO: limb range
                    //
                    auto inedge = node->get_in_edges()[0];
                    w << "#pragma unroll\n";
                    w << "for (int batch_idx = " << inedge->get_start_limb()
                      << " + threadIdx.x / n_threads; ";
                    w << "batch_idx < " << inedge->get_end_limb() << "; ";
                    w << "batch_idx += n_group)";

                    w.block_begin();
                    w << "const uint64_t q = "
                         "params->ntt_params->q[batch_idx];\n";
                    generate_ElemWiseOp(node, w, node->get_out_edges()[0],
                                        node->get_in_edges()[0],
                                        node->get_in_edges()[1], s_type);
                    w.block_end();
                } else if (op_type == core::OpType::NTTPhase2) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 1);
                    generate_NTT(node, w, true, false, has_defined_ntt);
                    has_defined_ntt = true;
                } else if (op_type == core::OpType::iNTTPhase2) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 1);
                    generate_NTT(node, w, false, false, has_defined_ntt);
                    has_defined_ntt = true;
                } else if (op_type == core::OpType::ModUp) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 1);
                    /*
                    w << "for (int idx = threadIdx.x; ";
                    w << "idx < params->n2; ";
                    w << "idx += blockDim.x)";
                    w.block_begin();
                    generate_modup(node, w, "params->n2",
                                   "blockIdx.x * params->n2 + idx", "idx");
                    w.block_end();
                    */
                } else {
                    LOG_ERROR(
                        "Unsupported op for SubgraphType::ElemLimb2Slot\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
                w << "__syncthreads();\n";
            }
        } else {
            LOG_ERROR("Not implemented\n");
        }

        w.block_end();
        w << "\n\n";
    }

    w.write_to_file(filename, if_append);
}

void CudaCodegen::generate_call_kernels(
    std::shared_ptr<polyfhe::core::Graph>& graph, CodeWriter& w) {
    w << "// Call kernel\n";
    w << "// Timer start\n";
    w << "auto start = std::chrono::high_resolution_clock::now();\n";
    for (auto subgraph : graph->get_subgraphs()) {
        core::KernelLaunchConfig kconfig = subgraph->get_kernel_launch_config();
        w << subgraph->get_name() << "<<<" << kconfig.grid_size << ", "
          << kconfig.block_size << ", " << kconfig.shared_mem_size << ">>>";
        w << "(params_d";
        if (subgraph->get_block_phase() !=
            polyfhe::core::BlockPhase::NTTPhase0) {
            w << ", " << subgraph->get_nx_batch();
            w << ", " << subgraph->get_ny_batch();
        }
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    // Check if the src node has branch
                    auto node_src = edge->get_src();
                    if (node_src->get_out_edges().size() > 1) {
                        w << ", " << edge->get_name() << "_d";
                        /*
                        // Branch
                        if (node_src->get_op_type() == core::OpType::Init) {
                            w << ", " << edge->get_name() << "_d";
                        } else {
                            w << ", "
                              << node_src->get_out_edges()[0]->get_name()
                              << "_d";
                        }
                        */
                    } else {
                        // No branch
                        w << ", " << edge->get_name() << "_d";
                    }
                }
            }
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
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

void define_edge(CodeWriter& w, std::shared_ptr<polyfhe::core::Edge>& edge,
                 const bool initialize) {
    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
      << edge->get_dst()->get_op_name() << "\n";
    w << "uint64_t *" << edge->get_name() << "_h;\n";
    w << "uint64_t *" << edge->get_name() << "_d;\n";
    std::vector<std::string> args;
    w << "cudaMallocHost((void **)&";
    w << edge->get_name() << "_h,";
    w << "N * " << edge->get_limb() << " * sizeof(uint64_t));\n";
    w << "cudaMalloc((void **)&";
    w << edge->get_name() << "_d,";
    w << "N * " << edge->get_limb() << " * sizeof(uint64_t));\n";
    if (initialize) {
        w << "for (int i = 0; i < " << edge->get_limb() << "; i++) {";
        w << "for (int j = 0; j < N; j++) {";
        w << edge->get_name() << "_h[i * N + j] = j;";
        w << "}\n";
        w << "}\n";
        w << "cudaMemcpy(" << edge->get_name() << "_d, ";
        w << edge->get_name() << "_h,";
        w << "N * " << edge->get_limb() << " * sizeof(uint64_t),";
        w << "cudaMemcpyHostToDevice);\n";
    }
}

void CudaCodegen::generate_entry(std::shared_ptr<polyfhe::core::Graph>& graph,
                                 const std::string& filename,
                                 const bool if_append) {
    LOG_INFO("Start Generate entry kernel\n");
    CodeWriter w;

    w << "void entry_kernel(FHEContext &context)";
    w.block_begin();

    w << "Params *params_d = context.get_d_params();\n";
    w << "Params *params_h = context.get_h_params().get();\n";
    w << "const long N = params_h->N;\n";

    w << "\n";
    w << "// =====================================\n";
    w << "// Input arguments\n";
    w << "// =====================================\n";
    std::shared_ptr<polyfhe::core::Node> init_node = graph->get_init_node();
    if (init_node == nullptr) {
        LOG_ERROR("No init node\n");
        assert(false);
    }
    int i = 0;
    for (auto edge : init_node->get_out_edges()) {
        define_edge(w, edge, true);
        i++;
    }

    w << "\n";
    w << "// =====================================\n";
    w << "// Output arguments\n";
    w << "// =====================================\n";
    std::shared_ptr<polyfhe::core::Node> output_node = graph->get_exit_node();
    if (output_node == nullptr) {
        LOG_ERROR("No output node\n");
        assert(false);
    }
    for (auto edge : output_node->get_in_edges()) {
        define_edge(w, edge, false);
        w << "uint64_t *" << edge->get_name() << "_h_from_d;\n";
        w << "cudaMallocHost((void **)&" << edge->get_name()
          << "_h_from_d, N * " << edge->get_limb() << " * sizeof(uint64_t));\n";
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
            for (auto edge : node->get_out_edges()) {
                // TODO: treat Init as a alone subgraph
                if (edge->get_src()->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (edge->get_dst()->get_op_type() == core::OpType::End) {
                    continue;
                }
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    /*
                    // TODO: Why we need to define global edge only once?
                    if (has_global_edge) {
                        continue;
                    }
                    */
                    define_edge(w, edge, false);
                } else if (edge->get_level() ==
                           polyfhe::core::EdgeLevel::Shared) {
                    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
                      << edge->get_dst()->get_op_name() << "\n";
                    w << "uint64_t *" << edge->get_name() << "_h;\n";
                    w << "cudaMallocHost((void **)&" << edge->get_name()
                      << "_h, N * " << edge->get_limb()
                      << " * sizeof(uint64_t));\n";
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
    w << "std::cout << \"dnum : \" << params_h->dnum << std::endl;\n";
    w << "std::cout << \"K : \" << params_h->K << std::endl;\n";
    w << "std::cout << \"alpha : \" << params_h->alpha << std::endl;\n";
    w << "std::cout << \"q[0] : \" << params_h->ntt_params->q[0] << "
         "std::endl;\n";
    w << "std::cout << \"root[0] : \" << params_h->ntt_params->root[0] << "
         "std::endl;\n";
    w << "std::cout << \"------------------------------\" << std::endl;\n";

    // TODO: cudaFuncSetAttribute based on DeviceProp
    // w << "cudaDeviceProp prop;\n";
    // w << "cudaGetDeviceProperties(&prop, 0);\n";
    for (auto subgraph : graph->get_subgraphs()) {
        w << "cudaFuncSetAttribute(" << subgraph->get_name() << ", "
          << "cudaFuncAttributeMaxDynamicSharedMemorySize, "
          << subgraph->get_smem_size() << ");\n";
    }

    w.block_begin();

    w << "std::cout << \"### GPU\" << std::endl;\n";
    generate_call_kernels(graph, w);

    w << "\n";
    w << "// Call CPU\n";
    w << "std::cout << \"### CPU\" << std::endl;\n";
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::Add || op_type == core::OpType::Mult ||
                op_type == core::OpType::Sub) {
                assert(node->get_in_edges().size() == 2);
                for (auto outedge : node->get_out_edges()) {
                    w << node->get_op_type_str() << "_h";
                    w << "(params_h, ";
                    w << outedge->get_name() << "_h, ";
                    w << node->get_in_edges()[0]->get_name() << "_h, ";
                    w << node->get_in_edges()[1]->get_name() << "_h, ";
                    w << node->get_in_edges()[0]->get_start_limb() << ", ";
                    w << node->get_in_edges()[0]->get_end_limb() << ");\n";
                }
            } else if (op_type == core::OpType::NTTPhase1) {
                assert(node->get_in_edges().size() == 1);
                assert(node->get_out_edges().size() == 1);
                auto outedge = node->get_out_edges()[0];
                auto inedge = node->get_in_edges()[0];
                w << "NTTPhase1_h(params_h,";
                w << outedge->get_name() << "_h, ";
                w << inedge->get_name() << "_h, ";
                w << inedge->get_start_limb() << ", ";
                w << inedge->get_end_limb() << ");\n";
            } else if (op_type == core::OpType::NTTPhase2) {
                assert(node->get_in_edges().size() == 1);
                auto inedge = node->get_in_edges()[0];
                for (auto outedge : node->get_out_edges()) {
                    w << "NTTPhase2_h(params_h,";
                    w << outedge->get_name() << "_h, ";
                    w << inedge->get_name() << "_h, ";
                    w << inedge->get_start_limb() << ", ";
                    w << inedge->get_end_limb() << ");\n";
                }
            } else if (op_type == core::OpType::iNTTPhase2) {
                assert(node->get_out_edges().size() == 1);
                assert(node->get_in_edges().size() == 1);
                auto outedge = node->get_out_edges()[0];
                auto inedge = node->get_in_edges()[0];
                w << "iNTTPhase2_h(params_h,";
                w << outedge->get_name() << "_h, ";
                w << inedge->get_name() << "_h, ";
                w << inedge->get_start_limb() << ", ";
                w << inedge->get_end_limb() << ");\n";
            } else if (op_type == core::OpType::iNTTPhase1) {
                assert(node->get_in_edges().size() == 1);
                auto inedge = node->get_in_edges()[0];
                for (auto outedge : node->get_out_edges()) {
                    w << "iNTTPhase1_h(params_h,";
                    w << outedge->get_name() << "_h, ";
                    w << inedge->get_name() << "_h, ";
                    w << inedge->get_start_limb() << ", ";
                    w << inedge->get_end_limb() << ");\n";
                }
            } else if (op_type == core::OpType::ModUp) {
                assert(node->get_out_edges().size() == 1);
                assert(node->get_in_edges().size() == 1);
                w << node->get_op_type_str() << "_h";
                w << "(params_h, ";
                for (auto edge : node->get_out_edges()) {
                    w << edge->get_name() << "_h, ";
                }
                for (auto edge : node->get_in_edges()) {
                    w << edge->get_name() << "_h, ";
                }
                w << node->get_in_edges()[0]->get_start_limb() << ", ";
                w << node->get_in_edges()[0]->get_end_limb() << ");\n";
            }
        }
    }

    w << "\n";
    w << "// Copy back to host and check\n";
    w << "std::cout << \"### Check\" << std::endl;\n";
    w << "bool if_fail = false;\n";
    for (auto edge : output_node->get_in_edges()) {
        w << "cudaMemcpy(" << edge->get_name() << "_h_from_d, "
          << edge->get_name() << "_d, N * " << edge->get_limb()
          << "* sizeof(uint64_t), cudaMemcpyDeviceToHost);\n";
        w << "for (int i = 0; i < " << edge->get_limb() << "; i++)";
        w.block_begin();
        w << "for (int j = 0; j < N; j++)";
        w.block_begin();
        w << "if (";
        w << edge->get_name() << "_h[i * N + j] != ";
        w << edge->get_name() << "_h_from_d[i * N + j])";
        w.block_begin();
        w << "std::cout << \"Error[\" << i << \"][\" << j << \"] : \" << "
          << edge->get_name() << "_h_from_d[i * N + j] << \" vs ";
        w << "expected: \" << " << edge->get_name()
          << "_h[i * N + j] << std::endl;\n";
        w << "std::cout << \"Check failed\" << std::endl;\n";
        w << "if_fail = true;\n";
        w << "break;\n";
        w.block_end();
        w.block_end();
        w.block_end();
        w << "if (!if_fail) {std::cout << \"Check passed\" << "
             "std::endl;\n}else{std::cout << \"Check failed\" << std::endl;}\n";
    }
    w.block_end(); // warm up

    // Timer
    w << "\n";
    w << "// =====================================\n";
    w << "// Benchmark\n";
    w << "// =====================================\n";
    w << "std::cout << \"### Benchmark\" << std::endl;\n";
    w << "std::vector<double> elapsed_times;\n";
    w << "for (int i = 0; i < 10; i++)\n";
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

    w << "std::cout << \"Average time[us]: \" << "
         "std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) "
         "/ elapsed_times.size() << std::endl;\n";

    w.block_end(); // funcion end
    w.write_to_file(filename, if_append);
}

void CudaCodegen::generate_include(
    std::shared_ptr<polyfhe::core::Graph>& /*graph*/,
    const std::string& filename, const bool if_append) {
    LOG_INFO("Start Generate include\n");
    CodeWriter w;
    w << "// This file is generated by PolyFHE\n";
    w << "#include <cuda.h>\n";
    w << "#include <cuda_runtime.h>\n";
    w << "#include <chrono>\n";
    w << "#include <iostream>\n\n";
    w << "#include <numeric>\n";
    w << "#include <vector>\n";
    w << "#include <stdio.h>\n";
    w << "#include \"polyfhe/kernel/device_context.hpp\"\n\n";
    w << "#include \"polyfhe/kernel/polynomial.hpp\"\n\n";
    w << "#include \"polyfhe/kernel/ntt.hpp\"\n\n";
    w.write_to_file(filename, if_append);
}

bool CudaCodegen::run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) {
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
    w << graph->m_config->L << ",";
    w << graph->m_config->dnum << ");\n";

    w << "\n// Run the graph\n";
    w << "entry_kernel(context);\n";
    w << "\n";
    w << "std::cout << \"Finished Benchmarking...\" << std::endl;\n";

    w.block_end();
    w.write_to_file(output_filename, true);
    return true;
}
} // namespace engine
} // namespace polyfhe