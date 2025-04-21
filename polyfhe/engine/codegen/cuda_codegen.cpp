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
    } else if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
        m = "shared[" + s_idx + "]";
    } else if (edge->get_level() == polyfhe::core::EdgeLevel::Register) {
        // TODO: prepare r_idx
        m = "reg[" + s_idx + "]";
    } else {
        LOG_ERROR("Not supported yet\n");
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
        op = out + " = multiply_and_barrett_reduce_uint64(" + in0 + ", " + in1 +
             ", params->qVec[batch_idx], " +
             "params->modulus_const_ratio + batch_idx * 2);\n";
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
        w << "const size_t idx = blockIdx.x * blockDim.x * 8 + threadIdx.x * "
             "8;\n";
        w << "#pragma unroll\n";
        w << "for (int l = 0; l < 8; l++)";
        w.block_begin();
        std::vector<std::string> args;
        for (auto edge : {out, in0, in1}) {
            // TODO: use Register in data_reuse_pass.cpp
            if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
                edge->set_level(polyfhe::core::EdgeLevel::Register);
            }
            args.push_back(gen_edge_access(edge, "idx + l", "l"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);
        w.block_end();
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

    if (if_ntt) {
        LOG_ERROR("Not implemented;\n");
        assert(false);
        if (if_phase1) {
        } else {
        }
    } else {
        if (if_phase1) {
            w << "d_poly_inplace_inwt_radix8_phase1(in, params, params->L, 0, "
                 "shared, reg, i);\n";
        } else {
            w << "d_poly_inwt_radix8_phase2(params, params->L, 0, "
                 "shared, reg, i);\n";
        }
    }
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
            w << "const uint64_t q = params->qVec[l_idx];\n";
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
                    op = " , ";
                }

                // Output to only first outedge
                w << "res = ";

                if (op_type == core::OpType::Mult) {
                    w << "multiply_and_barrett_reduce_uint64(";
                }

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

                if (op_type == core::OpType::Mult) {
                    w << ", params->qVec[l_idx], params->modulus_const_ratio "
                         "+ l_idx * 2);\n ";
                } else {
                    w << ";\n";
                }

                // out
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub) {
                    w << "if (res >= q) res -= q;\n";
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
            w << "uint64_t reg[8];\n";
            const int n_nodes = subgraph->get_nodes().size();
            auto inedge = subgraph->get_nodes()[0]->get_in_edges()[0];
            auto outedge =
                subgraph->get_nodes()[n_nodes - 1]->get_out_edges()[0];
            assert(core::is_ntt_op(inedge->get_src()->get_op_type()));
            w << "uint64_t *in = " << inedge->get_name() << ";\n";
            w << "uint64_t *out = " << outedge->get_name() << ";\n";
            w << "for (size_t i = blockIdx.x * blockDim.x + "
                 "threadIdx.x; ";
            w << "i < (params->N / 8 * params->L); ";
            w << "i += blockDim.x * gridDim.x)";
            w.block_begin();
            w << "const size_t n_twr = params->N / 8;\n";
            w << "const size_t n_idx = i % n_twr;\n";
            w << "const size_t twr_idx = i / n_twr;\n";
            w << "const size_t group = params->n1 / 8;\n";
            w << "const size_t pad_tid = threadIdx.x % params->pad;\n";
            w << "const size_t pad_idx = threadIdx.x / params->pad;\n";
            w << "const size_t n_init = n_twr / group * pad_idx + pad_tid + "
                 "params->pad * (n_idx / (group * params->pad));\n";
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    LOG_ERROR("Not implemented\n");
                    assert(false);
                } else if (op_type == core::OpType::NTTPhase1) {
                    LOG_ERROR("Not implemented\n");
                    assert(false);
                    // generate_NTT_ElemLimb(node, w, true, true);
                } else if (op_type == core::OpType::iNTTPhase1) {
                    generate_NTT_ElemLimb(node, w, false, true);
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase1 and iNTTPhase1 "
                        "are "
                        "supported for SubgraphType::ElemLimb1\n");
                    std::cerr << "op_type: " << core::toStringOpType(op_type)
                              << std::endl;
                }
            }
            w << "\n// Store data from register\n";
            w << "const size_t idx = twr_idx * params->N + n_init;\n";
            w << "#pragma unroll\n";
            w << "for (int l = 0; l < 8; l++) {\n";
            w << "    *(out + idx + n_twr * l) = reg[l];\n";
            w << "}\n";
            w << "__syncthreads();\n";
            w.block_end();
        } else if (s_type == core::SubgraphType::ElemLimb2) {
            // ==============================
            // ElemLimb2
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "uint64_t reg[8];\n";
            w << "uint64_t *in = "
              << subgraph->get_nodes()[0]->get_in_edges()[0]->get_name()
              << ";\n";
            // TODO: out node
            const int n_subnodes = subgraph->get_nodes().size();
            w << "uint64_t *out = "
              << subgraph->get_nodes()[n_subnodes - 1]
                     ->get_out_edges()[0]
                     ->get_name()
              << ";\n";

            w << "for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < "
                 "params->L * params->N / 8; i += blockDim.x * gridDim.x)";
            w.block_begin();
            w << "size_t batch_idx = i / (params->N / 8);\n";
            // TODO: use i
            // 今の実装だとthreadが十分にあるときしか正しくうごかない
            w << "\n// Load data to register\n";
            w << "#pragma unroll\n";
            w << "for (int l = 0; l < 8; l++) {\n";
            w << "    reg[l] = *(in + blockIdx.x * blockDim.x * 8 + "
                 "threadIdx.x * 8 + l);\n";
            w << "}\n";
            w << "__syncthreads();\n";
            for (auto node : subgraph->get_nodes()) {
                w << "\n// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_out_edges().size() == 1);
                    assert(node->get_in_edges().size() == 2);
                    generate_ElemWiseOp(node, w, node->get_out_edges()[0],
                                        node->get_in_edges()[0],
                                        node->get_in_edges()[1], s_type);
                } else if (op_type == core::OpType::NTTPhase2) {
                    LOG_ERROR("Not implemented\n");
                    assert(false);
                    // generate_NTT_ElemLimb(node, w, true, false);
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
            w << "\n// Store data from register\n";
            w << "const size_t n_group = params->n2 / 8;\n";
            w << "#pragma unroll\n";
            w << "for (int l = 0; l < 8; l++) {\n";
            w << "    size_t idx = blockIdx.x * blockDim.x * "
                 "params->per_thread_ntt_size "
                 "+ (threadIdx.x / n_group) * n_group * "
                 "params->per_thread_ntt_size "
                 "+ (threadIdx.x % n_group) + n_group * l;\n";
            w << "    *(out + idx) = reg[l];\n";
            w << "}\n";
            w << "__syncthreads();\n";
            w.block_end(); // for
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
                 const bool if_malloc) {
    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
      << edge->get_dst()->get_op_name() << "\n";

    if (edge->get_src()->get_op_type() == polyfhe::core::OpType::Init) {
        w << "uint64_t *" << edge->get_name() << "_d = in"
          << edge->get_idx_argc() << " + " << edge->get_offset() << ";\n";
    } else if (edge->get_dst()->get_op_type() == polyfhe::core::OpType::End) {
        w << "uint64_t *" << edge->get_name() << "_d = out"
          << edge->get_idx_argc() << " + " << edge->get_offset() << ";\n";
    } else {
        w << "uint64_t *" << edge->get_name() << "_d;\n";
    }
    if (if_malloc) {
        w << "checkCudaErrors(cudaMalloc((void**)&" << edge->get_name()
          << "_d, " << edge->get_limb() << " * params_h->N"
          << "* sizeof(uint64_t)));\n";
    }
}

void CudaCodegen::generate_entry(std::shared_ptr<polyfhe::core::Graph>& graph,
                                 const std::string& filename,
                                 const bool if_append) {
    LOG_INFO("Start Generate entry kernel\n");
    CodeWriter w;

    w << "void entry_kernel(Params *params_d, Params *params_h, uint64_t *in0, "
         "uint64_t *in1, "
         "uint64_t *out0, bool if_benchmark)";
    w.block_begin();

    // w << "const long N = params_h->N;\n";

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
        define_edge(w, edge, false);
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
    }
    w << "\n";
    w << "// =====================================\n";
    w << "// Edges\n";
    w << "// Define global edges for GPU\n";
    w << "// =====================================\n";
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_out_edges()) {
                if (edge->get_src()->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (edge->get_dst()->get_op_type() == core::OpType::End) {
                    continue;
                }
                if (edge->get_level() == core::EdgeLevel::Global) {
                    define_edge(w, edge, true);
                }
            }
        }
    }

    // Warm up and Test
    w << "// =====================================\n";
    w << "std::cout << \"### Warm up and Test\" << std::endl;\n";
    w << "std::cout << \"N : \" << params_h->N << std::endl;\n";
    // w << "std::cout << \"n1: \" << params_h->n1 << std::endl;\n";
    // w << "std::cout << \"n2: \" << params_h->n2 << std::endl;\n";
    w << "std::cout << \"L : \" << params_h->L << std::endl;\n";
    // w << "std::cout << \"dnum : \" << params_h->dnum << std::endl;\n";
    // w << "std::cout << \"K : \" << params_h->K << std::endl;\n";
    // w << "std::cout << \"alpha : \" << params_h->alpha << std::endl;\n";
    // w << "std::cout << \"------------------------------\" << std::endl;\n";

    // TODO: cudaFuncSetAttribute based on DeviceProp
    // w << "cudaDeviceProp prop;\n";
    // w << "cudaGetDeviceProperties(&prop, 0);\n";
    for (auto subgraph : graph->get_subgraphs()) {
        // w << "cudaFuncSetAttribute(" << subgraph->get_name() << ", "
        //   << "cudaFuncAttributeMaxDynamicSharedMemorySize, "
        //   << subgraph->get_smem_size() << ");\n";
    }
    w << "\n";

    w << "// =====================================\n";
    w << "// Warm up\n";
    w << "// =====================================\n";
    w.block_begin();
    generate_call_kernels(graph, w);
    w.block_end();
    w << "\n";

    // Timer
    w << "\n";
    w << "// =====================================\n";
    w << "// Benchmark\n";
    w << "// =====================================\n";
    w << "if (if_benchmark)";
    w.block_begin();
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

    w.block_end(); // if benchmark end
    w.block_end(); // function end
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
    w << "#include \"polyfhe/kernel/ntt-phantom.hpp\"\n\n";
    w.write_to_file(filename, if_append);
}

bool CudaCodegen::run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running CudaCodegen\n");

    std::string output_filename = "build/generated.cu";

    generate_include(graph, output_filename, /*append=*/false);
    generate_kernel_defs(graph, output_filename, /*append=*/true);
    generate_entry(graph, output_filename, /*append=*/true);

    return true;
}
} // namespace engine
} // namespace polyfhe