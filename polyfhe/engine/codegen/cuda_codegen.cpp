#include "polyfhe/engine/codegen/cuda_codegen.hpp"

#include <cassert>
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
        op = out + " = xxx_multiply_and_barrett_reduce_uint64(" + in0 + ", " +
             in1 + ", params->qVec[batch_idx], " +
             "params->modulus_const_ratio + batch_idx * 2);\n";
    }
    return op;
}

void CudaCodegen::generate_ElemWiseOp(
    std::shared_ptr<polyfhe::core::Node>& node, CodeWriter& w,
    std::vector<std::shared_ptr<polyfhe::core::Edge>> out_vec,
    std::shared_ptr<polyfhe::core::Edge> in0,
    std::shared_ptr<polyfhe::core::Edge> in1,
    polyfhe::core::SubgraphType s_type) {
    auto op_type = node->get_op_type();

    if (s_type == polyfhe::core::SubgraphType::ElemLimb1) {
        // TODO: support multiple output
        assert(out_vec.size() == 1);
        std::vector<std::string> args;
        for (auto edge : {out_vec[0], in0, in1}) {
            args.push_back(gen_edge_access(
                edge, "batch_idx * params->N + n_idx",
                std::to_string(edge->get_offset_smem()) + " + threadIdx.x"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);

        args.clear();
        for (auto edge : {out_vec[0], in0, in1}) {
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
        w << "uint64_t res;\n";

        std::vector<std::string> in_strs;
        // inedge
        for (auto in : {in0, in1}) {
            // TODO: use Register in data_reuse_pass.cpp
            if (in->get_level() == polyfhe::core::EdgeLevel::Shared) {
                in->set_level(polyfhe::core::EdgeLevel::Register);
            }
            in_strs.push_back(gen_edge_access(in, "idx + l", "l"));
        }
        w << gen_ElemWiseOp_internal(op_type, "res", in_strs[0], in_strs[1]);

        // outedge
        std::vector<std::shared_ptr<core::Edge>> g_outedges;
        for (auto out : out_vec) {
            // TODO: use Register in data_reuse_pass.cpp
            if (out->get_level() == core::EdgeLevel::Shared) {
                out->set_level(polyfhe::core::EdgeLevel::Register);
                w << gen_edge_access(out, "idx + l", "l") << " = res;\n";
            } else if (out->get_level() == core::EdgeLevel::Global) {
                g_outedges.push_back(out);
            }
        }
        w.block_end();

        // store to global edge using st.cs.v2
        w << "#pragma unroll\n";
        w << "for (int l = 0; l < 4; l++)";
        w.block_begin();
        for (auto out : g_outedges) {
            w << "asm(\"st.cs.global.v2.u64 [%0], {%1, %2};\"";
            w << " : ";
            w << " : \"l\"(" << out->get_name() << " + idx + 2 * l),";
            w << "\"l\"(reg[2 * l]),";
            w << "\"l\"(reg[2 * l + 1]));";
            w << "\n";
        }
        w.block_end();
    } else if (s_type == polyfhe::core::SubgraphType::ElemLimb2Slot) {
        assert(out_vec.size() == 1);
        std::vector<std::string> args;
        for (auto edge : {out_vec[0], in0, in1}) {
            args.push_back(gen_edge_access(
                edge,
                "batch_idx * params->N + blockIdx.x + thread_idx * params->n1",
                "batch_idx * params->n2 + " +
                    std::to_string(edge->get_offset_smem()) + " + thread_idx"));
        }
        w << gen_ElemWiseOp_internal(op_type, args[0], args[1], args[2]);

        args.clear();
        for (auto edge : {out_vec[0], in0, in1}) {
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
            w << "d_poly_inplace_inwt_radix8_phase1(in, params, "
                 "start_limb, shared, reg, i);\n";
        } else {
            w << "d_poly_inwt_radix8_phase2(params, "
                 "start_limb, shared, reg, tid);\n";
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
        if (subgraph->get_subgraph_type() == core::SubgraphType::NoAccess) {
            // We don't need to generate kernel for NoAccess subgraph
            continue;
        } else if (subgraph->get_subgraph_type() == core::SubgraphType::L2) {
            continue;
        }

        w << "// Define kernel for subgraph[" << subgraph->get_idx() << "]";
        w << ", type: " << core::to_string(subgraph->get_subgraph_type())
          << "\n";
        w << "__global__ void " << subgraph->get_name() << "(Params *params";

        bool has_limb_range_arg = false;
        if (subgraph->get_start_limb() != -1 &&
            subgraph->get_end_limb() != -1) {
            w << ", int start_limb, int end_limb";
            has_limb_range_arg = true;
        }

        for (auto node : subgraph->get_nodes()) {
            // Input edges
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                }
            }
            // Output edges
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    w << ", uint64_t *" << edge->get_name();
                }
            }
            // Pre-computed constants
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::MultConst) {
                w << ", uint64_t *partQlHatInv_mod_Ql_concat";
                w << ", uint64_t *partQlHatInv_mod_Ql_concat_shoup";
            } else if (op_type == core::OpType::BConv) {
                w << ", const uint64_t *qiHat_mod_pj";
                w << ", const DModulus *ibase";
                w << ", uint64_t ibase_size";
                w << ", const DModulus *obase";
                w << ", uint64_t obase_size";
                w << ", size_t startPartIdx";
                w << ", size_t size_PartQl";
            } else if (op_type == core::OpType::NTTPhase1) {
                w << ", const uint64_t *twiddles";
                w << ", const uint64_t *twiddles_shoup";
                w << ", const DModulus *modulus";
            } else if (op_type == core::OpType::NTTPhase2) {
                w << ", const uint64_t *twiddles";
                w << ", const uint64_t *twiddles_shoup";
                w << ", const DModulus *modulus";
            } else if (op_type == core::OpType::MultKeyAccum) {
                w << ", uint64_t **relin_keys";
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
            CodeWriter w_head, w_body, w_tail;
            if (subgraph->get_nodes().size() > 1) {
                w_head << "extern __shared__ uint64_t shared[];\n";
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

            if (!has_limb_range_arg) {
                w_head << "const int start_limb = "
                       << node->get_in_edges()[0]->get_start_limb() << ";\n";
                w_head << "const int end_limb = "
                       << node->get_in_edges()[0]->get_end_limb() << ";\n";
            }

            w_head << "for (int idx = threadIdx.x + blockIdx.x * "
                      "blockDim.x;";
            w_head << "idx < params->N * (end_limb - start_limb);";
            w_head << "idx += blockDim.x * gridDim.x)";
            w_head.block_begin();

            w_head << "const int l_idx = idx / params->N + start_limb;\n";

            bool has_defined_q = false;
            for (auto node : subgraph->get_nodes()) {
                std::cout << "node: " << node->get_op_name() << std::endl;
                core::OpType op_type = node->get_op_type();

                if (op_type != core::OpType::MultKeyAccum) {
                    if (!has_defined_q) {
                        w_head << "const uint64_t q = params->qVec[l_idx];\n";
                        w_head << "uint64_t res;\n";
                        w_head << "const int n_idx = l_idx * params->N + idx % "
                                  "params->N;\n";
                        has_defined_q = true;
                    }
                }

                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_in_edges().size() == 2);
                    assert(node->get_out_edges().size() >= 1);

                    w_body << "// " << node->get_op_name() << "\n";
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
                    w_body << "res = ";

                    if (op_type == core::OpType::Mult) {
                        w_body << "xxx_multiply_and_barrett_reduce_uint64(";
                    }

                    // in0
                    auto edge = node->get_in_edges()[0];
                    w_body << gen_edge_access(
                        edge, "n_idx",
                        std::to_string(edge->get_offset_smem()) +
                            " + threadIdx.x");

                    w_body << op;

                    // in1
                    edge = node->get_in_edges()[1];
                    w_body << gen_edge_access(
                        edge, "n_idx",
                        std::to_string(edge->get_offset_smem()) +
                            " + threadIdx.x");

                    if (op_type == core::OpType::Mult) {
                        w_body << ", q, "
                                  "params->modulus_const_ratio "
                                  "+ l_idx * 2);\n ";
                    } else {
                        w_body << ";\n";
                    }

                    // out
                    if (op_type == core::OpType::Add ||
                        op_type == core::OpType::Sub) {
                        w_body << "if (res >= q) res -= q;\n";
                    }

                    for (auto out_edge : node->get_out_edges()) {
                        w_body << gen_edge_access(
                            out_edge, "n_idx",
                            std::to_string(out_edge->get_offset_smem()) +
                                " + threadIdx.x");
                        w_body << " = res;\n";
                    }
                } else if (op_type == core::OpType::MultConst) {
                    assert(node->get_in_edges().size() == 1);
                    auto inedge = node->get_in_edges()[0];
                    w_body << "res = xxx_multiply_and_reduce_shoup(";
                    w_body << gen_edge_access(
                        inedge, "n_idx",
                        std::to_string(inedge->get_offset_smem()) +
                            " + threadIdx.x");
                    w_body << ", partQlHatInv_mod_Ql_concat[l_idx]";
                    w_body << ", partQlHatInv_mod_Ql_concat_shoup[l_idx]";
                    w_body << ", q);\n";
                    std::shared_ptr<core::Edge> g_store_to = nullptr;
                    for (auto outedge : node->get_out_edges()) {
                        assert(outedge->get_level() ==
                               polyfhe::core::EdgeLevel::Global);
                        if (g_store_to == nullptr) {
                            w_body << gen_edge_access(
                                outedge, "n_idx",
                                std::to_string(outedge->get_offset_smem()) +
                                    " + threadIdx.x");
                            w_body << " = res;\n";
                            g_store_to = outedge;
                        } else {
                            outedge->set_same_edge(g_store_to);
                        }
                    }
                } else if (op_type == core::OpType::MultKeyAccum) {
                    w_body << "// " << node->get_op_name() << "\n";
                    w_body << "uint64_t *in_list[" << node->get_beta()
                           << "] = {";
                    std::cout << "beta: " << node->get_beta() << std::endl;
                    assert(node->get_in_edges().size() == node->get_beta());
                    for (size_t i = 0; i < node->get_beta(); i++) {
                        w_body << node->get_in_edges()[i]->get_name();
                        if (i != node->get_beta() - 1) {
                            w_body << ", ";
                        } else {
                            w_body << "};\n";
                        }
                    }
                    assert(node->get_out_edges().size() == 2);
                    w_body << "MulKeyAccumOp(params,"
                           << node->get_out_edges()[0]->get_name() << ", "
                           << node->get_out_edges()[1]->get_name()
                           << ", in_list"
                           << ", relin_keys"
                           << ", " << node->get_beta() << ", idx"
                           << ", l_idx);\n";
                } else {
                    LOG_ERROR(
                        "Only Add, Sub, Mult, MultConst and Copy are "
                        "supported "
                        "for SubgraphType::Elem\n");
                    std::cerr << "op_type: " << core::to_str(op_type)
                              << std::endl;
                }
            }
            w_head << w_body;
            w_head.block_end();
            w << w_head;
            w << w_tail;
        } else if (s_type == core::SubgraphType::ElemLimb1) {
            // ==============================
            // ElemLimb1
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "uint64_t reg[8];\n";

            auto first_node = subgraph->get_nodes()[0];
            auto inedge = subgraph->get_nodes()[0]->get_in_edges()[0];

            int start_limb = first_node->get_start_limb();
            int end_limb = first_node->get_end_limb();
            for (auto node : subgraph->get_nodes()) {
                if (node->get_start_limb() != start_limb) {
                    LOG_ERROR("start limb doesn't match: %s\n",
                              node->get_op_name().c_str());
                }
                if (node->get_end_limb() != end_limb) {
                    LOG_ERROR("end limb doesn't match: %s\n",
                              node->get_op_name().c_str());
                }
            }
            if (!has_limb_range_arg) {
                w << "const int start_limb = " << start_limb << ";\n";
                w << "const int end_limb = " << end_limb << ";\n";
            }

            w << "uint64_t *in = " << inedge->get_name() << ";\n";
            w << "for (size_t i = blockIdx.x * blockDim.x + "
                 "threadIdx.x; ";
            w << "i < (params->N / 8 * (end_limb - start_limb)); ";
            w << "i += blockDim.x * gridDim.x)";
            w.block_begin();
            w << "const size_t n_twr = params->N / 8;\n";
            w << "const size_t n_idx = i % n_twr;\n";
            w << "const size_t twr_idx = i / n_twr + start_limb;\n";
            w << "const size_t group = params->n1 / 8;\n";
            w << "const size_t pad_tid = threadIdx.x % params->pad;\n";
            w << "const size_t pad_idx = threadIdx.x / params->pad;\n";
            w << "const size_t n_init = n_twr / group * pad_idx + pad_tid "
                 "+ "
                 "params->pad * (n_idx / (group * params->pad));\n";
            w << "uint64_t *out;\n";
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
                } else if (op_type == core::OpType::MultConst) {
                    w << "#pragma unroll\n";
                    w << "for (int l = 0; l < 8; l++)";
                    w.block_begin();
                    w << "reg[l] = "
                         "xxx_multiply_and_reduce_shoup(";
                    w << "reg[l]";
                    w << ", partQlHatInv_mod_Ql_concat[twr_idx]";
                    w << ", partQlHatInv_mod_Ql_concat_shoup[twr_idx]";
                    w << ", params->qVec[twr_idx]);\n";
                    w.block_end();
                    std::shared_ptr<core::Edge> g_store_to = nullptr;
                    for (auto outedge : node->get_out_edges()) {
                        if (outedge->get_level() == core::EdgeLevel::Global) {
                            if (g_store_to == nullptr) {
                                w << "\n// Store data from register\n";
                                w << "const size_t idx_out = twr_idx * "
                                     "params->N + "
                                     "n_init;\n";
                                w << "out = " << outedge->get_name() << ";\n";
                                w << "#pragma unroll\n";
                                w << "for (int l = 0; l < 8; l++) {\n";
                                w << "    *(out + idx_out + n_twr * l) = "
                                     "reg[l];\n";
                                w << "}\n";
                                w << "__syncthreads();\n";
                                g_store_to = outedge;
                            } else {
                                outedge->set_same_edge(g_store_to);
                            }
                        }
                    }
                } else if (op_type == core::OpType::NTTPhase1) {
                    assert(node->get_in_edges().size() == 1);
                    assert(node->get_out_edges().size() == 1);
                    auto inedge = node->get_in_edges()[0];
                    auto outedge = node->get_out_edges()[0];
                    assert(outedge->get_level() == core::EdgeLevel::Global);

                    if (inedge->get_level() == core::EdgeLevel::Global) {
                        const int exclude_start = node->get_exclude_start_idx();
                        const int exclude_end = node->get_exclude_end_idx();
                        w << "const size_t exclude_end = " << exclude_end
                          << ";\n";
                        if (exclude_start != 0) {
                            w << "const size_t exclude_start = "
                              << exclude_start << ";\n";
                            w << "if (twr_idx >= exclude_start && twr_idx < "
                                 "exclude_end)";
                        } else {
                            w << "if (twr_idx < exclude_end)";
                        }
                        w.block_begin();
                        w << "continue;\n";
                        w.block_end();

                        // TODO: merge with other operations
                        w << "\n// Load to register\n";
                        w << "#pragma unroll\n";
                        w << "for (int l = 0; l < 8; l++)";
                        w.block_begin();
                        w << "reg[l] = *(in + twr_idx * params->N + n_init + "
                             "n_twr * l);\n";
                        w.block_end();

                        w << "const uint64_t size_P = params->K;\n";
                        w << "const uint64_t size_QP = params->KL;\n";
                        w << "out = " << outedge->get_name() << ";\n";
                        w << "size_t twr_idx2 = "
                          << "(twr_idx >= start_limb + end_limb - size_P "
                          << "? size_QP - (start_limb + end_limb - twr_idx)"
                          << " : twr_idx);\n";
                        w << "d_poly_fnwt_phase1(";
                        w << "params";
                        w << ", out";
                        w << ", shared";
                        w << ", reg";
                        w << ", twiddles";
                        w << ", twiddles_shoup";
                        w << ", modulus";
                        w << ", twr_idx";
                        w << ", twr_idx2";
                        w << ", n_init";
                        w << ", i);\n";
                    }
                } else if (op_type == core::OpType::iNTTPhase1) {
                    generate_NTT_ElemLimb(node, w, false, true);
                    std::shared_ptr<core::Edge> g_store_to = nullptr;
                    for (auto outedge : node->get_out_edges()) {
                        if (outedge->get_level() == core::EdgeLevel::Global) {
                            if (g_store_to != nullptr) {
                                outedge->set_same_edge(g_store_to);
                            } else {
                                w << "\n// Store data from register\n";
                                w << "const size_t idx_out = twr_idx * "
                                     "params->N + "
                                     "n_init;\n";
                                w << "out = " << outedge->get_name() << ";\n";
                                w << "#pragma unroll\n";
                                w << "for (int l = 0; l < 8; l++) {\n";
                                w << "    *(out + idx_out + n_twr * l) = "
                                     "reg[l];\n";
                                w << "}\n";
                                w << "__syncthreads();\n";
                                g_store_to = outedge;
                            }
                        }
                    }
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase1 and iNTTPhase1 "
                        "are "
                        "supported for SubgraphType::ElemLimb1\n");
                    std::cerr << "op_type: " << core::to_str(op_type)
                              << std::endl;
                }
            }

            w.block_end();
        } else if (s_type == core::SubgraphType::ElemLimb2) {
            // ==============================
            // ElemLimb2
            // ==============================
            CodeWriter w_head, w_body, w_tail;
            w_head << "extern __shared__ uint64_t shared[];\n";
            w_head << "uint64_t reg[8];\n";

            auto first_node = subgraph->get_nodes()[0];
            auto last_node =
                subgraph->get_nodes()[subgraph->get_nodes().size() - 1];
            auto inedge = first_node->get_in_edges()[0];
            auto outedge = last_node->get_out_edges()[0];

            bool requires_load_to_reg = true;
            bool requires_store_from_reg = true;
            if (first_node->get_op_type() == core::OpType::NTTPhase2) {
                requires_load_to_reg = false;
            }

            int start_limb = first_node->get_start_limb();
            int end_limb = first_node->get_end_limb();
            for (auto node : subgraph->get_nodes()) {
                if (node->get_start_limb() != start_limb) {
                    LOG_ERROR("start limb doesn't match: %s\n",
                              node->get_op_name().c_str());
                }
                if (node->get_end_limb() != end_limb) {
                    LOG_ERROR("end limb doesn't match: %s\n",
                              node->get_op_name().c_str());
                }
            }

            if (!has_limb_range_arg) {
                w_head << "const int start_limb = " << start_limb << ";\n";
                w_head << "const int end_limb = " << end_limb << ";\n";
            }
            w_head << "const size_t n_tower = params->N / 8;\n";

            w_head
                << "for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; "
                << "tid < (end_limb - start_limb) * n_tower; "
                << "tid += blockDim.x * gridDim.x)";
            w_head.block_begin();

            if (requires_load_to_reg) {
                w_head << "\n// Load data to register\n";
                w_head << "const int twr_idx = tid / params->N + start_limb;\n";
                w_head << "uint64_t *in = " << inedge->get_name()
                       << " + twr_idx * params->N;\n";
                w_head << "#pragma unroll\n";
                w_head << "for (int l = 0; l < 8; l++)";
                w_head.block_begin();
                w_head << "reg[l] = *(in + blockIdx.x * blockDim.x * 8 + "
                          "threadIdx.x * 8 + l);\n";
                w_head.block_end();
                w_head << "__syncthreads();\n";
            }

            for (auto node : subgraph->get_nodes()) {
                w_body << "\n// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    assert(node->get_in_edges().size() == 2);
                    w_head << "size_t batch_idx = tid / n_tower;\n";
                    generate_ElemWiseOp(node, w_body, node->get_out_edges(),
                                        node->get_in_edges()[0],
                                        node->get_in_edges()[1], s_type);
                } else if (op_type == core::OpType::MultKeyAccum) {
                    w_body << "// MultKeyAccum\n";
                } else if (op_type == core::OpType::NTTPhase2) {
                    assert(node->get_in_edges().size() == 1);
                    auto inedge = node->get_in_edges()[0];
                    assert(inedge->get_level() == core::EdgeLevel::Global);
                    w_body << "size_t twr_idx = end_limb - 1 - (tid / "
                              "n_tower) + start_limb;\n";

                    const int exclude_start = node->get_exclude_start_idx();
                    const int exclude_end = node->get_exclude_end_idx();
                    w_body << "const size_t exclude_end = " << exclude_end
                           << ";\n";
                    if (exclude_start != 0) {
                        w_body
                            << "const size_t exclude_start = " << exclude_start
                            << ";\n";
                        w_body << "if (twr_idx >= exclude_start && twr_idx < "
                                  "exclude_end){continue;}\n";
                    } else {
                        w_body << "if (twr_idx < exclude_end) {continue;}\n";
                    }

                    w_body << "uint64_t n_init;\n";
                    w_body << "d_poly_fnwt_phase2(";
                    w_body << "params, " << inedge->get_name()
                           << ", shared, reg, twiddles,"
                           << "twiddles_shoup, modulus, end_limb,"
                           << "start_limb, twr_idx, &n_init, tid);\n";

                    // define store here
                    std::shared_ptr<core::Edge> g_store_to = nullptr;
                    for (auto outedge : node->get_out_edges()) {
                        if (outedge->get_level() == core::EdgeLevel::Global) {
                            if (g_store_to == nullptr) {
                                g_store_to = outedge;
                            } else {
                                outedge->set_same_edge(g_store_to);
                            }
                        }
                    }
                    if (g_store_to) {
                        w_body << "uint64_t *out_ptr = " << outedge->get_name()
                               << " + twr_idx * params->N;\n";
                        w_body << "#pragma unroll\n";
                        w_body << "for (size_t j = 0; j < 8; j++)";
                        w_body.block_begin();
                        w_body << "*(out_ptr + n_init + params->n2 / 8 * j) = "
                                  "reg[j];\n";
                        w_body.block_end();
                        w_body << "__syncthreads();\n";
                    }
                    requires_store_from_reg = false;

                } else if (op_type == core::OpType::iNTTPhase2) {
                    w_body << "d_poly_inwt_radix8_phase2(params, "
                           << node->get_start_limb()
                           << ", shared, reg, tid);\n";
                } else {
                    LOG_ERROR(
                        "Only ElementWiseOp, NTTPhase2 and iNTTPhase2 are "
                        "supported for SubgraphType::ElemLimb2\n");
                    std::cerr << "op_type: " << core::to_str(op_type)
                              << std::endl;
                }
            }

            if (requires_store_from_reg) {
                w_body << "\n// Store data from register\n";
                w_body << "const size_t n_group = params->n2 / 8;\n";
                w_body << "const size_t idx_base = start_limb * params->N + "
                          "blockIdx.x * blockDim.x * "
                          "params->per_thread_ntt_size "
                          "+ (threadIdx.x / n_group) * n_group * "
                          "params->per_thread_ntt_size "
                          "+ (threadIdx.x % n_group);\n";
                w_body << "uint64_t *out = " << outedge->get_name() << ";\n";
                w_body << "#pragma unroll\n";
                w_body << "for (int l = 0; l < 8; l++)";
                w_body.block_begin();
                w_body << "*(out + idx_base + n_group * l) = reg[l];\n";
                w_body.block_end();
                w_body << "__syncthreads();\n";
            }
            w_head << w_body;
            w_head.block_end(); // for
            w << w_head;
            w << w_tail;
        } else if (s_type == polyfhe::core::SubgraphType::ElemSlot) {
            // ==============================
            // ElemSlot
            // ==============================
            w << "extern __shared__ uint64_t shared[];\n";
            w << "for (size_t i = threadIdx.x; ";
            w << "i < obase_size * ibase_size; ";
            w << "i += blockDim.x)";
            w.block_begin();
            w << "shared[i] = qiHat_mod_pj[i];\n";
            w.block_end();
            w << "__syncthreads();\n";
            // TODO: out node
            const int n_subnodes = subgraph->get_nodes().size();
            w << "uint64_t *out = "
              << subgraph->get_nodes()[n_subnodes - 1]
                     ->get_out_edges()[0]
                     ->get_name()
              << ";\n";

            w << "const int unroll_number = 2;\n";
            w << "for (size_t tid = blockIdx.x * blockDim.x + "
                 "threadIdx.x; ";
            w << "tid < (params->N * obase_size + unroll_number - 1) / "
                 "unroll_number; ";
            w << "tid += blockDim.x * gridDim.x)";
            w.block_begin();

            w << "const size_t n_idx = unroll_number * (tid / "
                 "obase_size);\n";
            w << "const size_t l_idx = tid % obase_size;\n";

            if (n_subnodes > 1) {
                // TODO: inedge of subgraph is the first one??
                auto inedge = subgraph->get_nodes()[0]->get_in_edges()[0];
                assert(inedge->get_level() == core::EdgeLevel::Global);
                w << "uint64_t *reg_ibase = ";
                w << "shared + (obase_size + threadIdx.x * 2) * "
                     "ibase_size;\n";
                w << "for (int i = 0; i < ibase_size; i++)";
                w.block_begin();
                w << "asm(\"ld.global.v2.u64 {%0,%1}, [%2];\"";
                w << ": \"=l\"(reg_ibase[2 * i]), \"=l\"(reg_ibase[2 * i + "
                     "1])";
                w << ": \"l\"(" << inedge->get_name();
                w << "+ params->N * (startPartIdx + i) + n_idx));\n";
                w.block_end();
                w << "\n";
            }

            w << "uint64_t res1, res2;\n";
            for (auto node : subgraph->get_nodes()) {
                w << "// " << node->get_op_name() << "\n";
                core::OpType op_type = node->get_op_type();
                if (op_type == core::OpType::Add ||
                    op_type == core::OpType::Sub ||
                    op_type == core::OpType::Mult) {
                    LOG_ERROR("Not implemented\n");
                } else if (op_type == core::OpType::ModUp) {
                    LOG_ERROR("Not implemented. USE BConv\n");
                } else if (op_type == core::OpType::ModDown) {
                    LOG_ERROR("Not implemented. USE BConv\n");
                } else if (op_type == core::OpType::BConv) {
                    assert(node->get_in_edges().size() == 1);
                    auto inedge = node->get_in_edges()[0];
                    if (inedge->get_level() == core::EdgeLevel::Global) {
                        w << "BConvOpNoReg(";
                        w << "params";
                        w << ", &res1";
                        w << ", &res2";
                        w << ", " << inedge->get_name()
                          << " + params->N * startPartIdx";
                        w << ", shared";
                        w << ", n_idx";
                        w << ", l_idx";
                        w << ", ibase";
                        w << ", ibase_size";
                        w << ", obase";
                        w << ", obase_size";
                        w << ", startPartIdx";
                        w << ", size_PartQl);\n";
                    } else {
                        w << "BConvOp(";
                        w << "params";
                        w << ", &res1";
                        w << ", &res2";
                        w << ", reg_ibase";
                        w << ", shared";
                        w << ", n_idx";
                        w << ", l_idx";
                        w << ", ibase";
                        w << ", ibase_size";
                        w << ", obase";
                        w << ", obase_size";
                        w << ", startPartIdx";
                        w << ", size_PartQl);\n";
                    }
                } else {
                    std::cerr << "op_type: " << core::to_str(op_type)
                              << std::endl;
                    LOG_ERROR(
                        "Only Add, Sub, Mult and BConV are "
                        "supported for SubgraphType::ElemSlot\n");
                }
            }
            w << "const size_t l_out_idx = ";
            w << "l_idx + ((l_idx >= startPartIdx) ? size_PartQl : 0);\n";
            w << "asm(\"st.cs.global.v2.u64 [%0], {%1, %2};\"";
            w << ":";
            w << ": \"l\"(out + l_out_idx * params->N + n_idx),";
            w << "\"l\"(res1), \"l\"(res2));";
            w << "\n";
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
                    generate_ElemWiseOp(node, w, node->get_out_edges(),
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
                    std::cerr << "op_type: " << core::to_str(op_type)
                              << std::endl;
                }
                w << "__syncthreads();\n";
            }
        } else if (s_type == polyfhe::core::SubgraphType::NoAccess) {
            // We don't need to generate kernel
        } else if (s_type == polyfhe::core::SubgraphType::L2) {
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
    w << "// const int current_limb = params_h->L;\n";
    w << "const int modup_limb = params_h->KL;\n";
    for (auto subgraph : graph->get_subgraphs()) {
        if (subgraph->get_subgraph_type() ==
            polyfhe::core::SubgraphType::NoAccess) {
            // Skip NoAccess subgraph
            continue;
        }
        core::KernelLaunchConfig kconfig = subgraph->get_kernel_launch_config();

        if (subgraph->get_subgraph_type() == core::SubgraphType::L2) {
            w << "const int limb_per = params_h->alpha * n_opt;\n";
            w << "int n_divide_ = std::ceil(1.0 * modup_limb / limb_per);\n";
            w << "for (int iter = 0; iter < n_divide_; iter++)\n";
            w.block_begin();
            w << "int start_li = iter * limb_per;\n";
            w << "int end_li = (iter + 1) * limb_per;\n";
            w << "BConv_general_part_allbeta<<<4096, 128>>>("
              << "params_d, d_bconv_in_list, d_bconv_out_list,"
              << "d_qhat_modp_list, params_h->alpha, start_li, limb_per,"
              << "params_h->alpha, beta, params_h->ntt_tables->twiddle(),"
              << "params_h->ntt_tables->twiddle_shoup(),"
              << "params_h->ntt_tables->modulus(), n_opt);\n";
            w << "NTTP1_part_allbeta<<<4096, 128, "
              << "(params_h->n1 + params_h->pad + 1) * params_h->pad * "
                 "sizeof(uint64_t)>>>("
              << "params_d, start_li, end_li, 0, modup_limb,"
              << "params_h->K, beta, "
              << "params_h->ntt_tables->twiddle(),"
              << "params_h->ntt_tables->twiddle_shoup(),"
              << "params_h->ntt_tables->modulus(), d_accum_in_list);\n";
            w << "NTTP2_MultKeyAccum_part<<<4096, 128,"
                 "8 * 128 * sizeof(uint64_t)>>>("
                 "params_d, start_li, end_li, 0, modup_limb, params_h->K, beta,"
                 "params_h->ntt_tables->twiddle(),"
                 "params_h->ntt_tables->twiddle_shoup(),"
                 "params_h->ntt_tables->modulus(), d_accum_in_list,"
                 "edge_MultKeyAccum_8_0_iNTTPhase2_12_0_d,"
                 "edge_MultKeyAccum_8_1_iNTTPhase2_9_0_d, relin_keys);\n";
            w.block_end();
            continue;
        }

        if (subgraph->get_subgraph_type() == core::SubgraphType::ElemSlot) {
            w.block_begin();
            auto bconv_op = subgraph->search_op(core::OpType::BConv, 1);
            w << "const size_t beta_idx = " << bconv_op->get_beta_idx()
              << ";\n";
            w << "const size_t startPartIdx = params_h->alpha * "
                 "beta_idx;\n";
            w << "const size_t size_PartQl = (beta_idx == beta - 1)?";
            w << "(params_h->L - params_h->alpha * (beta - 1))";
            w << ": params_h->alpha;\n";
            w << "auto &bconv_pre = "
                 "drns_tool->v_base_part_Ql_to_compl_part_QlP_conv()[beta_"
                 "idx];"
                 "\n";
            w << "auto &ibase = bconv_pre.ibase();\n";
            w << "auto &obase = bconv_pre.obase();\n";
            w << "constexpr int unroll_factor = 2;\n";
        }
        w << subgraph->get_name() << "<<<" << kconfig.grid_size << ", "
          << kconfig.block_size << ", " << kconfig.shared_mem_size << ">>>";
        w << "(params_d";

        if (subgraph->get_start_limb() != -1) {
            w << ", " << subgraph->get_start_limb() << ", "
              << subgraph->get_end_limb();
        }

        for (auto node : subgraph->get_nodes()) {
            // Input edges
            for (auto edge : node->get_in_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    // Check if the src node has branch
                    auto node_src = edge->get_src();
                    if (node_src->get_out_edges().size() > 1) {
                        w << ", " << edge->get_name() << "_d";
                    } else {
                        // No branch
                        w << ", " << edge->get_name() << "_d";
                    }
                }
            }

            // Output edges
            for (auto edge : node->get_out_edges()) {
                if (edge->get_level() == polyfhe::core::EdgeLevel::Global) {
                    w << ", " << edge->get_name() << "_d";
                }
            }

            // Pre-computed constants
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::MultConst) {
                w << ", rns_tool->partQlHatInv_mod_Ql_concat()";
                w << ", rns_tool->partQlHatInv_mod_Ql_concat_shoup()";
            } else if (op_type == core::OpType::BConv) {
                w << ", bconv_pre.QHatModp()";
                w << ", bconv_pre.ibase().base()";
                w << ", bconv_pre.ibase().size()";
                w << ", bconv_pre.obase().base()";
                w << ", bconv_pre.obase().size()";
                w << ", startPartIdx";
                w << ", size_PartQl";
            } else if (op_type == core::OpType::NTTPhase1) {
                w << ", params_h->ntt_tables->twiddle()";
                w << ", params_h->ntt_tables->twiddle_shoup()";
                w << ", params_h->ntt_tables->modulus()";
            } else if (op_type == core::OpType::NTTPhase2) {
                w << ", params_h->ntt_tables->twiddle()";
                w << ", params_h->ntt_tables->twiddle_shoup()";
                w << ", params_h->ntt_tables->modulus()";
            } else if (op_type == core::OpType::MultKeyAccum) {
                w << ", relin_keys";
            }
        }
        w << ");\n";

        if (subgraph->get_require_devicesync()) {
            LOG_INFO("subgraph %s requires devicesync %d\n",
                     subgraph->get_name().c_str(),
                     subgraph->get_require_devicesync());
            w << "checkCudaErrors(cudaDeviceSynchronize());\n";
        }

        if (subgraph->get_subgraph_type() == core::SubgraphType::ElemSlot) {
            w.block_end();
        }
    }
    w << "// Timer Stop\n";
    w << "checkCudaErrors(cudaDeviceSynchronize());\n";
    w << "auto end = std::chrono::high_resolution_clock::now();\n";
    w << "\n";
}

void define_edge(CodeWriter& w, std::shared_ptr<polyfhe::core::Edge>& edge,
                 const bool if_malloc) {
    if (edge->get_has_defined()) {
        return;
    }

    w << "// Edge: " << edge->get_src()->get_op_name() << " -> "
      << edge->get_dst()->get_op_name() << "\n";

    if (edge->get_src()->get_op_type() == polyfhe::core::OpType::Init) {
        auto dst = edge->get_dst();
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
          << "_d, " << edge->get_limb()
          << " * params_h->N * sizeof(uint64_t)));\n";
    }
    edge->set_has_defined(true);
}

void CudaCodegen::generate_entry(std::shared_ptr<polyfhe::core::Graph>& graph,
                                 const std::string& filename,
                                 const bool if_append) {
    LOG_INFO("Start Generate entry kernel\n");
    CodeWriter w;

    w << "void entry_kernel(Params *params_d, Params *params_h, "
         "PhantomContext "
         "&context, "
         "uint64_t **relin_keys, "
         "uint64_t *in0, "
         "uint64_t *in1, "
         "uint64_t *out0, uint64_t *out1, bool if_benchmark, int n_opt)";
    w.block_begin();

    // w << "const long N = params_h->N;\n";
    w << "phantom::DRNSTool *rns_tool = ";
    w << "params_h->rns_tools[1];\n";

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
    std::cout << "### Edges defined" << std::endl;
    // reversed order
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_out_edges()) {
                auto src = edge->get_src();
                if (src->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (edge->get_dst()->get_op_type() == core::OpType::End) {
                    continue;
                }
                if (edge->get_level() == core::EdgeLevel::Global) {
                    if (edge->get_same_edge()) {
                        continue;
                    }
                    if (edge->get_overwrite_edge()) {
                        auto overwrite_edge = edge->get_overwrite_edge();
                        if (overwrite_edge->get_same_edge() == nullptr) {
                            if (overwrite_edge->get_shared_counter() == 0) {
                                continue;
                            }
                        }
                    }
                    define_edge(w, edge, true);
                }
            }
        }
    }
    for (auto subgraph : graph->get_subgraphs()) {
        for (auto node : subgraph->get_nodes()) {
            for (auto edge : node->get_out_edges()) {
                auto src = edge->get_src();
                if (src->get_op_type() == core::OpType::Init) {
                    continue;
                }
                if (edge->get_dst()->get_op_type() == core::OpType::End) {
                    continue;
                }
                if (edge->get_level() == core::EdgeLevel::Global) {
                    if (edge->get_same_edge()) {
                        w << "uint64_t *" << edge->get_name();
                        w << "_d = " << edge->get_same_edge()->get_name()
                          << "_d;\n";
                        continue;
                    }
                    if (edge->get_overwrite_edge()) {
                        auto overwrite_edge = edge->get_overwrite_edge();
                        if (overwrite_edge->get_same_edge() == nullptr) {
                            if (overwrite_edge->get_shared_counter() == 0) {
                                w << "uint64_t *" << edge->get_name();
                                w << "_d = "
                                  << edge->get_overwrite_edge_final()
                                         ->get_name()
                                  << "_d;\n";
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "### Edges defined" << std::endl;

    // Warm up and Test
    w << "// =====================================\n";
    w << "std::cout << \"### Warm up and Test\" << std::endl;\n";
    w << "std::cout << \"N : \" << params_h->N << std::endl;\n";
    // w << "std::cout << \"n1: \" << params_h->n1 << std::endl;\n";
    // w << "std::cout << \"n2: \" << params_h->n2 << std::endl;\n";
    w << "std::cout << \"L : \" << params_h->L << std::endl;\n";
    w << "std::cout << \"dnum : \" << params_h->dnum << std::endl;\n";
    w << "std::cout << \"K : \" << params_h->K << std::endl;\n";
    w << "std::cout << \"KL : \" << params_h->KL << std::endl;\n";
    w << "std::cout << \"alpha : \" << params_h->alpha << std::endl;\n";
    // w << "std::cout << \"------------------------------\" <<
    // std::endl;\n";

    w << "phantom::DRNSTool *drns_tool = params_h->rns_tools[1];\n";
    w << "const int beta = std::ceil(1.0 * params_h->L / "
         "params_h->alpha);\n";
    w << "std::cout << \"beta: \" << beta << std::endl;\n";

    for (auto subgraph : graph->get_subgraphs()) {
        if (subgraph->get_subgraph_type() == core::SubgraphType::L2) {
            int n_beta = subgraph->get_beta();
            std::cout << "### subgraph beta: " << n_beta << std::endl;
            w << "// BConv input\n";
            w << "uint64_t **bconv_in_list = new uint64_t *[beta];\n";
            for (int j = 0; j < n_beta; j++) {
                w << "bconv_in_list[" << j << "] = ";
                assert(subgraph->get_nodes()[j] != nullptr);
                assert(subgraph->get_nodes()[j]->get_in_edges().size() == 1);
                assert(subgraph->get_nodes()[j]->get_in_edges()[0] != nullptr);
                w << subgraph->get_nodes()[j]->get_in_edges()[0]->get_name()
                  << "_d;\n";
            }
            w << "uint64_t **d_bconv_in_list;\n";
            w << "checkCudaErrors(cudaMalloc((void ***)&d_bconv_in_list, "
                 "beta * sizeof(uint64_t *)));\n";
            w << "checkCudaErrors(cudaMemcpy(d_bconv_in_list, "
                 "bconv_in_list, beta * sizeof(uint64_t *), "
                 "cudaMemcpyHostToDevice));\n";
            w << "\n";

            w << "// BConv output\n";
            w << "uint64_t **bconv_out_list = new uint64_t *[beta];\n";
            for (int j = 0; j < n_beta; j++) {
                w << "bconv_out_list[" << j << "] = ";
                assert(subgraph->get_nodes()[j] != nullptr);
                assert(subgraph->get_nodes()[j]->get_out_edges().size() == 1);
                assert(subgraph->get_nodes()[j]->get_out_edges()[0] != nullptr);
                w << subgraph->get_nodes()[j]->get_out_edges()[0]->get_name()
                  << "_d;\n";
            }
            w << "uint64_t **d_bconv_out_list;\n";
            w << "checkCudaErrors(cudaMalloc((void ***)&d_bconv_out_list, "
                 "beta * sizeof(uint64_t *)));\n";
            w << "checkCudaErrors(cudaMemcpy(d_bconv_out_list, "
                 "bconv_out_list, beta * sizeof(uint64_t *), "
                 "cudaMemcpyHostToDevice));\n";
            w << "\n";

            w << "// NTT inout\n";
            w << "uint64_t **ntt_in_list = new uint64_t *[beta];\n";
            for (int j = 0; j < n_beta; j++) {
                w << "ntt_in_list[" << j << "] = ";
                assert(subgraph->get_nodes()[n_beta + j] != nullptr);
                assert(
                    subgraph->get_nodes()[n_beta + j]->get_out_edges().size() ==
                    1);
                assert(subgraph->get_nodes()[n_beta + j]->get_out_edges()[0] !=
                       nullptr);
                w << subgraph->get_nodes()[n_beta + j]
                         ->get_out_edges()[0]
                         ->get_name()
                  << "_d;\n";
            }
            w << "uint64_t **d_ntt_in_list;\n";
            w << "checkCudaErrors(cudaMalloc((void ***)&d_ntt_in_list, "
                 "beta * sizeof(uint64_t *)));\n";
            w << "checkCudaErrors(cudaMemcpy(d_ntt_in_list, "
                 "ntt_in_list, beta * sizeof(uint64_t *), "
                 "cudaMemcpyHostToDevice));\n";
            w << "\n";

            w << "// Accum input\n";
            w << "uint64_t **accum_in_list = new uint64_t *[beta];\n";
            auto accum =
                subgraph->get_nodes()[subgraph->get_nodes().size() - 1];
            assert(accum->get_op_type() == core::OpType::MultKeyAccum);
            assert(accum->get_in_edges().size() == n_beta);
            for (int j = 0; j < n_beta; j++) {
                w << "accum_in_list[" << j << "] = ";
                w << accum->get_in_edges()[j]->get_name() << "_d;\n";
            }
            w << "uint64_t **d_accum_in_list;\n";
            w << "checkCudaErrors(cudaMalloc((void ***)&d_accum_in_list, "
                 "beta * sizeof(uint64_t *)));\n";
            w << "checkCudaErrors(cudaMemcpy(d_accum_in_list, "
                 "accum_in_list, beta * sizeof(uint64_t *), "
                 "cudaMemcpyHostToDevice));\n";
            w << "\n";

            w << "// qHatModp\n";
            w << "uint64_t **qhat_modp_list = new uint64_t *[beta];\n";
            w << "for (int j = 0; j < beta; j++) {\n";
            w << "qhat_modp_list[j] = "
                 "drns_tool->v_base_part_Ql_to_compl_part_QlP_conv()[j]."
                 "QHatModp();\n";
            w << "}\n";
            w << "uint64_t** d_qhat_modp_list;\n";
            w << "checkCudaErrors(cudaMalloc((void**) &d_qhat_modp_list, "
                 "sizeof(uint64_t*) * beta));\n";
            w << "checkCudaErrors(cudaMemcpy(d_qhat_modp_list, "
                 "qhat_modp_list,"
                 "sizeof(uint64_t*) * beta,"
                 "cudaMemcpyHostToDevice));\n";
            w << "\n";

            w << "// ModUp\n";
            w << "uint64_t *modup_mult = "
                 "rns_tool->partQlHatInv_mod_Ql_concat();\n";
            w << "uint64_t *modup_mult_shoup = "
                 "rns_tool->partQlHatInv_mod_Ql_concat_shoup();\n";

            w << "// ModDown\n";
            w << "const DBaseConverter moddown_converter = "
                 "drns_tool->base_P_to_Ql_conv();\n";
            w << "uint64_t *d_moddown_mult = "
                 "moddown_converter.ibase().QHatInvModq();\n";
            w << "uint64_t *d_moddown_mult_shoup = "
                 "moddown_converter.ibase().QHatInvModq_shoup();\n";
            w << "uint64_t *d_moddown_matmul = "
                 "moddown_converter.QHatModp();\n";
        }
    }
    w << "\n";

    // w << "// =====================================\n";
    // w << "// Warm up\n";
    // w << "// =====================================\n";
    // w.block_begin();
    // generate_call_kernels(graph, w);
    // w.block_end();
    // w << "\n";

    // Timer
    w << "\n";
    w << "// =====================================\n";
    w << "// Benchmark\n";
    w << "// =====================================\n";
    w << "if (if_benchmark)";
    w.block_begin();
    w << "std::cout << \"### Benchmark\" << std::endl;\n";
    w << "std::vector<double> elapsed_times;\n";
    w << "for (int i = 0; i < 100; i++)\n";
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
    w << "#include <iostream>\n";
    w << "#include <numeric>\n";
    w << "#include <vector>\n";
    w << "#include <stdio.h>\n";
    w << "#include \"polyfhe/kernel/device_context.hpp\"\n";
    w << "#include \"polyfhe/kernel/polynomial.cuh\"\n";
    w << "#include \"polyfhe/kernel/ntt.hpp\"\n";
    w << "#include \"polyfhe/kernel/ntt-phantom.hpp\"\n";
    w << "#include \"phantom-fhe/include/phantom.h\"\n";
    w << "#include \"phantom-fhe/include/uintmodmath.cuh\"\n";

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