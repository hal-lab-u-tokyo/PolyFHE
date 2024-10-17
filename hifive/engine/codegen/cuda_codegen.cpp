#include "hifive/engine/codegen/cuda_codegen.hpp"

#include <unordered_map>

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

void CudaCodegen::generate_kernel_defs(
    std::shared_ptr<hifive::core::Graph>& graph, const std::string& filename,
    const bool if_append) {
    LOG_INFO("Start Generate kernel definitions\n");

    // Hash map to store the kernel signature
    m_cu_kernels = std::map<std::string, std::shared_ptr<CodeUnitKernel>>();

    // Iterate the graph and generate the kernel signature
    for (auto node : graph->get_nodes()) {
        if (node == nullptr) {
            continue;
        }

        std::string op_type = node->get_op_type();
        if (m_cu_kernels.contains(op_type)) {
            continue;
        }

        if ((op_type == "Init") | (op_type == "End")) {
            continue;
        }

        std::shared_ptr<CodeUnitKernel> cu = std::make_shared<CodeUnitKernel>();
        cu->op_type = node->get_op_type();
        cu->func_name = "kernel_" + node->get_op_type();
        cu->input_signature =
            generate_signature(node->get_input_types(), "in_");
        cu->output_signature =
            generate_signature(node->get_output_types(), "out_");
        m_cu_kernels[op_type] = cu;

        CodeWriter w;
        w << "// Define kernel for node: " << node->get_op_type() << "\n";
        w << "__global__ void " << cu->func_name << "(";
        w << "DeviceContext *dc, const int N, const int L";
        if (cu->input_signature.size() > 0) {
            w << ", ";
        }
        w << cu->input_signature;
        if (cu->output_signature.size() > 0) {
            w << ", ";
        }
        w << cu->output_signature;
        w << ")";

        w.block_begin();
        w << "extern __shared__ uint64_t shared[];\n";
        // TODO: consider parameters
        w << "const int block_x = 128;\n";
        w << "const int block_y = L;\n";

        const int n_inedge = node->get_in_edges().size();
        const int n_outedge = node->get_out_edges().size();
        for (int i = 0; i < n_inedge; i++) {
            w << "uint64_t *in_" << i << "i = in_" << i
              << " + blockIdx.x * block_x;\n";
        }
        for (int i = 0; i < n_outedge; i++) {
            w << "uint64_t *out_" << i << "i = out_" << i
              << " + blockIdx.x * block_x;\n";
        }

        const std::vector<std::string> ops = node->get_ops();
        int in_used = 0;
        for (size_t i = 0; i < ops.size(); i++) {
            if (ops[i] == "Add") {
                std::string out, in0, in1;
                bool if_dst_shared, if_a_shared, if_b_shared;

                if (i == 0) {
                    in0 = "in_0i";
                    in1 = "in_1i";
                    if_a_shared = false;
                    if_b_shared = false;
                    in_used = 2;
                } else {
                    in0 = "shared";
                    in1 = "in_" + std::to_string(in_used) + "i";
                    if_a_shared = true;
                    if_b_shared = false;
                    in_used++;
                }
                if (i == ops.size() - 1) {
                    out = "out_0i";
                    if_dst_shared = false;
                } else {
                    out = "shared";
                    if_dst_shared = true;
                }

                w << "Add(dc, N, block_x, block_y, " << out << ", " << in0
                  << ", " << in1 << ", " << if_dst_shared << ", " << if_a_shared
                  << ", " << if_b_shared << ");\n";
            }
        }

        w.block_end();
        w << "\n";
        w.write_to_file(filename, if_append);
    }
}

void CudaCodegen::generate_entry(std::shared_ptr<hifive::core::Graph>& graph,
                                 const std::string& filename,
                                 const bool if_append) {
    LOG_INFO("Start Generate entry kernel\n");
    CodeWriter w, w_body;

    // Signature
    std::string arg = "DeviceContext *dc, const int N, const int L";
    for (auto edge : graph->get_init_node()->get_out_edges()) {
        arg += ", uint64_t * in_" + edge->get_name();
    }
    for (auto edge : graph->get_exit_node()->get_in_edges()) {
        arg += ", uint64_t * out_" + edge->get_name();
    }

    w << "void entry_kernel(" << arg << "){\n";
    w.indent_inc();
    w_body.indent_inc();

    // Iterate the graph and generate the kernel signature
    const int n = graph->get_nodes().size();
    std::vector<bool> visited(n, false);
    std::vector<int> stack;

    stack.push_back(graph->get_init_node_id());

    w << "// Define each node's output edges\n\n";

    // Timer
    w_body << "std::vector<double> elapsed_times;\n";
    w_body << "for (int i = 0; i < 5; i++) {\n";
    w_body.indent_inc();
    w_body << "\n// Timer Start\n";
    w_body << "auto start = std::chrono::high_resolution_clock::now();\n";
    w_body << "\n// Call kernels\n";

    while (!stack.empty()) {
        int node_idx = stack.back();
        stack.pop_back();
        if (visited[node_idx]) {
            continue;
        }
        visited[node_idx] = true;
        auto node = graph->get_nodes()[node_idx];
        if (node == nullptr) {
            // Fused node is nullptr
            continue;
        }

        w << "// " << node->get_op_name() << "'s output\n";
        w_body << "// " << node->get_op_type() << "(" << node->get_op_name()
               << ")\n";

        // define output edge
        w_body << "//\t outputs: ";
        for (auto edge : node->get_out_edges()) {
            w << "uint64_t *" << edge->get_name() << ";\n";
            if (node->get_op_type() != "Init") {
                w << "checkCudaErrors(cudaMalloc((void **)&" << edge->get_name()
                  << ", sizeof(uint64_t) * " << edge->get_shape(0) << " * "
                  << edge->get_shape(1) << "));\n";
            }
            w_body << edge->get_name() << ", ";
        }
        w_body << "\n";
        w_body << "//\t inputs: ";
        for (auto edge : node->get_in_edges()) {
            w_body << edge->get_name() << ", ";
        }
        w_body << "\n";

        // Call the kernel
        if ((node->get_op_type() == "Init") | (node->get_op_type() == "End")) {
            w_body << "// Nothing to call\n\n";
        } else {
            std::string args = "dc, N, L";
            for (auto edge : node->get_in_edges()) {
                args += ", " + edge->get_name();
            }
            for (auto edge : node->get_out_edges()) {
                args += ", " + edge->get_name();
            }
            w_body << "kernel_" << node->get_op_type()
                   << "<<<N/128, 128, L*128*sizeof(uint64_t)>>>"
                   << "(" << args << ");\n";
            w_body << "checkCudaErrors(cudaDeviceSynchronize());\n\n";
        }

        // Update the stack
        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    w << "\n// Combine input edges\n";
    for (auto edge : graph->get_init_node()->get_out_edges()) {
        w << edge->get_name() << " = in_" << edge->get_name() << ";\n";
    }

    w << "\n// Combine output edges\n";
    for (auto edge : graph->get_exit_node()->get_in_edges()) {
        w << "out_" << edge->get_name() << " = " << edge->get_name() << ";\n";
    }
    w.write_to_file(filename, if_append);

    // Timer
    w_body << "// Timer Stop\n";
    w_body << "auto end = std::chrono::high_resolution_clock::now();\n";
    w_body << "auto elapsed_usec = "
              "std::chrono::duration_cast<std::chrono::microseconds>(end - "
              "start);\n";
    w_body << "std::cout << \"Elapsed time: \" << elapsed_usec.count() << "
              "\"us\" << std::endl;\n";
    w_body << "if (i != 0) {elapsed_times.push_back(elapsed_usec.count());}\n";
    w_body.indent_dec();
    w_body << "}\n";
    w_body
        << "std::cout << \"Average time: \" << "
           "std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) "
           "/ elapsed_times.size() << \"us\" << std::endl;\n";
    w_body.indent_dec();
    w_body << "}\n\n";
    w_body.write_to_file(filename, true);
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
    w << "DeviceContext dc;\n";
    w << "const int N = 65535;\n";
    w << "const int L = 20;\n\n";

    w << "// Input arguments\n";
    std::shared_ptr<hifive::core::Node> init_node = graph->get_init_node();
    int i = 0;
    for (auto edge : init_node->get_out_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_dst();
        std::string name_size = "sizeof(uint64_t) * " +
                                std::to_string(edge->get_shape(0)) + " * " +
                                std::to_string(edge->get_shape(1));
        w << "// Edge: " << init_node->get_op_name() << " -> "
          << e->get_op_name() << "\n";
        w << "uint64_t *" << edge->get_name() << "_h;\n";
        w << "uint64_t *" << edge->get_name() << "_d;\n";
        w << "cudaMallocHost((void **)&" << edge->get_name() << "_h, "
          << name_size << ");\n";
        w << "cudaMalloc((void **)&" << edge->get_name() << "_d, " << name_size
          << ");\n";
        i++;
    }

    w << "\n// Output arguments\n";
    std::shared_ptr<hifive::core::Node> exit_node = graph->get_exit_node();
    i = 0;
    for (auto edge : exit_node->get_in_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_src();
        std::string name_size = "sizeof(uint64_t) * " +
                                std::to_string(edge->get_shape(0)) + " * " +
                                std::to_string(edge->get_shape(1));
        w << "// Edge: " << e->get_op_name() << " -> "
          << exit_node->get_op_name() << "\n";
        w << "uint64_t *" << edge->get_name() << "_h;\n";
        w << "uint64_t *" << edge->get_name() << "_d;\n";
        w << "cudaMallocHost((void **)&" << edge->get_name() << "_h, "
          << name_size << ");\n";
        w << "cudaMalloc((void **)&" << edge->get_name() << "_d, " << name_size
          << ");\n";
        i++;
    }

    w << "\n// Fill input arguments\n";

    std::string input_args_str = "&dc, N, L";
    for (auto edge : init_node->get_out_edges()) {
        input_args_str += ", " + edge->get_name() + "_d";
    }
    for (auto edge : exit_node->get_in_edges()) {
        input_args_str += ", " + edge->get_name() + "_d";
    }

    w << "\n";
    w << "// Run the graph\n";
    w << "entry_kernel(" << input_args_str << ");\n";
    w << "std::cout << \"Finished Benchmarking...\" << std::endl;\n";
    w.block_end();

    w.write_to_file(output_filename, true);
    return true;
}
} // namespace engine
} // namespace hifive