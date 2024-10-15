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
        w << "DeviceContext *dc";
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
    const int n_inedge = graph->get_init_node()->get_out_edges().size();
    const int n_outedge = graph->get_exit_node()->get_in_edges().size();

    // Signature
    std::string arg = "DeviceContext *dc";
    for (int i = 0; i < n_inedge; i++) {
        arg += ", uint64_t *in" + std::to_string(i);
    }
    for (int i = 0; i < n_outedge; i++) {
        arg += ", uint64_t *out" + std::to_string(i);
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
    w_body << "\n// Call kernels\n";
    std::vector<std::string> graph_input_edges;
    std::vector<std::string> graph_output_edges;

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
        std::unordered_map<std::string, int> inedge_map;
        std::unordered_map<std::string, int> outedge_map;
        std::vector<std::string> inedge_names;
        std::vector<std::string> outedge_names;
        w_body << "//\t outputs: ";
        for (auto edge : node->get_out_edges()) {
            auto src = edge->get_src();
            auto dst = edge->get_dst();
            std::string edge_name =
                "edge_" + src->get_op_name() + "_" + dst->get_op_name();
            if (outedge_map.contains(edge_name)) {
                outedge_map[edge_name]++;
                edge_name += "_" + std::to_string(outedge_map[edge_name]);
            } else {
                outedge_map[edge_name] = 1;
            }
            w << "uint64_t *" << edge_name << " = nullptr;\n";
            w_body << edge_name << ", ";
            outedge_names.push_back(edge_name);
            if (node->get_op_type() == "Init") {
                graph_input_edges.push_back(edge_name);
            }
        }
        w_body << "\n";
        w_body << "//\t inputs: ";
        for (auto edge : node->get_in_edges()) {
            auto src = edge->get_src();
            auto dst = edge->get_dst();
            std::string edge_name =
                "edge_" + src->get_op_name() + "_" + dst->get_op_name();
            if (inedge_map.contains(edge_name)) {
                inedge_map[edge_name]++;
                edge_name += "_" + std::to_string(inedge_map[edge_name]);
            } else {
                inedge_map[edge_name] = 1;
            }
            w_body << edge_name << ", ";
            inedge_names.push_back(edge_name);
            if (node->get_op_type() == "End") {
                graph_output_edges.push_back(edge_name);
            }
        }
        w_body << "\n";

        // Call the kernel
        if ((node->get_op_type() == "Init") | (node->get_op_type() == "End")) {
            w_body << "// Nothing to call\n\n";
        } else {
            std::string args = "dc";
            for (auto edge_name : inedge_names) {
                args += ", " + edge_name;
            }
            for (auto edge_name : outedge_names) {
                args += ", " + edge_name;
            }
            w_body << "kernel_" << node->get_op_type() << "<<<1024, 1024>>>"
                   << "(" << args << ");\n";
            w_body << "checkCudaErrors(cudaDeviceSynchronize());\n\n";
        }

        // Update the stack
        for (auto edge : node->get_out_edges()) {
            stack.push_back(edge->get_dst()->get_id());
        }
    }

    w << "\n// Combine input edges\n";
    for (int i = 0; i < n_inedge; i++) {
        w << graph_input_edges[i] << " = in" << i << ";\n";
    }

    w_body << "\n// Combine output edges\n";
    for (int i = 0; i < n_outedge; i++) {
        w_body << "out" << i << " = " << graph_output_edges[i] << ";\n";
    }

    w.indent_dec();
    w_body.indent_dec();
    w_body << "}\n\n";
    w.write_to_file(filename, if_append);
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
    w << "#include \"hifive/kernel/device_context.hpp\"\n\n";
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
    w << "DeviceContext dc;\n\n";

    std::vector<std::string> input_args;
    std::vector<std::string> output_args;

    w << "// Input arguments\n";
    std::shared_ptr<hifive::core::Node> init_node = graph->get_init_node();
    int i = 0;
    for (auto edge : init_node->get_out_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_dst();
        std::string name =
            "init" + std::to_string(i) + "_to_" + e->get_op_name();
        std::string name_h = name + "_h";
        std::string name_d = name + "_d";
        std::string name_size = "sizeof(uint64_t) * " +
                                std::to_string(edge->get_shape(0)) + " * " +
                                std::to_string(edge->get_shape(1));
        input_args.push_back(name_d);
        w << "// Edge: " << init_node->get_op_name() << " -> "
          << e->get_op_name() << "\n";
        w << "uint64_t *" << name_h << ";\n";
        w << "uint64_t *" << name_d << ";\n";
        w << "cudaMallocHost((void **)&" << name_h << ", " << name_size
          << ");\n";
        w << "cudaMalloc((void **)&" << name_d << ", " << name_size << ");\n";
        i++;
    }

    w << "\n// Output arguments\n";
    std::shared_ptr<hifive::core::Node> exit_node = graph->get_exit_node();
    i = 0;
    for (auto edge : exit_node->get_in_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_src();
        std::string name =
            "end" + std::to_string(i) + "_from_" + e->get_op_name();
        std::string name_h = name + "_h";
        std::string name_d = name + "_d";
        std::string name_size = "sizeof(uint64_t) * " +
                                std::to_string(edge->get_shape(0)) + " * " +
                                std::to_string(edge->get_shape(1));
        output_args.push_back(name_d);
        w << "// Edge: " << e->get_op_name() << " -> "
          << exit_node->get_op_name() << "\n";
        w << "uint64_t *" << name_h << ";\n";
        w << "uint64_t *" << name_d << ";\n";
        w << "cudaMallocHost((void **)&" << name_h << ", " << name_size
          << ");\n";
        w << "cudaMalloc((void **)&" << name_d << ", " << name_size << ");\n";
        i++;
    }

    w << "\n// Fill input arguments\n";

    std::string input_args_str = "&dc";
    for (auto arg : input_args) {
        input_args_str += ", " + arg;
    }
    for (auto arg : output_args) {
        input_args_str += ", " + arg;
    }

    w << "\n";
    w << "for (int i = 0; i < 5; i++) {\n";
    w.indent_inc();
    w << "// Run the graph\n";
    w << "auto start = std::chrono::high_resolution_clock::now();\n";
    w << "entry_kernel(" << input_args_str << ");\n";
    w << "auto end = std::chrono::high_resolution_clock::now();\n";
    w << "auto elapsed_usec = "
         "std::chrono::duration_cast<std::chrono::microseconds>(end - "
         "start);\n";
    w << "std::cout << \"Elapsed time: \" << elapsed_usec.count() << \"us\" << "
         "std::endl;\n";
    w.indent_dec();
    w << "}\n";
    w << "std::cout << \"Finished Benchmarking...\" << std::endl;\n";
    w.block_end();

    w.write_to_file(output_filename, true);
    return true;
}
} // namespace engine
} // namespace hifive