#include "hifive/engine/codegen/cuda_codegen.hpp"

#include "hifive/core/logger.h"
#include "hifive/engine/codegen/codegen_writer.hpp"

namespace hifive {
namespace engine {
bool CudaCodegen::run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running CudaCodegen\n");

    CodeWriter w;
    w << "#include <cuda.h>\n";
    w << "#include <cuda_runtime.h>\n";

    w << "int main(int argc, char *argv[])";
    w.block_begin();
    w << "// cuda_init();\n\n";

    w << "// Input arguments\n";
    std::shared_ptr<hifive::core::Node> init_node = graph->get_init_node();
    for (auto edge : init_node->get_out_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_dst();
        w << "// Edge: " << init_node->get_op_name() << " -> "
          << e->get_op_name() << "\n";
        w << "uint64_t *input_" << e->get_op_name() << "_h;\n";
        w << "uint64_t *input_" << e->get_op_name() << "_d;\n";
    }

    w << "// Output arguments\n";
    std::shared_ptr<hifive::core::Node> exit_node = graph->get_exit_node();
    for (auto edge : exit_node->get_in_edges()) {
        std::shared_ptr<hifive::core::Node> e = edge->get_src();
        w << "// Edge: " << e->get_op_name() << " -> "
          << exit_node->get_op_name() << "\n";
        w << "uint64_t *output_" << e->get_op_name() << "_h;\n";
        w << "uint64_t *output_" << e->get_op_name() << "_d;\n";
    }

    w.block_end();

    w.write_to_file("build/gen_cuda_main.cu");
    return true;
}
} // namespace engine
} // namespace hifive