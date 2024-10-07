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
    // for graph->inputs
    // gen cudaMallocHost
    // gen cudaMalloc
    // gen cudaMemcpy
    
    w << "// Output arguments\n";
    // as well for graph->outputs

    w.block_end();

    w.write_to_file("build/gen_cuda_main.cu");
    return true;
}
} // namespace engine
} // namespace hifive