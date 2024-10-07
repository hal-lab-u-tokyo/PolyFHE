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

    w.write_to_file("build/gen_cuda_main.cu");
    return true;
}
} // namespace engine
} // namespace hifive