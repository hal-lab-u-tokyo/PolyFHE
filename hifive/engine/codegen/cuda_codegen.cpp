#include "hifive/engine/codegen/cuda_codegen.hpp"

#include "hifive/core/logger.h"

namespace hifive {
namespace engine {
bool CudaCodegen::run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) {
    LOG_INFO("Running CudaCodegen\n");
    return true;
}
} // namespace engine
} // namespace hifive