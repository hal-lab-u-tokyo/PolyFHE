#pragma once

#include <map>
#include <memory>

#include "hifive/engine/codegen/codegen_base.hpp"
#include "hifive/engine/codegen/codegen_writer.hpp"

namespace hifive {
namespace engine {
class CudaCodegen : public CodegenBase {
public:
    bool run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) override;
    void emit_kernel(std::shared_ptr<hifive::core::Graph>& graph,
                     std::string filename);

private:
    std::map<std::string, std::shared_ptr<CodeUnitKernel>> m_cu_kernels;
};
} // namespace engine
} // namespace hifive