#pragma once

#include "hifive/engine/codegen/codegen_base.hpp"

namespace hifive {
namespace engine {
class CudaCodegen : public CodegenBase {
public:
    bool run_on_graph(std::shared_ptr<hifive::core::Graph>& graph) override;
};
} // namespace engine
} // namespace hifive