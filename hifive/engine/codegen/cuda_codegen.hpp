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
    void generate_kernel_defs(std::shared_ptr<hifive::core::Graph>& graph,
                              const std::string& filename,
                              const bool if_append);
    void generate_entry(std::shared_ptr<hifive::core::Graph>& graph,
                        const std::string& filename, const bool if_append);
    void generate_include(std::shared_ptr<hifive::core::Graph>& graph,
                          const std::string& filename, const bool if_append);
    void generate_call_kernels(std::shared_ptr<hifive::core::Graph>& graph,
                               CodeWriter& w);

    void generate_modup(std::shared_ptr<hifive::core::Node>& node,
                        CodeWriter& w, std::string sPoly_x, std::string n_gidx,
                        std::string n_sidx);
};
} // namespace engine
} // namespace hifive