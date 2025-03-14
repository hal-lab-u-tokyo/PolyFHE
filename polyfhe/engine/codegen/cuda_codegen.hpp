#pragma once

#include <map>
#include <memory>

#include "polyfhe/engine/codegen/codegen_base.hpp"
#include "polyfhe/engine/codegen/codegen_writer.hpp"

namespace polyfhe {
namespace engine {
class CudaCodegen : public CodegenBase {
public:
    bool run_on_graph(std::shared_ptr<polyfhe::core::Graph>& graph) override;
    void generate_kernel_defs(std::shared_ptr<polyfhe::core::Graph>& graph,
                              const std::string& filename,
                              const bool if_append);
    void generate_entry(std::shared_ptr<polyfhe::core::Graph>& graph,
                        const std::string& filename, const bool if_append);
    void generate_include(std::shared_ptr<polyfhe::core::Graph>& graph,
                          const std::string& filename, const bool if_append);
    void generate_call_kernels(std::shared_ptr<polyfhe::core::Graph>& graph,
                               CodeWriter& w);

    void generate_ElemWiseOp(std::shared_ptr<polyfhe::core::Node>& node,
                             CodeWriter& w,
                             std::shared_ptr<polyfhe::core::Edge> out,
                             std::shared_ptr<polyfhe::core::Edge> in0,
                             std::shared_ptr<polyfhe::core::Edge> in1,
                             polyfhe::core::SubgraphType s_type);
    void generate_NTT(std::shared_ptr<polyfhe::core::Node>& node, CodeWriter& w,
                      bool if_ntt, bool if_phase1);
    void generate_NTT_ElemLimb(std::shared_ptr<polyfhe::core::Node>& node,
                               CodeWriter& w, bool if_ntt, bool if_phase1);
    void generate_modup(std::shared_ptr<polyfhe::core::Node>& node,
                        CodeWriter& w, std::string sPoly_x, std::string n_gidx,
                        std::string n_sidx);
};
} // namespace engine
} // namespace polyfhe