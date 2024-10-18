#include "hifive/engine/pass/lowering_ckks_to_poly_pass.hpp"

#include "lowering_ckks_to_poly_pass.hpp"

namespace hifive {
namespace engine {
bool LoweringCKKSToPolyPass::run_on_graph(
    std::shared_ptr<hifive::core::Graph>& /*graph*/) {
    LOG_INFO("Running LoweringCKKSToPolyPass\n");

    return true;
}
} // namespace engine
} // namespace hifive