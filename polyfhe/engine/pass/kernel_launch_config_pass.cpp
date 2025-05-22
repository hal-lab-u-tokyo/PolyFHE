#include "polyfhe/engine/pass/kernel_launch_config_pass.hpp"

#include <climits>
#include <iostream>

namespace polyfhe {
namespace engine {
bool KernelLaunchConfigPass::run_on_graph(
    std::shared_ptr<polyfhe::core::Graph>& graph) {
    LOG_INFO("Running KernelLaunchConfigPass\n");

    std::vector<std::shared_ptr<polyfhe::core::SubGraph>> subgraphs =
        graph->get_subgraphs();
    for (auto subgraph : subgraphs) {
        core::SubgraphType s_type = subgraph->get_subgraph_type();

        core::KernelLaunchConfig k_config;
        polyfhe::Config* config = graph->m_config.get();
        std::cout << "s_type: " << core::to_string(s_type) << std::endl;

        if (s_type == core::SubgraphType::Elem) {
            // If subgraph contains only ONE node, we don't need to malloc
            // shared memory
            if (subgraph->get_nodes().size() > 1) {
                k_config.shared_mem_size =
                    std::to_string(subgraph->get_smem_size());
            } else {
                k_config.shared_mem_size = "0";
            }
            k_config.block_size = "256";
            k_config.grid_size = "4096";
        } else if (s_type == core::SubgraphType::ElemLimb1) {
            k_config.shared_mem_size =
                "(params_h->n1 + params_h->pad + 1) * params_h->pad "
                "* sizeof(uint64_t)";
            k_config.block_size = "(params_h->n1 / 8) * params_h->pad";
            k_config.grid_size = "4096";
        } else if (s_type == core::SubgraphType::ElemLimb2) {
            k_config.shared_mem_size =
                "params_h->per_thread_ntt_size * 128 * sizeof(uint64_t)";
            k_config.block_size = "128";
            k_config.grid_size = "4096";
        } else if (s_type == core::SubgraphType::ElemSlot) {
            k_config.block_size = "128";
            k_config.grid_size =
                "params_h->N * obase.size() / 128 / unroll_factor";
            // k_config.shared_mem_size =
            //     "(128 * 2 + obase.size()) * ibase.size() * sizeof(uint64_t)";
            k_config.shared_mem_size =
                "obase.size() * ibase.size() * sizeof(uint64_t)";
        } else if (s_type == core::SubgraphType::ElemLimb1Slot) {
            // TODO
            k_config.block_size = "128";
            k_config.grid_size = "4096";
            k_config.shared_mem_size = "params_h->L * sizeof(uint64_t)";
        } else if (s_type == core::SubgraphType::ElemLimb2Slot) {
            // TODO
            k_config.block_size = "128";
            k_config.grid_size = "4096";
            k_config.shared_mem_size = "params_h->L * sizeof(uint64_t)";
        } else if (s_type == core::SubgraphType::NoAccess) {
            // We don't need to launch kernel
            k_config.block_size = "0";
            k_config.grid_size = "0";
            k_config.shared_mem_size = "0";
        } else if (s_type == core::SubgraphType::L2) {
            // pass
        } else {
            LOG_ERROR("Unexpected subgraph type\n");
        }

        // Set kernel launch config
        subgraph->set_kernel_launch_config(k_config);

        std::cout << "Subgraph " << core::to_string(s_type) << std::endl;
        std::cout << "  grid_size: " << k_config.grid_size << std::endl;
        std::cout << "  block_size: " << k_config.block_size << std::endl;
        std::cout << "  shared_mem_size: " << k_config.shared_mem_size
                  << std::endl;
    }

    return true;
}
} // namespace engine
} // namespace polyfhe