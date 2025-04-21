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
            // TODO
            k_config.block_size = "params_h->n2";
            k_config.grid_size = "params_h->n1 * params_h->L";
            k_config.shared_mem_size =
                std::to_string(subgraph->get_smem_size());
        } else if (s_type == core::SubgraphType::ElemLimb1Slot) {
            int driver_shared_mem_size_per_block = 1;
            // TODO: use limb instead of L
            int shared_mem_size_phase1 =
                config->n1 * sizeof(uint64_t) * config->L;
            int block_per_sm_limited_by_shared_mem =
                config->SharedMemKB /
                (shared_mem_size_phase1 + driver_shared_mem_size_per_block);
            int ntt_threads = config->n1 / 2;
            // TODO: Max thread per block is 1024
            int thread_limited_by_shared_mem =
                1024 / block_per_sm_limited_by_shared_mem;
            int thread_ceil_to_ntt =
                ((ntt_threads + thread_limited_by_shared_mem - 1) /
                 ntt_threads) *
                ntt_threads;
            k_config.block_size = std::to_string(thread_ceil_to_ntt);
            k_config.grid_size = "params_h->n2";
            k_config.shared_mem_size = std::to_string(shared_mem_size_phase1);
        } else if (s_type == core::SubgraphType::ElemLimb2Slot) {
            int driver_smem_kb = 1;
            int smem = config->n2 * sizeof(uint64_t) * config->L;
            int smem_kb = smem / 1024;
            int block_per_sm_limited_by_smem =
                config->SharedMemKB / (smem_kb + driver_smem_kb);
            int ntt_threads = config->n2 / 2;
            int thread_limited_by_smem = 1024 / block_per_sm_limited_by_smem;
            int thread_ceil_to_ntt =
                ((ntt_threads + thread_limited_by_smem - 1) / ntt_threads) *
                ntt_threads;
            k_config.block_size = std::to_string(thread_ceil_to_ntt);
            k_config.grid_size = "params_h->n1";
            k_config.shared_mem_size = std::to_string(smem);
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