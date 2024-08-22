#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "gpucontext.h"

namespace hifive {

GPUContext::GPUContext() {
    cudaFree(0);

    GetGPUInfo(false);
}

void GPUContext::GetGPUInfo(bool verbose) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        if (verbose)
            std::cout << "No CUDA devices found" << std::endl;
    } else {
        if (verbose)
            std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    }
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        if (verbose) {
            std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
            std::cout << "  Compute capability: " << deviceProp.major << "."
                      << deviceProp.minor << std::endl;
            std::cout << "  Total global memory: "
                      << deviceProp.totalGlobalMem * 1.0 / (1 << 30) << " GB"
                      << std::endl;
            std::cout << "  Shared memory per block: "
                      << deviceProp.sharedMemPerBlock * 1.0 / (1 << 10) << " KB"
                      << std::endl;
            std::cout << "  Shared memory per Multiprocessor: "
                      << deviceProp.sharedMemPerMultiprocessor * 1.0 / (1 << 10)
                      << " KB" << std::endl;
            std::cout << "  Total Multiprocessors: "
                      << deviceProp.multiProcessorCount << std::endl;
            std::cout << "  Threads per Multiprocessor: "
                      << deviceProp.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  Registers per block: " << deviceProp.regsPerBlock
                      << std::endl;
            std::cout << "  Max threads per block: "
                      << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Max threads dimensions: "
                      << deviceProp.maxThreadsDim[0] << " x "
                      << deviceProp.maxThreadsDim[1] << " x "
                      << deviceProp.maxThreadsDim[2] << std::endl;
            std::cout << "  Max grid size: " << deviceProp.maxGridSize[0]
                      << " x " << deviceProp.maxGridSize[1] << " x "
                      << deviceProp.maxGridSize[2] << std::endl;
            std::cout << "  Clock rate: "
                      << deviceProp.clockRate / std::pow(10, 6) << " GHz"
                      << std::endl;
        }
        max_threads_per_block_ = deviceProp.maxThreadsPerBlock;
    }
}

} // namespace hifive