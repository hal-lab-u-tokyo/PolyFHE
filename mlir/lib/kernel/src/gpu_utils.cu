#include <iostream>

#include "gpu_utils.h"

void __checkCudaErrors(cudaError_t err, const char *filename, int line) {
    assert(filename);
    if (cudaSuccess != err) {
        const char *ename = cudaGetErrorName(err);
        printf(
            "CUDA API Error %04d: \"%s\" from file <%s>, "
            "line %i.\n",
            err, ((ename != NULL) ? ename : "Unknown"), filename, line);
        exit(err);
    }
}

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

gpu_ptr make_and_copy_gpu_ptr(uint64_t *src, uint64_t size) {
    uint64_t *d_ptr;
    checkCudaErrors(cudaMalloc(&d_ptr, size * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_ptr, src, size * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    return gpu_ptr(d_ptr, size);
}

gpu_ptr make_gpu_ptr(uint64_t size) {
    uint64_t *d_ptr;
    checkCudaErrors(cudaMalloc(&d_ptr, size * sizeof(uint64_t)));
    return gpu_ptr(d_ptr, size);
}

} // namespace hifive