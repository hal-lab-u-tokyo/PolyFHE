#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "=== Device " << dev << ": " << prop.name << " ===\n";
        std::cout << "SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor
                  << "\n";
        std::cout << "L1 Cache per SM: "
                  << prop.sharedMemPerMultiprocessor / 1024
                  << " KB (shared memory)\n";
        std::cout << "L2 Cache: " << prop.l2CacheSize / (1024 * 1024.0)
                  << " MB\n";
        std::cout << "Global Memory (DRAM): " << std::fixed
                  << std::setprecision(2)
                  << static_cast<double>(prop.totalGlobalMem) /
                         (1024.0 * 1024 * 1024)
                  << " GB\n";

        double bandwidth = 2.0 * prop.memoryClockRate *
                           (prop.memoryBusWidth / 8) / 1.0e6; // GB/s
        std::cout << "Theoretical Memory Bandwidth: " << bandwidth << " GB/s\n";

        std::cout << std::endl;
    }

    return 0;
}