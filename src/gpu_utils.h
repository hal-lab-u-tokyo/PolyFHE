#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace hifive {

class GPUContext {
public:
    GPUContext();
    ~GPUContext() = default;
    void GetGPUInfo(bool verbose);

    int max_threads_per_block_;
};

} // namespace hifive