#include <gtest/gtest.h>

#include "gpu_utils.h"

using namespace hifive;

TEST(GPUUtils, GetGPUConfig) {
    GPUContext gpu;
    gpu.GetGPUInfo(true);
}

TEST(GPUUtils, MallocDeviceVector) {
    uint64_t *h_vec = new uint64_t[10];
    gpu_ptr<uint64_t> d_ptr = make_and_copy_gpu_ptr<uint64_t>(h_vec, 10);
    ASSERT_EQ(d_ptr.size(), 10);
    delete[] h_vec;
    // d_ptr will be automatically freed
}