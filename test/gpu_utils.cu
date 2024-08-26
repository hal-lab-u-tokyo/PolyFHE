#include "test.h"

__global__ void check_can_access(uint64_t *data, uint64_t idx) {
    data[idx] = 1;
}

TEST(GPUUtils, GetGPUConfig) {
    GPUContext gpu;
    gpu.GetGPUInfo(true);
}

TEST(GPUUtils, MakeAndCopyGPUPtr) {
    int size = 10;
    uint64_t *h_vec = new uint64_t[size];
    gpu_ptr d_ptr = make_and_copy_gpu_ptr(h_vec, size);
    check_can_access<<<1, 1>>>(d_ptr.get(), 0);
    check_can_access<<<1, 1>>>(d_ptr.get(), size - 1);
    ASSERT_EQ(d_ptr.size(), size);
    delete[] h_vec;
    // d_ptr will be automatically freed
}

TEST(GPUUtils, MakeGPUPtr) {
    int size = 10;
    gpu_ptr d_ptr = make_gpu_ptr(size);
    check_can_access<<<1, 1>>>(d_ptr.get(), 0);
    check_can_access<<<1, 1>>>(d_ptr.get(), size - 1);
    ASSERT_EQ(d_ptr.size(), size);
}