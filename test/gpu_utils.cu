#include <gtest/gtest.h>

#include "gpu_utils.h"

TEST(GPUUtils, GetGPUConfig) { 
    hifive::GPUContext gpu;
    gpu.GetGPUInfo(true);
}