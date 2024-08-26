#pragma once

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

namespace hifive {

void __checkCudaErrors(cudaError_t err, const char *filename, int line);
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

class GPUContext {
public:
    GPUContext();
    ~GPUContext() = default;
    void GetGPUInfo(bool verbose);

    int max_threads_per_block_;
};

template <typename T>
class gpu_ptr {
public:
    gpu_ptr(T *d_ptr, uint64_t size) : d_ptr_(d_ptr), n_(size) {}
    ~gpu_ptr() { cudaFree(d_ptr_); }

    T *get() const { return d_ptr_; }
    uint64_t size() const { return n_; }

private:
    T *d_ptr_ = nullptr;
    uint64_t n_ = 0;
};

template <typename T>
gpu_ptr<T> make_gpu_ptr(T *src, uint64_t size) {
    T *d_ptr;
    checkCudaErrors(cudaMalloc(&d_ptr, size * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_ptr, src, size * sizeof(T), cudaMemcpyHostToDevice));
    return gpu_ptr<T>(d_ptr, size);
}

} // namespace hifive