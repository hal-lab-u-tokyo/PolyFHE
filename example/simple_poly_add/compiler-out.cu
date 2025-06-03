#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError_t err, const char *filename, int line);
inline void __checkCudaErrors(cudaError_t err, const char *filename, int line) {
    assert(filename);
    if (cudaSuccess != err) {
        const char *ename = cudaGetErrorName(err);
        printf(
            "CUDA API Error %04d: \"%s\" from file <%s>, "
            "line %i.\n",
            err, ((ename != NULL) ? ename : "Unknown"), filename, line);
        // exit(err);
    }
}

__global__ void add_kernel(uint64_t *a, uint64_t *b, uint64_t *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// void entry_kernel(uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, size_t
// n) {
//     uint64_t *a = reinterpret_cast<uint64_t *>(a_ptr);
//     uint64_t *b = reinterpret_cast<uint64_t *>(b_ptr);
//     uint64_t *c = reinterpret_cast<uint64_t *>(c_ptr);
//     add_kernel<<<1, 32>>>(a, b, c, n);
//     cudaDeviceSynchronize(); // Ensure the kernel execution is complete
// }
void entry_kernel(pybind11::dict input) {
    uint64_t *d_a =
        reinterpret_cast<uint64_t *>(pybind11::cast<uintptr_t>(input["a"]));
    uint64_t *d_b =
        reinterpret_cast<uint64_t *>(pybind11::cast<uintptr_t>(input["b"]));
    uint64_t *d_c =
        reinterpret_cast<uint64_t *>(pybind11::cast<uintptr_t>(input["c"]));
    size_t n = pybind11::cast<size_t>(input["n"]);
    add_kernel<<<1, 32>>>(d_a, d_b, d_c, n);
    checkCudaErrors(cudaDeviceSynchronize());
}

PYBIND11_MODULE(polygraph, m) {
    m.def("entry_kernel", &entry_kernel, "A function which adds two numbers");
}