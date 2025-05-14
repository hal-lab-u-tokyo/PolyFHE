#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>

__device__ void Scale2(uint64_t *a, int n, int l) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * l;
         idx += blockDim.x * gridDim.x) {
        a[idx] = a[idx] * 2;
    }
}

__device__ void Scale2Limb(uint64_t *a_i, int n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        a_i[idx] = a_i[idx] * 2;
    }
}

__global__ void Scale2Twice(uint64_t *a, int n, int l) {
    Scale2(a, n, l);
    Scale2(a, n, l);
}

__global__ void Scale2TwiceLimbByLimb(uint64_t *a, int n, int l) {
    for (int i = 0; i < l; i++) {
        Scale2Limb(a + i * n, n);
        Scale2Limb(a + i * n, n);
    }
}

__global__ void Scale2TwiceRegister(uint64_t *a, int n, int l) {
    for (int i = 0; i < l; i++) {
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
             idx += blockDim.x * gridDim.x) {
            uint64_t tmp = a[i * n + idx];
            tmp = tmp * 2;
            tmp = tmp * 2;
            a[i * n + idx] = tmp;
        }
    }
}

enum class ParamSize {
    Small,
    Medium,
    Large,
};

enum class OptLevel {
    None,
    Cache,
    Register,
};

int main(int argc, char **argv) {
    ParamSize param_size = ParamSize::Small;
    OptLevel opt_level = OptLevel::Register;

    if (argc != 2) {
        printf("Usage: %s <opt_level>\n", argv[0]);
        return -1;
    }
    int in_opt_level = atoi(argv[1]);
    printf("opt_level: %d\n", in_opt_level);
    switch (in_opt_level) {
    case 0:
        opt_level = OptLevel::None;
        break;
    case 1:
        opt_level = OptLevel::Cache;
        break;
    case 2:
        opt_level = OptLevel::Register;
        break;
    default:
        printf("Invalid opt_level\n");
        return -1;
    }

    int N, L;
    switch (param_size) {
    case ParamSize::Small:
        N = 1 << 15;
        L = 10;
        break;
    case ParamSize::Medium:
        N = 1 << 16;
        L = 20;
        break;
    case ParamSize::Large:
        N = 1 << 17;
        L = 50;
        break;
    default:
        printf("Invalid parameter size\n");
        return -1;
    }

    uint64_t *a, *d_a;
    size_t size = N * L * sizeof(uint64_t);
    a = (uint64_t *) malloc(size);
    cudaMalloc((void **) &d_a, size);
    for (int i = 0; i < N * L; i++) {
        a[i] = i;
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    switch (opt_level) {
    case OptLevel::Register:
        Scale2TwiceRegister<<<4096, 128>>>(d_a, N, L);
        cudaDeviceSynchronize();
        break;
    case OptLevel::Cache:
        Scale2TwiceLimbByLimb<<<4096, 128>>>(d_a, N, L);
        cudaDeviceSynchronize();
        break;
    case OptLevel::None:
        Scale2Twice<<<4096, 128>>>(d_a, N, L);
        cudaDeviceSynchronize();
        break;
    default:
        printf("Invalid optimization level\n");
        return -1;
    }

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * L; i++) {
        if (a[i] != i * 4) {
            printf("Error at index %d: expected %d, got %lu\n", i, i * 2, a[i]);
            break;
        }
    }
    printf("OK\n");

    double sum = 0;
    for (int iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        switch (opt_level) {
        case OptLevel::Register:
            Scale2TwiceRegister<<<4096, 128>>>(d_a, N, L);
            cudaDeviceSynchronize();
            break;
        case OptLevel::Cache:
            Scale2TwiceLimbByLimb<<<4096, 128>>>(d_a, N, L);
            cudaDeviceSynchronize();
            break;
        case OptLevel::None:
            Scale2Twice<<<4096, 128>>>(d_a, N, L);
            cudaDeviceSynchronize();
            break;
        default:
            printf("Invalid optimization level\n");
            return -1;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        sum += us.count();
        printf("Iteration %d: %ld us\n", iter, us.count());
    }
    printf("Average time: %lf us\n", sum / 10);
}