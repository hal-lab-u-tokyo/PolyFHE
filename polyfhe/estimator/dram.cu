#include <cuda.h>

#include <cstdio>

__global__ void memory_Queue(uint32_t *A, uint64_t *t_start, uint64_t *t_end,
                             uint32_t *smid, uint32_t *warpid) {
    // Time recording
    uint64_t t[2] = {0};
    uint32_t p_smid, p_warpid;
    asm("mov.u32 %0, %smid;" : "=r"(p_smid));
    asm("mov.u32 %0, %warpid;" : "=r"(p_warpid));

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *a_ptr = A + idx;
    int val = 0;
    t[0] = clock();
    val = *a_ptr;
    asm volatile("membar.gl;");
    //__threadfence();
    t[1] = clock();

    // Time recording
    t_start[idx] = t[0];
    t_end[idx] = t[1];
    smid[idx] = p_smid;
    warpid[idx] = p_warpid;

    // dummy
    A[idx] = val;
}

int main() {
    const int grid_size = 48;
    const int block_size = 1024;
    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);
    uint32_t *d_A, *d_smid, *d_warpid;
    uint64_t *d_t_start, *d_t_end;
    cudaMalloc(&d_A, grid_size * block_size * sizeof(uint32_t));
    cudaMalloc(&d_t_start, grid_size * block_size * sizeof(uint64_t));
    cudaMalloc(&d_t_end, grid_size * block_size * sizeof(uint64_t));
    cudaMalloc(&d_smid, grid_size * block_size * sizeof(uint32_t));
    cudaMalloc(&d_warpid, grid_size * block_size * sizeof(uint32_t));

    memory_Queue<<<grid, block>>>(d_A, d_t_start, d_t_end, d_smid, d_warpid);

    // Copy back
    uint32_t *A =
        (uint32_t *) malloc(grid_size * block_size * sizeof(uint32_t));
    uint64_t *t_start =
        (uint64_t *) malloc(grid_size * block_size * sizeof(uint64_t));
    uint64_t *t_end =
        (uint64_t *) malloc(grid_size * block_size * sizeof(uint64_t));
    uint32_t *smid =
        (uint32_t *) malloc(grid_size * block_size * sizeof(uint32_t));
    uint32_t *warpid =
        (uint32_t *) malloc(grid_size * block_size * sizeof(uint32_t));
    cudaMemcpy(A, d_A, grid_size * block_size * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(t_start, d_t_start, grid_size * block_size * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(t_end, d_t_end, grid_size * block_size * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(smid, d_smid, grid_size * block_size * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(warpid, d_warpid, grid_size * block_size * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Print CSV
    printf("t_start,t_end,delta,smid,warpid\n");
    for (uint32_t i = 0; i < grid_size * block_size; i++) {
        printf("%ld,%ld,%ld,%d,%d\n", t_start[i], t_end[i],
               t_end[i] - t_start[i], smid[i], warpid[i]);
    }

    // Free
    free(A);
    free(t_start);
    free(t_end);
    free(smid);
    free(warpid);
    cudaFree(d_A);
    cudaFree(d_t_start);
    cudaFree(d_t_end);
    cudaFree(d_smid);
    cudaFree(d_warpid);

    return 0;
}