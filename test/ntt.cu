#include "test.h"

TEST(NTT, NTT) {
    size_t batch_size = 2;
    size_t log_dim = 12;
    size_t dim = 1 << log_dim;

    // Modulus
    // TODO: wrap DModulus in gpu_ptr
    const auto modulus =
        seal::CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    hifive::DModulus *d_modulus;
    checkCudaErrors(
        cudaMalloc(&d_modulus, sizeof(hifive::DModulus) * batch_size));
    for (size_t i = 0; i < batch_size; i++) {
        checkCudaErrors(cudaMemcpy(&d_modulus[i], &modulus[i],
                                   sizeof(hifive::DModulus),
                                   cudaMemcpyHostToDevice));
    }

    hifive::DNTTTable d_ntt_table;
}