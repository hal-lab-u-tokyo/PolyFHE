cmake_minimum_required(VERSION 3.20)

project(HiFive CUDA CXX)
set(CMAKE_CXX_FLAGS "-Wall")

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(src)
add_subdirectory(test)

# Phantom
add_library(Phantom SHARED
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/context.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/batchencoder.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/prng.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/polymath.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/secretkey.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/rns.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/rns_base.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/rns_bconv.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/scalingvariant.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/evaluate.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/fft.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ckks.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/galois.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/eval_key_switch.cu

        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/fntt_2d.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/ntt_1d.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/ntt_keyswitch_old.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/intt_2d.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/ntt_modup.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/ntt/ntt_moddown.cu

        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/blake2b.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/blake2xb.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/globals.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/hash.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/modulus.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/ntt.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/numth.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/rns.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/uintarith.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/uintarithmod.cu
        ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/host/uintarithsmallmod.cu
)
target_include_directories(Phantom PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include)
target_compile_options(Phantom PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(Phantom PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
target_compile_features(Phantom PUBLIC cxx_std_17 cuda_std_17)