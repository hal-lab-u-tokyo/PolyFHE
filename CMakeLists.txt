cmake_minimum_required(VERSION 3.20)

project(PolyFHE VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "-O3 -g -flto -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all)
endif ()

add_subdirectory(thirdparty)