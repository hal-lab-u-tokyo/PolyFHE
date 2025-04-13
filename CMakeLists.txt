cmake_minimum_required(VERSION 3.13)

project(PolyFHE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "-march=native -O3 -g -flto -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
)

add_subdirectory(thirdparty)
add_subdirectory(example)