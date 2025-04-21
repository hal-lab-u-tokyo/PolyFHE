cmake_minimum_required(VERSION 3.20)

project(PolyFHE VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS
    "-O3 -g -flto -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all)
endif ()

set(POLYFHE_SOURCE_FILES "")
add_subdirectory(polyfhe)
add_subdirectory(thirdparty)
message(STATUS "PolyFHE source files: ${POLYFHE_SOURCE_FILES}")

add_executable(polyfhe-cc ${POLYFHE_SOURCE_FILES})
target_include_directories(polyfhe-cc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# Boost
find_package(Boost 1.70 REQUIRED COMPONENTS graph program_options)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    target_link_libraries(polyfhe-cc ${Boost_LIBRARIES})
    target_include_directories(polyfhe-cc PRIVATE ${Boost_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Boost not found")
endif ()