add_library(HiFive
    evaluate.cu
    gpucontext.cu
    polynomial.cu
)
target_link_libraries(HiFive Phantom)
target_include_directories(HiFive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include)
target_include_directories(HiFive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/examples)