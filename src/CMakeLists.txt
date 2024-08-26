add_library(HiFive
    ciphertext.cu
    evaluate.cu
    gpu_utils.cu
    ntt.cu
    polynomial.cu
)
target_link_libraries(HiFive Phantom)
target_link_libraries(HiFive seal)
target_include_directories(HiFive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include)
target_include_directories(HiFive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/examples)
target_include_directories(HiFive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/SEAL/native/src)