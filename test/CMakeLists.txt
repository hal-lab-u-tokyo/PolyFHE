include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(test_ckks
    cuckks.cu
    cuckks-phantom.cu
    gpucontext.cu
    phantom.cu
)
target_link_libraries(test_ckks HiFive)
target_link_libraries(test_ckks Phantom)
target_link_libraries(test_ckks GTest::gtest_main)
target_link_libraries(test_ckks seal)
target_include_directories(test_ckks PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_ckks PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include)
target_include_directories(test_ckks PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/examples)
target_include_directories(test_ckks PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/SEAL/native/src)
target_include_directories(test_ckks PUBLIC /usr/local/include)