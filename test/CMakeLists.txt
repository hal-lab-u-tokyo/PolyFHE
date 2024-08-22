add_executable(test_hifive
    phantom.cu
)

target_include_directories(test_hifive PUBLIC /usr/local/cuda/include)

target_link_libraries(test_hifive Phantom)
target_include_directories(test_hifive PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
target_link_libraries(test_hifive GTest::gtest_main)