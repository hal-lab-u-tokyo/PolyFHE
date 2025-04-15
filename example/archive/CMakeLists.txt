# ==============
# googletest
# ==============
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(polyfhe_example example.cu ${CMAKE_SOURCE_DIR}/build/generated.cu ${CMAKE_SOURCE_DIR}/polyfhe/kernel/device_context.cu) 
target_link_libraries(polyfhe_example Phantom gtest_main)
target_include_directories(polyfhe_example
    PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/
    PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
    PRIVATE ${CMAKE_SOURCE_DIR}/
)
enable_testing()