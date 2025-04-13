add_executable(polyfhe_example example.cpp ckks.cpp)

target_link_libraries(polyfhe_example seal)
target_include_directories(polyfhe_example
    PRIVATE thirdparty/SEAL/native/src/seal
)