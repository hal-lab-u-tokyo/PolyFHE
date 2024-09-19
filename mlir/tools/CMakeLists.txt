get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    AffineFullUnroll
    MLIROptLib
    MLIRPass
    MLIRPoly
)

add_llvm_executable(hifive-opt hifive-opt.cpp)

llvm_update_compile_flags(hifive-opt)
target_link_libraries(hifive-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hifive-opt)