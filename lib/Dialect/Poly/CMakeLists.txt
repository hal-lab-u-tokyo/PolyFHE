# Inlining `add_mlir_dialect(Poly poly)` commands so that
# we can custom name `*.inc` generated files.
set(LLVM_TARGET_DEFINITIONS PolyDialect.td)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)

add_mlir_dialect_library(Poly
    PolyDialect.cc
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly

    LINK_LIBS PUBLIC
    )