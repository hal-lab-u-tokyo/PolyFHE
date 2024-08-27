add_mlir_library(AffineFullUnroll
    AffineFullUnroll.cpp
    
    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransforms
)