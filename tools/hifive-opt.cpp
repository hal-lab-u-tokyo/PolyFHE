#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::hifive::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::hifive::AffineFullUnrollPassAsPatternRewrite>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hifive Pass Driver", registry));
}