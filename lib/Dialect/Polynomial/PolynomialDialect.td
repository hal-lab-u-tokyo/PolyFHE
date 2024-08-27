include "mlir/IR/DialectBase.td"

def Polynomial_Dialect : Dialect {
  let name = "polynomial";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::hifive::polynomial";
}