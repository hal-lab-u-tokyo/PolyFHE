include "mlir/IR/DialectBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for poly math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polys over integers.
  }];

  let cppNamespace = "::mlir::hifive::poly";
}