// RUN: hifive-opt %s

module {
  func.func @main(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    return %0 : !poly.poly<10>
  }
}
