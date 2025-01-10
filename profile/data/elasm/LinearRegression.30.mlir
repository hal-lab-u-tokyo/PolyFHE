module @"traced/LinearRegression.mlir" {
  func.func @_hecate_LinearRegression(%arg0: tensor<1x!ckks.poly<2 * 0>>, %arg1: tensor<1x!ckks.poly<2 * 0>>) -> (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) attributes {arg_level = array<i64: 8, 8>, arg_scale = array<i64: 30, 30>, est_error = 0.0031475038143703991 : f64, est_latency = 8.246530e+05 : f64, init_level = 8 : i64, no_mutation = true, res_level = array<i64: 1, 1>, res_scale = array<i64: 43, 43>, sm_plan_edge = array<i64: 7, 21, 18, 25, 6, 39, 1, 22>, sm_plan_level = array<i64: 1, 1, 2, 0, 2, 1, 2, 1>, sm_plan_scale = array<i64: 0, 2, 14, 8, 0, 13, 0, 13>, smu0 = 0 : i64, smu1 = 1 : i64, smu_attached = false} {
    %0 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %1 = "ckks.modswitchc"(%0, %arg1) {downFactor = 2 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %2 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %3 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %4 = "ckks.encode"(%3) {level = 6 : i64, scale = 60 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %5 = "ckks.mulcp"(%2, %1, %4) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %6 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %7 = "ckks.negatec"(%6, %5) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %8 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %9 = "ckks.modswitchc"(%8, %arg0) {downFactor = 2 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %10 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %11 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %12 = "ckks.encode"(%11) {level = 6 : i64, scale = 60 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %13 = "ckks.mulcp"(%10, %9, %12) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %14 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %15 = "ckks.addcc"(%14, %13, %7) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %16 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %17 = "ckks.rescalec"(%16, %15) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %18 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %19 = "ckks.modswitchc"(%18, %arg0) {downFactor = 3 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %20 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %21 = "ckks.mulcc"(%20, %17, %19) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %22 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %23 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %24 = "ckks.encode"(%23) {level = 5 : i64, scale = 43 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %25 = "ckks.mulcp"(%22, %21, %24) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %26 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %27 = "ckks.encode"(%26) {level = 5 : i64, scale = 30 : i64, value = 2 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %28 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %29 = "ckks.mulcp"(%28, %25, %27) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %30 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %31 = "ckks.rescalec"(%30, %29) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %32 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %33 = "ckks.rotatec"(%32, %31) {offset = array<i64: 2048>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %34 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %35 = "ckks.addcc"(%34, %31, %33) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %36 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %37 = "ckks.rotatec"(%36, %35) {offset = array<i64: 1024>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %38 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %39 = "ckks.addcc"(%38, %35, %37) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %40 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %41 = "ckks.rotatec"(%40, %39) {offset = array<i64: 512>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %42 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %43 = "ckks.addcc"(%42, %39, %41) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %44 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %45 = "ckks.rotatec"(%44, %43) {offset = array<i64: 256>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %46 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %47 = "ckks.addcc"(%46, %43, %45) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %48 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %49 = "ckks.rotatec"(%48, %47) {offset = array<i64: 128>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %50 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %51 = "ckks.addcc"(%50, %47, %49) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %52 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %53 = "ckks.rotatec"(%52, %51) {offset = array<i64: 64>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %54 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %55 = "ckks.addcc"(%54, %51, %53) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %56 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %57 = "ckks.rotatec"(%56, %55) {offset = array<i64: 32>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %58 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %59 = "ckks.addcc"(%58, %55, %57) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %60 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %61 = "ckks.rotatec"(%60, %59) {offset = array<i64: 16>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %62 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %63 = "ckks.addcc"(%62, %59, %61) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %64 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %65 = "ckks.rotatec"(%64, %63) {offset = array<i64: 8>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %66 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %67 = "ckks.addcc"(%66, %63, %65) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %68 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %69 = "ckks.rotatec"(%68, %67) {offset = array<i64: 4>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %70 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %71 = "ckks.addcc"(%70, %67, %69) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %72 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %73 = "ckks.rotatec"(%72, %71) {offset = array<i64: 2>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %74 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %75 = "ckks.addcc"(%74, %71, %73) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %76 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %77 = "ckks.rotatec"(%76, %75) {offset = array<i64: 1>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %78 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %79 = "ckks.addcc"(%78, %75, %77) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %80 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %81 = "ckks.modswitchc"(%80, %15) {downFactor = 1 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %82 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %83 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %84 = "ckks.encode"(%83) {level = 5 : i64, scale = 43 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %85 = "ckks.mulcp"(%82, %81, %84) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %86 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %87 = "ckks.rescalec"(%86, %85) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %88 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %89 = "ckks.encode"(%88) {level = 4 : i64, scale = 30 : i64, value = 2 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %90 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %91 = "ckks.mulcp"(%90, %87, %89) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %92 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %93 = "ckks.rotatec"(%92, %91) {offset = array<i64: 2048>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %94 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %95 = "ckks.addcc"(%94, %91, %93) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %96 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %97 = "ckks.rotatec"(%96, %95) {offset = array<i64: 1024>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %98 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %99 = "ckks.addcc"(%98, %95, %97) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %100 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %101 = "ckks.rotatec"(%100, %99) {offset = array<i64: 512>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %102 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %103 = "ckks.addcc"(%102, %99, %101) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %104 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %105 = "ckks.rotatec"(%104, %103) {offset = array<i64: 256>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %106 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %107 = "ckks.addcc"(%106, %103, %105) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %108 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %109 = "ckks.rotatec"(%108, %107) {offset = array<i64: 128>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %110 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %111 = "ckks.addcc"(%110, %107, %109) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %112 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %113 = "ckks.rotatec"(%112, %111) {offset = array<i64: 64>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %114 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %115 = "ckks.addcc"(%114, %111, %113) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %116 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %117 = "ckks.rotatec"(%116, %115) {offset = array<i64: 32>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %118 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %119 = "ckks.addcc"(%118, %115, %117) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %120 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %121 = "ckks.rotatec"(%120, %119) {offset = array<i64: 16>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %122 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %123 = "ckks.addcc"(%122, %119, %121) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %124 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %125 = "ckks.rotatec"(%124, %123) {offset = array<i64: 8>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %126 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %127 = "ckks.addcc"(%126, %123, %125) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %128 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %129 = "ckks.rotatec"(%128, %127) {offset = array<i64: 4>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %130 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %131 = "ckks.addcc"(%130, %127, %129) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %132 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %133 = "ckks.rotatec"(%132, %131) {offset = array<i64: 2>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %134 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %135 = "ckks.addcc"(%134, %131, %133) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %136 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %137 = "ckks.rotatec"(%136, %135) {offset = array<i64: 1>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %138 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %139 = "ckks.addcc"(%138, %135, %137) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %140 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %141 = "ckks.encode"(%140) {level = 4 : i64, scale = 30 : i64, value = 1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %142 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %143 = "ckks.mulcp"(%142, %79, %141) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %144 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %145 = "ckks.mulcp"(%144, %139, %141) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %146 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %147 = "ckks.rescalec"(%146, %145) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %148 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %149 = "ckks.encode"(%148) {level = 4 : i64, scale = 103 : i64, value = 0 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %150 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %151 = "ckks.addcp"(%150, %143, %149) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %152 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %153 = "ckks.rescalec"(%152, %151) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %154 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %155 = "ckks.modswitchc"(%154, %arg0) {downFactor = 5 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %156 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %157 = "ckks.mulcc"(%156, %155, %153) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %158 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %159 = "ckks.addcc"(%158, %157, %147) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %160 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %161 = "ckks.modswitchc"(%160, %7) {downFactor = 2 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %162 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %163 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %164 = "ckks.encode"(%163) {level = 4 : i64, scale = 43 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %165 = "ckks.mulcp"(%162, %161, %164) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %166 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %167 = "ckks.rescalec"(%166, %165) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %168 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %169 = "ckks.addcc"(%168, %159, %167) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %170 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %171 = "ckks.mulcc"(%170, %169, %155) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %172 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %173 = "ckks.rescalec"(%172, %171) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %174 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %175 = "ckks.encode"(%174) {level = 2 : i64, scale = 30 : i64, value = 2 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %176 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %177 = "ckks.mulcp"(%176, %173, %175) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %178 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %179 = "ckks.rotatec"(%178, %177) {offset = array<i64: 2048>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %180 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %181 = "ckks.addcc"(%180, %177, %179) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %182 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %183 = "ckks.rotatec"(%182, %181) {offset = array<i64: 1024>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %184 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %185 = "ckks.addcc"(%184, %181, %183) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %186 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %187 = "ckks.rotatec"(%186, %185) {offset = array<i64: 512>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %188 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %189 = "ckks.addcc"(%188, %185, %187) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %190 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %191 = "ckks.rotatec"(%190, %189) {offset = array<i64: 256>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %192 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %193 = "ckks.addcc"(%192, %189, %191) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %194 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %195 = "ckks.rotatec"(%194, %193) {offset = array<i64: 128>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %196 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %197 = "ckks.addcc"(%196, %193, %195) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %198 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %199 = "ckks.rotatec"(%198, %197) {offset = array<i64: 64>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %200 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %201 = "ckks.addcc"(%200, %197, %199) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %202 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %203 = "ckks.rotatec"(%202, %201) {offset = array<i64: 32>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %204 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %205 = "ckks.addcc"(%204, %201, %203) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %206 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %207 = "ckks.rotatec"(%206, %205) {offset = array<i64: 16>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %208 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %209 = "ckks.addcc"(%208, %205, %207) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %210 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %211 = "ckks.rotatec"(%210, %209) {offset = array<i64: 8>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %212 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %213 = "ckks.addcc"(%212, %209, %211) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %214 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %215 = "ckks.rotatec"(%214, %213) {offset = array<i64: 4>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %216 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %217 = "ckks.addcc"(%216, %213, %215) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %218 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %219 = "ckks.rotatec"(%218, %217) {offset = array<i64: 2>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %220 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %221 = "ckks.addcc"(%220, %217, %219) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %222 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %223 = "ckks.rotatec"(%222, %221) {offset = array<i64: 1>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %224 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %225 = "ckks.addcc"(%224, %221, %223) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %226 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %227 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %228 = "ckks.encode"(%227) {level = 3 : i64, scale = 30 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %229 = "ckks.mulcp"(%226, %169, %228) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %230 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %231 = "ckks.encode"(%230) {level = 3 : i64, scale = 30 : i64, value = 2 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %232 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %233 = "ckks.mulcp"(%232, %229, %231) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %234 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %235 = "ckks.rescalec"(%234, %233) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %236 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %237 = "ckks.rotatec"(%236, %235) {offset = array<i64: 2048>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %238 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %239 = "ckks.addcc"(%238, %235, %237) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %240 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %241 = "ckks.rotatec"(%240, %239) {offset = array<i64: 1024>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %242 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %243 = "ckks.addcc"(%242, %239, %241) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %244 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %245 = "ckks.rotatec"(%244, %243) {offset = array<i64: 512>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %246 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %247 = "ckks.addcc"(%246, %243, %245) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %248 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %249 = "ckks.rotatec"(%248, %247) {offset = array<i64: 256>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %250 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %251 = "ckks.addcc"(%250, %247, %249) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %252 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %253 = "ckks.rotatec"(%252, %251) {offset = array<i64: 128>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %254 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %255 = "ckks.addcc"(%254, %251, %253) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %256 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %257 = "ckks.rotatec"(%256, %255) {offset = array<i64: 64>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %258 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %259 = "ckks.addcc"(%258, %255, %257) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %260 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %261 = "ckks.rotatec"(%260, %259) {offset = array<i64: 32>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %262 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %263 = "ckks.addcc"(%262, %259, %261) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %264 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %265 = "ckks.rotatec"(%264, %263) {offset = array<i64: 16>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %266 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %267 = "ckks.addcc"(%266, %263, %265) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %268 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %269 = "ckks.rotatec"(%268, %267) {offset = array<i64: 8>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %270 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %271 = "ckks.addcc"(%270, %267, %269) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %272 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %273 = "ckks.rotatec"(%272, %271) {offset = array<i64: 4>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %274 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %275 = "ckks.addcc"(%274, %271, %273) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %276 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %277 = "ckks.rotatec"(%276, %275) {offset = array<i64: 2>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %278 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %279 = "ckks.addcc"(%278, %275, %277) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %280 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %281 = "ckks.rotatec"(%280, %279) {offset = array<i64: 1>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %282 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %283 = "ckks.addcc"(%282, %279, %281) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %284 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %285 = "ckks.encode"(%284) {level = 2 : i64, scale = 30 : i64, value = 1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %286 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %287 = "ckks.mulcp"(%286, %225, %285) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %288 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %289 = "ckks.rescalec"(%288, %287) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %290 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %291 = "ckks.mulcp"(%290, %283, %285) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %292 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %293 = "ckks.modswitchc"(%292, %151) {downFactor = 2 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %294 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %295 = "ckks.rescalec"(%294, %293) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %296 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %297 = "ckks.addcc"(%296, %295, %289) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %298 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %299 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %300 = "ckks.encode"(%299) {level = 3 : i64, scale = 30 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %301 = "ckks.mulcp"(%298, %147, %300) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %302 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %303 = "ckks.modswitchc"(%302, %301) {downFactor = 1 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %304 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %305 = "ckks.addcc"(%304, %303, %291) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %306 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %307 = "ckks.rescalec"(%306, %305) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    return %297, %307 : tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>
  }
}

