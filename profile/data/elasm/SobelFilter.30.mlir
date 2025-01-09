module @"traced/SobelFilter.mlir" {
  func.func @_hecate_SobelFilter(%arg0: tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>> attributes {arg_level = array<i64: 5>, arg_scale = array<i64: 30>, est_error = 0.1467751783975236 : f64, est_latency = 3.832710e+05 : f64, init_level = 5 : i64, no_mutation = true, res_level = array<i64: 2>, res_scale = array<i64: 84>, sm_plan_edge = array<i64: 0, 11, 20, 8, 5>, sm_plan_level = array<i64: 1, 1, 1, 1, 1>, sm_plan_scale = array<i64: 15, 0, 0, 5, 13>, smu0 = 0 : i64, smu_attached = false} {
    %0 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %1 = "ckks.modswitchc"(%0, %arg0) {downFactor = 1 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %2 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %3 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %4 = "ckks.encode"(%3) {level = 4 : i64, scale = 19 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %5 = "ckks.mulcp"(%2, %1, %4) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %6 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %7 = "ckks.rotatec"(%6, %5) {offset = array<i64: 0>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %8 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %9 = "ckks.negatec"(%8, %7) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %10 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %11 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %12 = "ckks.encode"(%11) {level = 5 : i64, scale = 49 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %13 = "ckks.mulcp"(%10, %arg0, %12) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %14 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %15 = "ckks.rotatec"(%14, %13) {offset = array<i64: 1>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %16 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %17 = "ckks.encode"(%16) {level = 5 : i64, scale = 30 : i64, value = 0 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %18 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %19 = "ckks.mulcp"(%18, %15, %17) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %20 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %21 = "ckks.rescalec"(%20, %19) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %22 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %23 = "ckks.addcc"(%22, %9, %21) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %24 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %25 = "ckks.rotatec"(%24, %5) {offset = array<i64: 2>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %26 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %27 = "ckks.negatec"(%26, %25) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %28 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %29 = "ckks.addcc"(%28, %9, %25) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %30 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %31 = "ckks.addcc"(%30, %23, %27) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %32 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %33 = "ckks.rotatec"(%32, %13) {offset = array<i64: 64>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %34 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %35 = "ckks.mulcp"(%34, %33, %17) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %36 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %37 = "ckks.rescalec"(%36, %35) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %38 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %39 = "ckks.addcc"(%38, %29, %37) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %40 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %41 = "ckks.rotatec"(%40, %13) {offset = array<i64: 66>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %42 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %43 = "ckks.encode"(%42) {level = 5 : i64, scale = 30 : i64, value = 1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %44 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %45 = "ckks.mulcp"(%44, %41, %43) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %46 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %47 = "ckks.rescalec"(%46, %45) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %48 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %49 = "ckks.addcc"(%48, %39, %47) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %50 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %51 = "ckks.rotatec"(%50, %5) {offset = array<i64: 128>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %52 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %53 = "ckks.negatec"(%52, %51) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %54 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %55 = "ckks.addcc"(%54, %49, %53) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %56 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %57 = "ckks.addcc"(%56, %31, %51) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %58 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %59 = "ckks.rotatec"(%58, %13) {offset = array<i64: 129>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %60 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %61 = "ckks.mulcp"(%60, %59, %43) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %62 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %63 = "ckks.rescalec"(%62, %61) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %64 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %65 = "ckks.addcc"(%64, %57, %63) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %66 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %67 = "ckks.rotatec"(%66, %5) {offset = array<i64: 130>} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %68 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %69 = "ckks.addcc"(%68, %55, %67) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %70 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %71 = "ckks.addcc"(%70, %65, %67) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %72 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %73 = "ckks.mulcc"(%72, %69, %69) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %74 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %75 = "ckks.rescalec"(%74, %73) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %76 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %77 = "ckks.mulcc"(%76, %71, %71) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %78 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %79 = "ckks.rescalec"(%78, %77) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %80 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %81 = "ckks.addcc"(%80, %75, %79) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %82 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %83 = "ckks.mulcc"(%82, %81, %81) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %84 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %85 = "ckks.mulcc"(%84, %83, %81) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %86 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %87 = "ckks.rescalec"(%86, %85) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %88 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %89 = "ckks.encode"(%88) {level = 2 : i64, scale = 30 : i64, value = 2 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %90 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %91 = "ckks.mulcp"(%90, %87, %89) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %92 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %93 = "ckks.encode"(%92) {level = 2 : i64, scale = 30 : i64, value = 3 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %94 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %95 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %96 = "ckks.encode"(%95) {level = 3 : i64, scale = 38 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %97 = "ckks.mulcp"(%94, %83, %96) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %98 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %99 = "ckks.rescalec"(%98, %97) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %100 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %101 = "ckks.mulcp"(%100, %99, %93) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %102 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %103 = "ckks.negatec"(%102, %101) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %104 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %105 = "ckks.addcc"(%104, %91, %103) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %106 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %107 = "ckks.encode"(%106) {level = 2 : i64, scale = 30 : i64, value = 4 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %108 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %109 = "ckks.modswitchc"(%108, %81) {downFactor = 1 : i64} : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %110 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %111 = tensor.empty() : tensor<1x!ckks.poly<1 * 0>>
    %112 = "ckks.encode"(%111) {level = 2 : i64, scale = 16 : i64, value = -1 : i64} : (tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<1 * 0>>
    %113 = "ckks.mulcp"(%110, %109, %112) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %114 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %115 = "ckks.mulcp"(%114, %113, %107) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<1 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    %116 = tensor.empty() : tensor<1x!ckks.poly<2 * 0>>
    %117 = "ckks.addcc"(%116, %105, %115) : (tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>, tensor<1x!ckks.poly<2 * 0>>) -> tensor<1x!ckks.poly<2 * 0>>
    return %117 : tensor<1x!ckks.poly<2 * 0>>
  }
}

