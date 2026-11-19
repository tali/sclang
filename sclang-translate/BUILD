package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "sclang-translate",
    srcs = ["sclang-translate.cpp"],
    deps = [
        "//:AwlDialect",
        "//:SclDialect",
        "@llvm-project//mlir:MLIRTranslateLib",
        "@llvm-project//mlir:MLIRTargetLLVMIRExport",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//llvm:Support",
    ],
)
