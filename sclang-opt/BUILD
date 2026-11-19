package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "sclang-opt",
    srcs = ["sclang-opt.cpp"],
    deps = [
        "//:AwlDialect",
        "//:SclDialect",
        "//:SclTransforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:Support",
    ],
)
