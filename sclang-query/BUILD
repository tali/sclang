package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "sclang-query",
    srcs = ["sclang-query.cpp"],
    deps = [
        "//:AwlDialect",
        "//:SclDialect",
        "@llvm-project//mlir:MlirQueryLib",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:Support",
    ],
)
