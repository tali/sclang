package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "sclang-lsp-server",
    srcs = ["sclang-lsp-server.cpp"],
    deps = [
        "//:AwlDialect",
        "//:SclDialect",
        "@llvm-project//mlir:MLIRLspServerLib",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllExtensions",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//llvm:SupportLSP",
    ],
)
