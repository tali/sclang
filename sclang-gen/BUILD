package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "sclang-gen",
    srcs = ["sclang-gen.cpp"],
    includes = ["include"],
    deps = [
        "//:SclDialect",
        "//:SclGen",
        "//:SclTransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CastOpInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
