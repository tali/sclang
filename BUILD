load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

td_library(
    name = "SclOpsTdFiles",
    srcs = [
        "include/sclang/SclDialect/SclBase.td",
        "include/sclang/SclDialect/SclOps.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:CastInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "SclOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "include/sclang/SclDialect/Dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/sclang/SclDialect/Dialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/sclang/SclDialect/SclOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/sclang/SclDialect/SclOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/sclang/SclDialect/SclOps.td",
    deps = [
        ":SclOpsTdFiles",
    ],
)

cc_library(
    name = "SclDialect",
    srcs = glob([
        "lib/SclDialect/*.cpp",
    ]),
    hdrs = glob([
        "include/sclang/SclDialect/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":SclOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
    ],
)

cc_library(
    name = "SclGen",
    srcs = glob([
        "lib/SclGen/*.cpp",
    ]),
    hdrs = glob([
        "include/sclang/SclGen/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":SclDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "SclTransforms",
    srcs = glob([
        "lib/SclTransforms/*.cpp",
    ]),
    hdrs = glob([
        "include/sclang/SclTransforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":SclDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
