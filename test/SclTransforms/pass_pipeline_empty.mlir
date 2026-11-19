// RUN: sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std,scl-lower-to-llvm)' %s | FileCheck %s

// Regression test (fuzz_invalid --mlir found this): an EMPTY module must not
// crash the std lowering. The pass creates scl types itself (builtin FB
// layouts in forceBuiltinFbLayouts), so the scl dialect must be listed in
// getDependentDialects — otherwise, with lazy dialect loading (sclang-opt),
// the dialect is not in the context and type uniquing crashes in builds
// without assertions.

// CHECK: module
module {
}
