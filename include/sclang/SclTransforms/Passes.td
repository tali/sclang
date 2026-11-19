//===- Passes.td - sclang pass definitions -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen definitions for the sclang transform passes. Registering them here
// (Passes.h includes this file with GEN_PASS_DECL/GEN_PASS_REGISTRATION) makes
// them addressable by name in `--pass-pipeline` strings, e.g. in sclang-opt:
//
//   sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std)' in.mlir
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_TRANSFORMS_PASSES_TD
#define SCLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerToStd : Pass<"scl-lower-to-std", "mlir::ModuleOp"> {
  let summary = "Lower the scl/awl dialects to std dialects "
                "(func/arith/memref/scf/cf/math)";
  let description = [{
    Partial conversion of the `scl` and `awl` dialects to the std-level
    dialects. Implemented by the TUs LowerToStd{,Util,Scl,Awl}.cpp sharing
    LowerToStdInternal.h (pure code motion, one pass).
  }];
  let constructor = "mlir::sclang::createLowerToStdPass()";
  let dependentDialects = ["mlir::scl::SclDialect", "mlir::awl::AwlDialect",
    "mlir::func::FuncDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect", "mlir::math::MathDialect"];
}

def LowerToAWL : Pass<"scl-lower-to-awl", "mlir::ModuleOp"> {
  let summary = "Lower the scl dialect to the awl dialect (SCL->AWL, "
                "docs/design/scl-to-awl.md)";
  let description = [{
    In-memory SCL->AWL lowering; the result can be printed as S7 AWL text
    (`sclang-gen --emit=awl`) or consumed by the std lowering directly
    (`sclang --emit=awl-mlir` / `--emit=jit-awl`). The registered pipeline
    entry uses the default `synthCiface=false`; `sclang --emit=jit-awl`
    calls createLowerToAWLPass(true) directly to also synthesize the
    `_mlir_ciface_OB1` JIT wrapper.
  }];
  let constructor = "mlir::sclang::createLowerToAWLPass()";
  let dependentDialects = ["mlir::scl::SclDialect", "mlir::awl::AwlDialect",
    "mlir::func::FuncDialect", "mlir::cf::ControlFlowDialect"];
}

def LowerToLLVM : Pass<"scl-lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower std dialects (func/scf/memref/arith/cf/math) to the "
                "LLVM dialect";
  let description = [{
    Final MLIR lowering to the LLVM dialect (lib/SclTransforms/LowerToLLVM.cpp),
    after which the module can be translated to LLVM IR and JIT-compiled or
    emitted as a host object file.
  }];
  let constructor = "mlir::sclang::createLowerToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::math::MathDialect"];
}

#endif // SCLANG_TRANSFORMS_PASSES_TD
