//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_PASSES_H
#define SCLANG_PASSES_H

#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace mlir {
class Pass;

namespace sclang {

/// Create a pass for lowering to operations in the `Std` dialect,
/// for a subset of the SCL IR (e.g. expressions).
std::unique_ptr<mlir::Pass> createLowerToStdPass();

/// Create a pass for lowering `Std` to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Create a pass for lowering the SCL dialect to the AWL dialect
/// (SCL->AWL text backend, docs/design/scl-to-awl.md); the result is printed
/// as S7 AWL text by `awl::printAwlText` (`sclang-gen --emit=awl`). With
/// `synthCiface` (used by `sclang --emit=jit-awl`), also synthesize the
/// `_mlir_ciface_OB1` awl wrapper block so the result can be JIT-run directly
/// without the AWL text round-trip.
std::unique_ptr<mlir::Pass> createLowerToAWLPass(bool synthCiface = false);

// TableGen'd pass registration: makes scl-lower-to-std / scl-lower-to-awl /
// scl-lower-to-llvm addressable by name in --pass-pipeline strings (used by
// sclang-opt and any tool that parses pipelines after calling
// registerSclangTransformsPasses()).
#define GEN_PASS_DECL
#include "sclang/SclTransforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sclang/SclTransforms/Passes.h.inc"

/// Register all sclang transform passes with the global pass registry.
inline void registerSclangTransformsPasses() {
  registerLowerToStd();
  registerLowerToAWL();
  registerLowerToLLVM();
}

} // namespace sclang
} // end namespace mlir

#endif // SCLANG_PASSES_H
