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

#include <memory>

namespace mlir {
class Pass;

namespace sclang {

/// Create a pass for lowering to operations in the `Std` dialect,
/// for a subset of the SCL IR (e.g. expressions).
std::unique_ptr<mlir::Pass> createLowerToStdPass();

/// Create a pass for lowering to operations in the `Loop` dialect,
/// for a subset of the SCL IR (e.g. if-then-else, for-do).
std::unique_ptr<mlir::Pass> createLowerToLoopPass();

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace sclang
} // end namespace mlir

#endif // SCLANG_PASSES_H
