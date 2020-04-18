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

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToStdPass();

} // end namespace toy
} // end namespace mlir

#endif // SCLANG_PASSES_H
