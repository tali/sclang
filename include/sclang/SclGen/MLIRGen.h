//===- MLIRGen.h - MLIR Generation from a Toy AST -------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_MLIRGEN_H_
#define SCLANG_MLIRGEN_H_

#include <memory>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace sclang {
class ModuleAST;

/// Emit IR for the given SCL moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);

/// Like mlirGen, but additionally records which module-level operations were
/// produced from which source file (key = file name from the unit Location).
/// Used to split a whole-project module into per-file modules for the build
/// cache (docs/design/multi-file-incremental.md, phase 2).
mlir::OwningOpRef<mlir::ModuleOp>
mlirGenPerFile(mlir::MLIRContext &context, ModuleAST &moduleAST,
               llvm::StringMap<llvm::SmallVector<mlir::Operation *, 4>> &opsByFile);
} // namespace sclang

#endif // SCLANG_MLIRGEN_H_
