// ProjectCache.h - per-file build cache for multi-file projects
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Content-hash cache of per-file scl-dialect MLIR modules: the Bazel
// action-cache / ccache analog for SCL projects. Key = SHA-256(file content
// + compiler/ABI tag + language mode), artifact = `<key>.mlir` (text MLIR of
// that file's module-level ops). See docs/design/multi-file-incremental.md.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_PROJECTCACHE_H
#define SCLANG_PROJECTCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace sclang::projectcache {

/// Cache key: SHA-256 over file content + compiler/ABI tag + language mode.
std::string key(llvm::StringRef content, llvm::StringRef langTag);

/// Load a cached per-file module. The parse is NOT verified (cross-file
/// symbol references only resolve after merging — verify the merged module).
/// Returns null on any error (missing/corrupt entry = cache miss).
mlir::OwningOpRef<mlir::ModuleOp> load(mlir::MLIRContext &ctx,
                                       llvm::StringRef cacheDir,
                                       llvm::StringRef key);

/// Store a per-file module as text MLIR. Returns false on IO error.
bool store(llvm::StringRef cacheDir, llvm::StringRef key, mlir::ModuleOp module);

/// True if a cache entry exists and parses.
bool exists(mlir::MLIRContext &ctx, llvm::StringRef cacheDir,
            llvm::StringRef key);

/// Extract a per-file module from a whole-project module: move the given ops
/// (recorded per file by mlirGenPerFile) into a fresh module. Ops that are no
/// longer linked into `full` (e.g. a forward declaration replaced by the real
/// definition) are skipped.
mlir::OwningOpRef<mlir::ModuleOp>
extractFileModule(mlir::ModuleOp full, llvm::ArrayRef<mlir::Operation *> ops);

/// Merge `src` into `dst` (both module ops). Colliding PRIVATE symbols in
/// src are renamed (same trick as llvm::Linker); a colliding PUBLIC symbol
/// is an error (parseInputFiles already rejects duplicate definitions).
mlir::LogicalResult mergeInto(mlir::ModuleOp dst, mlir::ModuleOp src);

} // namespace sclang::projectcache

#endif // SCLANG_PROJECTCACHE_H
