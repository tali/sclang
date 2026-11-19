// ProjectCache.cpp - per-file build cache for multi-file projects
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectCache.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include <system_error>

using namespace mlir;

namespace sclang::projectcache {

/// Compiler/ABI tag mixed into every cache key, so artifacts from a
/// different compiler build or cache format never get reused.
static std::string abiTag(llvm::StringRef langTag) {
  return std::string("sclang-cache-v1 llvm-") + LLVM_VERSION_STRING + " mode=" +
         langTag.str();
}

std::string key(llvm::StringRef content, llvm::StringRef langTag) {
  llvm::SHA256 h;
  h.update(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(content.data()), content.size()));
  std::string tag = abiTag(langTag);
  h.update(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(tag.data()), tag.size()));
  auto digest = h.final();
  static const char hexdig[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (uint8_t b : digest) {
    hex.push_back(hexdig[b >> 4]);
    hex.push_back(hexdig[b & 15]);
  }
  return hex;
}

static std::string entryPath(llvm::StringRef cacheDir, llvm::StringRef key) {
  return (llvm::Twine(cacheDir) + "/" + key + ".mlir").str();
}

OwningOpRef<ModuleOp> load(MLIRContext &ctx, llvm::StringRef cacheDir,
                           llvm::StringRef key) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(entryPath(cacheDir, key));
  if (!bufOrErr)
    return OwningOpRef<ModuleOp>();
  // No verify on parse: per-file modules may reference symbols defined in
  // other files' modules; the merged module is verified by the caller.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*bufOrErr), llvm::SMLoc());
  ParserConfig config(&ctx, /*verifyAfterParse=*/false);
  return parseSourceFile<ModuleOp>(sourceMgr, config);
}

bool exists(MLIRContext &ctx, llvm::StringRef cacheDir, llvm::StringRef key) {
  auto m = load(ctx, cacheDir, key);
  return m.get() != nullptr;
}

bool store(llvm::StringRef cacheDir, llvm::StringRef key, ModuleOp module) {
  std::error_code ec;
  llvm::sys::fs::create_directories(cacheDir, /*IgnoreExisting=*/true);
  if (ec)
    return false;
  llvm::raw_fd_ostream os(entryPath(cacheDir, key), ec, llvm::sys::fs::OF_None);
  if (ec)
    return false;
  module.print(os);
  os.flush();
  return !os.has_error();
}

OwningOpRef<ModuleOp> extractFileModule(ModuleOp full,
                                        llvm::ArrayRef<Operation *> ops) {
  auto mod = ModuleOp::create(full.getLoc());
  auto &dstOps = mod.getBody()->getOperations();
  auto &srcOps = full.getBody()->getOperations();
  for (Operation *op : ops) {
    // Skip ops that were replaced/erased after recording (forward decls).
    if (op->getParentOp() != full.getOperation())
      continue;
    dstOps.splice(dstOps.end(), srcOps, op);
  }
  return mod;
}

LogicalResult mergeInto(ModuleOp dst, ModuleOp src) {
  SymbolTable dstTable(dst);
  SymbolTable srcTable(src);
  unsigned renameCounter = 0;
  llvm::SmallVector<Operation *, 8> srcSymbols;
  for (auto &op : src.getBody()->getOperations())
    if (auto sym = dyn_cast<SymbolOpInterface>(&op))
      srcSymbols.push_back(&op);

  for (Operation *op : srcSymbols) {
    auto sym = cast<SymbolOpInterface>(op);
    auto name = sym.getNameAttr();
    Operation *existing = dstTable.lookup(name);
    if (!existing)
      continue;
    bool srcPrivate = sym.isPrivate();
    bool dstPrivate =
        cast<SymbolOpInterface>(existing).isPrivate();
    if (srcPrivate && !dstPrivate) {
      // A private declaration (e.g. a runtime stub) must yield to the public
      // definition: rename the private one and remap uses within src.
      std::string newName = (name.getValue() + "__priv" +
                             std::to_string(renameCounter++))
                                .str();
      if (failed(SymbolTable::replaceAllSymbolUses(
              name, StringAttr::get(src.getContext(), newName),
              src.getOperation())))
        return src.emitError("failed to rename private symbol '")
               << name.getValue() << "' during merge";
      sym.setName(newName);
    } else if (!srcPrivate && !dstPrivate) {
      return src.emitError("duplicate public symbol '")
             << name.getValue() << "' while merging modules";
    } else {
      // both private: rename src's to keep both
      std::string newName = (name.getValue() + "__priv" +
                             std::to_string(renameCounter++))
                                .str();
      if (failed(SymbolTable::replaceAllSymbolUses(
              name, StringAttr::get(src.getContext(), newName),
              src.getOperation())))
        return failure();
      sym.setName(newName);
    }
  }
  dst.getBody()->getOperations().splice(dst.getBody()->getOperations().end(),
                                        src.getBody()->getOperations());
  return success();
}

} // namespace sclang::projectcache
