//===- sclang-lsp-server.cpp - MLIR LSP server with scl/awl ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `sclang-lsp-server` is the standard mlir-lsp-server with the scl and awl
// dialects registered. It serves *MLIR text* documents (.mlir): hover,
// go-to-definition, references, rename, document symbols, diagnostics, ...
// come from the standard server, driven by the op interfaces our dialect ops
// implement (Symbol/SymbolTable/FunctionOpInterface/CallOpInterface).
//
// For SCL/AWL *source* files use `sclang --lsp` instead (sclang/Lsp.cpp),
// which parses the source language directly.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "sclang/AwlDialect/Dialect.h"
#include "sclang/SclDialect/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::scl::SclDialect>();
  registry.insert<mlir::awl::AwlDialect>();

  auto registryFn = [&registry](const llvm::lsp::URIForFile &)
      -> mlir::DialectRegistry & { return registry; };
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registryFn));
}
