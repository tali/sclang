//===- Printer.h - AWL (STL) text printer --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_AWL_PRINTER_H
#define SCLANG_AWL_PRINTER_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace awl {

/// Print all `awl.block` ops of the module as classic S7 AWL (STL) source
/// text (`OB 1` ... `BE`), using the S7 mnemonics the AWL frontend parses
/// (see lib/AwlGen/AwlGen.cpp), so the output round-trips through
/// `sclang-gen -x awl`. Used by `sclang-gen --emit=awl` (SCL->AWL backend,
/// docs/design/scl-to-awl.md). The `_mlir_ciface_OB1` JIT wrapper block is
/// skipped (it is re-synthesized by the frontend when OB 1 exists).
void printAwlText(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace awl
} // namespace mlir

#endif // SCLANG_AWL_PRINTER_H
