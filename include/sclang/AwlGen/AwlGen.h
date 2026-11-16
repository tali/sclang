//===- AwlGen.h - AWL (STL) frontend entry point --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_AWLGEN_AWLGEN_H_
#define SCLANG_AWLGEN_AWLGEN_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace sclang {

/// Parse and codegen an AWL (STL) source buffer into a module of `awl`
/// dialect ops (ADR 003). Returns null on error.
mlir::OwningOpRef<mlir::ModuleOp> awlGen(mlir::MLIRContext &context,
                                         llvm::StringRef source,
                                         llvm::StringRef filename);

/// Register a named DB from an external symbol table (e.g. an awlsim
/// .awlpro "Symbol table": `126,Data DB 42`), so that P#"Data".VAR and
/// DATA_BLOCK "Data" resolve to that DB number. Call before awlGen().
void awlRegisterNamedDb(llvm::StringRef name, int32_t db);

} // namespace sclang

#endif // SCLANG_AWLGEN_AWLGEN_H_
