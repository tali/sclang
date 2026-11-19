//===- sclang-opt.cpp - sclang MLIR optimizer driver ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `sclang-opt` is an mlir-opt-style driver that additionally knows the scl and
// awl dialects and the sclang lowering passes, so they can be exercised from
// --pass-pipeline strings on plain .mlir files, e.g.:
//
//   sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std)' in.mlir
//   sclang-opt --pass-pipeline='builtin.module(scl-lower-to-awl,scl-lower-to-std,scl-lower-to-llvm)' in.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "sclang/AwlDialect/Dialect.h"
#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclTransforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::sclang::registerSclangTransformsPasses();

  registry.insert<mlir::scl::SclDialect>();
  registry.insert<mlir::awl::AwlDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv,
                        "sclang MLIR optimizer (scl/awl dialects + sclang "
                        "lowering passes)\n",
                        registry));
}
