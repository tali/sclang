//===-- Passes.td - Scl pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_DIALECT_SCL_PASSES
#define SCLANG_DIALECT_SCL_PASSES

include "mlir/Pass/PassBase.td"

def SCFForLoopSpecialization : FunctionPass<"for-loop-specialization"> {
  let summary = "Specialize `for` loops for vectorization";
  let constructor = "mlir::createForLoopSpecializationPass()";
}

def SCFParallelLoopFusion : Pass<"parallel-loop-fusion"> {
  let summary = "Fuse adjacent parallel loops";
  let constructor = "mlir::createParallelLoopFusionPass()";
}

def SCFParallelLoopSpecialization
    : FunctionPass<"parallel-loop-specialization"> {
  let summary = "Specialize parallel loops for vectorization";
  let constructor = "mlir::createParallelLoopSpecializationPass()";
}

def SCFParallelLoopTiling : FunctionPass<"parallel-loop-tiling"> {
  let summary = "Tile parallel loops";
  let constructor = "mlir::createParallelLoopTilingPass()";
  let options =
      [ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
                  "Factors to tile parallel loops by",
                  "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">];
  let dependentDialects = ["AffineDialect"];
}

#endif // SCLANG_DIALECT_SCL_PASSES
