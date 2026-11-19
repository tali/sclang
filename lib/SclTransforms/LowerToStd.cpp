//====- LowerToStd.cpp - Partial lowering from SCL/AWL to Std + SCF -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LowerToStd pass: partial conversion of the scl and awl dialects to
// std/scf/memref/func/cf. The conversion patterns and shared helpers live in
// LowerToStdScl.cpp / LowerToStdAwl.cpp / LowerToStdUtil.cpp (declared in
// LowerToStdInternal.h) so the TUs compile in parallel; this file only holds
// the pass itself. See AGENTS.md ("LowerToStd split").
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"
#include "sclang/AwlDialect/AwlOps.h"
#include "sclang/SclTransforms/Passes.h"
#include "LowerToStdInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::sclang::lowering;

// MARK: SclToStdLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct SclToStdLoweringPass
    : public PassWrapper<SclToStdLoweringPass, OperationPass<ModuleOp>> {
  // Registered as "scl-lower-to-std" (Passes.td) so it is usable in
  // --pass-pipeline strings.
  llvm::StringRef getArgument() const override { return "scl-lower-to-std"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    // The pass creates scl types itself (builtin FB layouts), so the scl
    // dialect must be loaded even if the input module contains no scl ops
    // yet (otherwise type uniquing crashes in builds without assertions).
    registry.insert<scl::SclDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<awl::AwlDialect>();
  }
  void runOnOperation() final;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SclToStdLoweringPass)
};
} // end anonymous namespace.

void SclToStdLoweringPass::runOnOperation() {
  MLIRContext * context = &getContext();

  ConversionTarget target(*context);
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<math::MathDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<cf::ControlFlowDialect>();
  target.addLegalOp<func::FuncOp>();
  // Legal only once its optional value operand is a std type (converted by
  // DebugPrintOpConversion); the LLVM-stage lowering then emits the runtime
  // call (scl_rt_debug_print*).
  target.addDynamicallyLegalOp<scl::DebugPrintOp>(
      [](scl::DebugPrintOp op) {
        return !op.getValue() ||
               !llvm::isa<scl::SclDialect>(op.getValue().getType().getDialect());
      });

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();
  // ... and all of AWL (parallel source dialect, ADR 003):
  target.addIllegalDialect<awl::AwlDialect>();

  // Compute FB instance layouts before conversion (the ops are erased).
  SclTypeConverter converter{computeFbLayouts(getOperation())};

  RewritePatternSet patterns(context);
  auto loopTargets = std::make_shared<LoopTerminatorTargets>();
  // The pattern sets live in their own TUs (LowerToStdScl/LowerToStdAwl.cpp);
  // the split is pure code motion, the pass behavior is unchanged.
  populateSclToStdPatterns(patterns, converter, context, loopTargets);
  populateAwlToStdPatterns(patterns, converter, context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
