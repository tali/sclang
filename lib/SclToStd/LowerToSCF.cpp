//====- LowerToStd.cpp - Partial lowering from SCL to Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of SCF operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclToStd/Passes.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {

// MARK: EndOpLowering

struct EndOpLowering : public OpRewritePattern<scl::EndOp> {
  using OpRewritePattern<scl::EndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::EndOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
  }
};

// MARK: IfThenElseOpLowering

struct IfThenElseOpLowering : public OpRewritePattern<scl::IfThenElseOp> {
  using OpRewritePattern<scl::IfThenElseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::IfThenElseOp op,
                                PatternRewriter &rewriter) const final {

    auto loweredOp = rewriter.create<scf::IfOp>(op.getLoc(), op.cond(),
                                                 /*withElseRegion=*/true);
    // inline the block from our then body into the lowered region,
    // then remove the implicitly created one
    rewriter.inlineRegionBefore(op.thenBody(), &loweredOp.thenRegion().back());
    loweredOp.thenRegion().back().erase();

    // same for the else part
    rewriter.inlineRegionBefore(op.elseBody(), &loweredOp.elseRegion().back());
    loweredOp.elseRegion().back().erase();

    rewriter.replaceOp(op, loweredOp.getResults());
    return success();
  }
};

} // end anonymous namespace.

// MARK: SclToSCFLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct SclToSCFLoweringPass
    : public PassWrapper<SclToSCFLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void SclToSCFLoweringPass::runOnFunction() {
  ConversionTarget target(getContext());
  target.addLegalDialect<scf::SCFDialect>();

  // We want only some parts of SCL to be lowered
  target.addIllegalOp<scl::IfThenElseOp>();

  OwningRewritePatternList patterns;
  patterns.insert<IfThenElseOpLowering, EndOpLowering>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `SCF` dialect,
/// for a subset of the SCL IR (e.g. if-then-else, for-do).
std::unique_ptr<Pass> mlir::sclang::createLowerToSCFPass() {
  return std::make_unique<SclToSCFLoweringPass>();
}
