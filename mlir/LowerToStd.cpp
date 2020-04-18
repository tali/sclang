//====- LowerToStd.cpp - Partial lowering from SCL to Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "sclang/Dialect.h"
#include "sclang/Passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;


namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LoweredBinaryOp>(op, operands[0], operands[1]);
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<scl::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<scl::MulOp, MulFOp>;

template <typename CompareOp, mlir::CmpFPredicate predicate>
struct CompareOpLowering: public ConversionPattern {
  CompareOpLowering(MLIRContext *ctx)
      : ConversionPattern(CompareOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<CmpFOp>(op, predicate, operands[0], operands[1]);
    return success();
  }
};
using LessThanLowering = CompareOpLowering<scl::LessThanOp, mlir::CmpFPredicate::ULT>;
using GreaterThanLowering = CompareOpLowering<scl::GreaterThanOp, mlir::CmpFPredicate::UGT>;
using EqualLowering = CompareOpLowering<scl::EqualOp, mlir::CmpFPredicate::UEQ>;


template <typename BinaryOp, typename LoweredBinaryOp>
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LoweredBinaryOp>(op, operands[0]);
    return success();
  }
};
using UnaryMinusOpLowering = UnaryOpLowering<scl::UnaryMinusOp, NegFOp>;


// MARK: ConstantOpLowering

struct ConstantOpLowering : public OpRewritePattern<scl::ConstantOp> {
using OpRewritePattern<scl::ConstantOp>::OpRewritePattern;

LogicalResult matchAndRewrite(scl::ConstantOp op,
                              PatternRewriter &rewriter) const final {

    // directly lower to Std
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return success();
  }
};


// MARK: Load / Store

struct LoadOpLowering : public OpRewritePattern<scl::LoadOp> {
  using OpRewritePattern<scl::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::LoadOp op,
                                PatternRewriter &rewriter) const final {

    rewriter.replaceOpWithNewOp<LoadOp>(op, op.address());
    return success();
  }
};

struct StoreOpLowering : public OpRewritePattern<scl::StoreOp> {
  using OpRewritePattern<scl::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::StoreOp op,
                                PatternRewriter &rewriter) const final {

    rewriter.replaceOpWithNewOp<StoreOp>(op, op.rhs(), op.lhs());
    return success();
  }
};

} // end anonymous namespace.


// MARK: SclToStdLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct SclToStdLoweringPass : public PassWrapper<SclToStdLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void SclToStdLoweringPass::runOnFunction() {
  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();

  // We want all of SCL to be lowered
  target.addIllegalDialect<scl::SclDialect>();
  // TBD: these are not finished yeet:
  target.addLegalOp<scl::TempVariableOp>();
  target.addLegalOp<scl::IfThenElseOp>();
  target.addLegalOp<scl::ReturnOp>();
  target.addLegalOp<scl::TerminatorOp>();

  OwningRewritePatternList patterns;
  patterns.insert<
    AddOpLowering,
    ConstantOpLowering,
    EqualLowering,
    GreaterThanLowering,
    LessThanLowering,
    LoadOpLowering,
    MulOpLowering,
//    ReturnOpLowering,
    StoreOpLowering,
    UnaryMinusOpLowering
  >(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
