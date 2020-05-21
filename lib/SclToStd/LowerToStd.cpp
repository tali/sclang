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

#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclToStd/Passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {

/// Return `true` if all elements are of the given type.
template <typename U> bool all_of_type(ArrayRef<Value> range) {
  return all_of(range, [](Value elem) { return elem.getType().isa<U>(); });
}

// MARK: RewritePatterns: numeric operations

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredFloatOp, typename LoweredIntegerOp>
struct NumericOpLowering : public ConversionPattern {
  NumericOpLowering(MLIRContext *ctx)
      : ConversionPattern(SclOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<mlir::NamedAttribute, 0> attrs;
    if (all_of_type<FloatType>(operands)) {
      rewriter.replaceOpWithNewOp<LoweredFloatOp>(op, operands, attrs);
      return success();
    }
    if (all_of_type<IntegerType>(operands)) {
      rewriter.replaceOpWithNewOp<LoweredIntegerOp>(op, operands, attrs);
      return success();
    }
    return failure();
  }
};
using AddOpLowering = NumericOpLowering<scl::AddOp, AddFOp, AddIOp>;
using SubOpLowering = NumericOpLowering<scl::SubOp, SubFOp, SubIOp>;
using MulOpLowering = NumericOpLowering<scl::MulOp, MulFOp, MulIOp>;
using DivOpLowering = NumericOpLowering<scl::DivOp, DivFOp, SignedDivIOp>;
using ModOpLowering = NumericOpLowering<scl::ModOp, RemFOp, SignedRemIOp>;

// MARK: RewritePatterns: compare operations

/// Lower to either a floating point or an integer comparision, depending on the
/// type.
template <typename CompareOp, CmpFPredicate predF, CmpIPredicate predI>
struct CompareOpLowering : public ConversionPattern {
  CompareOpLowering(MLIRContext *ctx)
      : ConversionPattern(CompareOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (all_of_type<FloatType>(operands)) {
      rewriter.replaceOpWithNewOp<CmpFOp>(op, predF, operands[0], operands[1]);
      return success();
    }
    if (all_of_type<IntegerType>(operands)) {
      rewriter.replaceOpWithNewOp<CmpIOp>(op, predI, operands[0], operands[1]);
      return success();
    }
    return failure();
  }
};
using EqualLowering =
    CompareOpLowering<scl::EqualOp, CmpFPredicate::UEQ, CmpIPredicate::eq>;
using NotEqualLowering =
    CompareOpLowering<scl::NotEqualOp, CmpFPredicate::UNE, CmpIPredicate::ne>;
using LessThanLowering =
    CompareOpLowering<scl::LessThanOp, CmpFPredicate::ULT, CmpIPredicate::slt>;
using LessEqualLowering =
    CompareOpLowering<scl::LessEqualOp, CmpFPredicate::ULE, CmpIPredicate::sle>;
using GreaterThanLowering =
    CompareOpLowering<scl::GreaterThanOp, CmpFPredicate::UGT,
                      CmpIPredicate::sgt>;
using GreaterEqualLowering =
    CompareOpLowering<scl::GreaterEqualOp, CmpFPredicate::UGE,
                      CmpIPredicate::sge>;

struct UnaryMinusOpLowering : public ConversionPattern {
  UnaryMinusOpLowering(MLIRContext *ctx)
      : ConversionPattern(scl::UnaryMinusOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto elementType = operands[0].getType();
    if (elementType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<NegFOp>(op, operands[0]);
      return success();
    }
    if (elementType.isa<IntegerType>()) {
      auto loc = op->getLoc();
      auto intType = elementType.dyn_cast<IntegerType>();
      auto zero = rewriter.create<ConstantIntOp>(loc, 0, intType.getWidth());
      rewriter.replaceOpWithNewOp<SubIOp>(op, zero, operands[0]);
      return success();
    }
    return failure();
  }
};

// MARK: RewritePatterns: logical conversion

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredOp>
struct LogicalOpLowering : public ConversionPattern {
  LogicalOpLowering(MLIRContext *ctx)
      : ConversionPattern(SclOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<mlir::NamedAttribute, 0> attrs;
    rewriter.replaceOpWithNewOp<LoweredOp>(op, operands, attrs);
    return success();
  }
};
using AndOpLowering = LogicalOpLowering<scl::AndOp, AndOp>;
using OrOpLowering = LogicalOpLowering<scl::OrOp, OrOp>;
using XOrOpLowering = LogicalOpLowering<scl::XOrOp, XOrOp>;

struct UnaryNotOpLowering : public OpConversionPattern<scl::UnaryNotOp> {
  using OpConversionPattern<scl::UnaryNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UnaryNotOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto falseVal = rewriter.create<ConstantIntOp>(loc, 0, 1);
    auto trueVal = rewriter.create<ConstantIntOp>(loc, 1, 1);
    rewriter.replaceOpWithNewOp<SelectOp>(op, op.rhs(), falseVal, trueVal);
    return success();
  }
};

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

struct TempVariableOpLowering : public OpRewritePattern<scl::TempVariableOp> {
  using OpRewritePattern<scl::TempVariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::TempVariableOp op,
                                PatternRewriter &rewriter) const final {

    rewriter.replaceOpWithNewOp<AllocaOp>(
        op, op.result().getType().dyn_cast<MemRefType>());
    return success();
  }
};

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

struct ReturnOpLowering : public OpRewritePattern<scl::ReturnOp> {
  using OpRewritePattern<scl::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scl::ReturnOp op,
                                PatternRewriter &rewriter) const final {

    if (op.value()) {
      auto retval = rewriter.create<LoadOp>(op.getLoc(), op.value());
      rewriter.replaceOpWithNewOp<ReturnOp>(op, retval.getResult());
    } else {
      rewriter.replaceOpWithNewOp<ReturnOp>(op);
    }
    return success();
  }
};

} // end anonymous namespace.

// MARK: SclToStdLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct SclToStdLoweringPass
    : public PassWrapper<SclToStdLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void SclToStdLoweringPass::runOnFunction() {
  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();
  target.addLegalOp<scl::IfThenElseOp>();
  target.addLegalOp<scl::EndOp>();

  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, AndOpLowering, ConstantOpLowering,
                  DivOpLowering, EqualLowering, GreaterEqualLowering,
                  GreaterThanLowering, LessEqualLowering, LessThanLowering,
                  LoadOpLowering, ModOpLowering, MulOpLowering,
                  NotEqualLowering, OrOpLowering, ReturnOpLowering,
                  SubOpLowering, StoreOpLowering, TempVariableOpLowering,
                  UnaryMinusOpLowering, UnaryNotOpLowering, XOrOpLowering>(
      &getContext());

  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
