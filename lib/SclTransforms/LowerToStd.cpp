//====- LowerToStd.cpp - Partial lowering from SCL to Std + SCF          --===//
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
#include "sclang/SclTransforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Return `true` if all elements are of the given type.
template <typename U> bool all_of_type(ValueRange range) {
  return llvm::all_of(range, [](Value elem) { return elem.getType().isa<U>(); });
}

// MARK: SclTypeConverter

class SclTypeConverter : public TypeConverter {
public:
  SclTypeConverter() {
    addConversion([&](scl::AddressType type) {
      return mlir::MemRefType::get({}, convertType(type.getElementType()));
    });
    addConversion([&](scl::IntegerType type) {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([&](scl::LogicalType type) {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([&](scl::RealType type) {
      return FloatType::getF32(type.getContext());
    });

    // keep structs and arrays
    // addConversion([&](scl::StructType type) { return type; });

    // keep Std types
    addConversion([&](FloatType type) { return type; });
    addConversion([&](IntegerType type) { return type; });

    // Add generic source and target materializations to handle cases where
    // SCL types persist after conversion to Std.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Optional<Value> {
      if (inputs.size() != 1)
        return None;
      auto op = builder.create<scl::DialectCastOp>(loc, resultType, inputs[0]);
      return op.getResult();
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Optional<Value> {
      if (inputs.size() != 1)
        return None;
      return builder.create<scl::DialectCastOp>(loc, resultType, inputs[0])
          .getResult();
    });
  }
};

// MARK: CallFcOpLowering

struct CallFcOpLowering : public OpConversionPattern<scl::CallFcOp> {
  using OpConversionPattern<scl::CallFcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::CallFcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Type returnType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<CallOp>(op, op.callee(), returnType,
                                        adaptor.arguments());
    return success();
  }
};

// MARK: CompareOpLowering

/// Lower to either a floating point or an integer comparision, depending on the
/// type.
template <typename CompareOp, arith::CmpFPredicate predF,
          arith::CmpIPredicate predI>
struct CompareOpLowering : public OpConversionPattern<CompareOp> {
  using OpConversionPattern<CompareOp>::OpConversionPattern;
  using OpAdaptor = typename CompareOp::Adaptor;

  LogicalResult
  matchAndRewrite(CompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (all_of_type<FloatType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, predF, adaptor.lhs(),
                                                 adaptor.rhs());
      return success();
    }
    if (all_of_type<IntegerType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, predI, adaptor.lhs(),
                                                 adaptor.rhs());
      return success();
    }
    return failure();
  }
};
using EqualLowering = CompareOpLowering<scl::EqualOp, arith::CmpFPredicate::UEQ,
                                        arith::CmpIPredicate::eq>;
using NotEqualLowering =
    CompareOpLowering<scl::NotEqualOp, arith::CmpFPredicate::UNE,
                      arith::CmpIPredicate::ne>;
using LessThanLowering =
    CompareOpLowering<scl::LessThanOp, arith::CmpFPredicate::ULT,
                      arith::CmpIPredicate::slt>;
using LessEqualLowering =
    CompareOpLowering<scl::LessEqualOp, arith::CmpFPredicate::ULE,
                      arith::CmpIPredicate::sle>;
using GreaterThanLowering =
    CompareOpLowering<scl::GreaterThanOp, arith::CmpFPredicate::UGT,
                      arith::CmpIPredicate::sgt>;
using GreaterEqualLowering =
    CompareOpLowering<scl::GreaterEqualOp, arith::CmpFPredicate::UGE,
                      arith::CmpIPredicate::sge>;

// MARK: ConstantOpLowering

struct ConstantOpLowering : public OpConversionPattern<scl::ConstantOp> {
  using OpConversionPattern<scl::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.value());
    return success();
  }
};

// MARK: DialectCastOpLowering

/// Removes DialectCastOps when they are not necessary any more.
struct DialectCastOpLowering : public OpConversionPattern<scl::DialectCastOp> {
  using OpConversionPattern<scl::DialectCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DialectCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value value = op.value();

    if (value.getType() != typeConverter->convertType(op.getType())) {
      return failure();
    }
    rewriter.replaceOp(op, value);
    return success();
  }
};

// MARK: EndOpLowering

struct EndOpLowering : public OpConversionPattern<scl::EndOp> {
  using OpConversionPattern<scl::EndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::EndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
  }
};

// MARK: FunctionOpLowering

struct FunctionOpLowering : public OpConversionPattern<scl::FunctionOp> {
  using OpConversionPattern<scl::FunctionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::FunctionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    StringRef name = op.getName();
    auto sclType = op.getType();

    // convert argument types
    TypeConverter::SignatureConversion signatureConverter(
        sclType.getNumInputs());
    for (auto argType : enumerate(sclType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }
    auto inputs = signatureConverter.getConvertedTypes();

    // convert result type
    SmallVector<Type, 4> results;
    if (failed(typeConverter->convertTypes(sclType.getResults(), results)))
      return failure();

    // Create the converted func op.
    auto newFuncType = rewriter.getFunctionType(inputs, results);
    auto newFuncOp = rewriter.create<FuncOp>(loc, name, newFuncType);

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// MARK: IfThenElseOpLowering

struct IfThenElseOpLowering : public OpConversionPattern<scl::IfThenElseOp> {
  using OpConversionPattern<scl::IfThenElseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::IfThenElseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto loweredOp = rewriter.create<scf::IfOp>(op.getLoc(), adaptor.cond(),
                                                /*withElseRegion=*/true);
    // inline the block from our then body into the lowered region,
    // then remove the implicitly created one
    rewriter.inlineRegionBefore(op.thenBody(), &loweredOp.getThenRegion().back());
    rewriter.eraseBlock(&loweredOp.getThenRegion().back());

    // same for the else part
    rewriter.inlineRegionBefore(op.elseBody(), &loweredOp.getElseRegion().back());
    rewriter.eraseBlock(&loweredOp.getElseRegion().back());

    rewriter.replaceOp(op, loweredOp.getResults());
    return success();
  }
};

// MARK: LoadOpLowering

struct LoadOpLowering : public OpConversionPattern<scl::LoadOp> {
  using OpConversionPattern<scl::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    scl::LoadOp::Adaptor transformed(adaptor.getOperands());

    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, transformed.address());
    return success();
  }
};

// MARK: LogicalOpLowering

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredOp>
struct LogicalOpLowering : public OpConversionPattern<SclOp> {
  using OpConversionPattern<SclOp>::OpConversionPattern;
  using OpAdaptor = typename SclOp::Adaptor;

  LogicalResult
  matchAndRewrite(SclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<mlir::NamedAttribute, 0> attrs;

    rewriter.replaceOpWithNewOp<LoweredOp>(op, adaptor.getOperands(), attrs);
    return success();
  }
};
using AndOpLowering = LogicalOpLowering<scl::AndOp, arith::AndIOp>;
using OrOpLowering = LogicalOpLowering<scl::OrOp, arith::OrIOp>;
using XOrOpLowering = LogicalOpLowering<scl::XOrOp, arith::XOrIOp>;

// MARK: NumericOpLowering

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredFloatOp, typename LoweredIntegerOp>
struct NumericOpLowering : public OpConversionPattern<SclOp> {
  using OpConversionPattern<SclOp>::OpConversionPattern;
  using OpAdaptor = typename SclOp::Adaptor;

  LogicalResult
  matchAndRewrite(SclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<mlir::NamedAttribute, 0> attrs;

    if (all_of_type<FloatType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<LoweredFloatOp>(op, adaptor.getOperands(), attrs);
      return success();
    }
    if (all_of_type<IntegerType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<LoweredIntegerOp>(op, adaptor.getOperands(), attrs);
      return success();
    }
    return failure();
  }
};
using AddOpLowering =
    NumericOpLowering<scl::AddOp, arith::AddFOp, arith::AddIOp>;
using SubOpLowering =
    NumericOpLowering<scl::SubOp, arith::SubFOp, arith::SubIOp>;
using MulOpLowering =
    NumericOpLowering<scl::MulOp, arith::MulFOp, arith::MulIOp>;
using DivOpLowering =
    NumericOpLowering<scl::DivOp, arith::DivFOp, arith::DivSIOp>;
using ModOpLowering =
    NumericOpLowering<scl::ModOp, arith::RemFOp, arith::RemSIOp>;

// MARK: ReturnOpLowering

struct ReturnOpLowering : public OpConversionPattern<scl::ReturnOp> {
  using OpConversionPattern<scl::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};

// MARK: ReturnValueOpLowering

struct ReturnValueOpLowering : public OpConversionPattern<scl::ReturnValueOp> {
  using OpConversionPattern<scl::ReturnValueOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ReturnValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    scl::ReturnValueOp::Adaptor transformed(adaptor.getOperands());

    auto retval =
        rewriter.create<memref::LoadOp>(op.getLoc(), transformed.value());

    rewriter.replaceOpWithNewOp<ReturnOp>(op, retval.getResult());
    return success();
  }
};

// MARK: StoreOpLowering

struct StoreOpLowering : public OpConversionPattern<scl::StoreOp> {
  using OpConversionPattern<scl::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    scl::StoreOp::Adaptor transformed(adaptor.getOperands());

    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, transformed.rhs(),
                                                 transformed.lhs());
    return success();
  }
};

// MARK: TempVariableOpLowering

struct TempVariableOpLowering
    : public OpConversionPattern<scl::TempVariableOp> {
  using OpConversionPattern<scl::TempVariableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::TempVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resultType = getTypeConverter()->convertType(op.result().getType());
    auto memref = resultType.dyn_cast<MemRefType>();

    rewriter.replaceOpWithNewOp<memref::AllocaOp>(op, memref);
    return success();
  }
};

// MARK: UnaryMinusOpLowering

struct UnaryMinusOpLowering : public OpConversionPattern<scl::UnaryMinusOp> {
  using OpConversionPattern<scl::UnaryMinusOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UnaryMinusOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    scl::UnaryMinusOp::Adaptor transformed(adaptor.getOperands());

    auto elementType = transformed.rhs().getType();
    return TypeSwitch<Type, LogicalResult>(elementType)
        .Case<FloatType>([&](auto elementType) {
          rewriter.replaceOpWithNewOp<arith::NegFOp>(op, transformed.rhs());
          return success();
        })
        .Case<IntegerType>([&](auto elementType) {
          auto loc = op.getLoc();
          auto zero = rewriter.create<arith::ConstantIntOp>(
              loc, 0, elementType.getWidth());
          rewriter.replaceOpWithNewOp<arith::SubIOp>(op, zero,
                                                     transformed.rhs());
          return success();
        })
        .Default([&](auto elementType) {
          emitError(op.getLoc())
              << "invalid type" << elementType << "for UnaryMinusOp";
          return failure();
        });
  }
};

// MARK: UnaryNotOpLowering

struct UnaryNotOpLowering : public OpConversionPattern<scl::UnaryNotOp> {
  using OpConversionPattern<scl::UnaryNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UnaryNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    scl::UnaryNotOp::Adaptor transformed(adaptor.getOperands());
    auto falseVal = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
    auto trueVal = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    rewriter.replaceOpWithNewOp<SelectOp>(op, transformed.rhs(), falseVal,
                                          trueVal);
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
    : public PassWrapper<SclToStdLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<arith::ArithmeticDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace.

void SclToStdLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(*context);
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalOp<FuncOp>();
  target.addLegalOp<scl::DebugPrintOp>();

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();

  SclTypeConverter converter{};

  RewritePatternSet patterns(context);
  patterns
      .add<AddOpLowering, AndOpLowering, CallFcOpLowering, ConstantOpLowering,
           DialectCastOpLowering, DivOpLowering, EndOpLowering, EqualLowering,
           FunctionOpLowering, GreaterEqualLowering, GreaterThanLowering,
           IfThenElseOpLowering, LessEqualLowering, LessThanLowering,
           LoadOpLowering, ModOpLowering, MulOpLowering, NotEqualLowering,
           OrOpLowering, ReturnOpLowering, ReturnValueOpLowering, SubOpLowering,
           StoreOpLowering, TempVariableOpLowering, UnaryMinusOpLowering,
           UnaryNotOpLowering, XOrOpLowering>(converter, context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
