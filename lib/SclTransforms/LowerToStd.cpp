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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using llvm::all_of;
using namespace mlir::arith;
using namespace mlir::func;

namespace {

/// Return `true` if all elements are of the given type.
template <typename U> bool all_of_type(ValueRange range) {
  return all_of(range, [](Value elem) { return elem.getType().isa<U>(); });
}

// MARK: SclTypeConverter

class SclTypeConverter : public TypeConverter {
public:
  SclTypeConverter() {
    addConversion([&](scl::AddressType type) {
      auto elem = convertType(type.getElementType());
      if (!elem)
        return Type();
      // 0-d memref with dynamic offset, so that array element addresses
      // produced by memref.reinterpret_cast have the same type.
      return (Type)mlir::MemRefType::get(
          {}, elem,
          StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic, {}));
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

    addConversion([&](scl::ArrayType type) {
      SmallVector<int64_t, 2> shape;
      for (auto &dim : type.getDimensions())
        shape.push_back(dim.second - dim.first + 1);
      auto elem = convertType(type.getElementType());
      if (!elem)
        return Type();
      return (Type)MemRefType::get(shape, elem);
    });

    // keep structs
    //addConversion([&](scl::StructType type) { return type; });

    // keep Std types
    addConversion([&](FloatType type) { return type; });
    addConversion([&](IntegerType type) { return type; });

    // Add generic source and target materializations to handle cases where
    // SCL types persist after conversion to Std.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;
      auto op = builder.create<scl::DialectCastOp>(loc, resultType, inputs[0]);
      return op.getResult();
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;
      return builder.create<scl::DialectCastOp>(loc, resultType, inputs[0])
          .getResult();
    });
  }
};

// MARK: CallFcOpLowering

struct CallFcOpLowering : public OpConversionPattern<scl::CallFcOp> {
  using OpConversionPattern<scl::CallFcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::CallFcOp op, scl::CallFcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type returnType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(), returnType,
                                        adaptor.getArguments());
    return success();
  }
};

// MARK: CompareOpLowering

/// Lower to either a floating point or an integer comparision, depending on the
/// type.
template <typename CompareOp, CmpFPredicate predF, CmpIPredicate predI>
struct CompareOpLowering : public OpConversionPattern<CompareOp> {
  using OpConversionPattern<CompareOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompareOp op, typename CompareOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (all_of_type<FloatType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<CmpFOp>(op, predF, adaptor.getLhs(),
                                          adaptor.getRhs());
      return success();
    }
    if (all_of_type<IntegerType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<CmpIOp>(op, predI, adaptor.getLhs(),
                                          adaptor.getRhs());
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

// MARK: ConstantOpLowering

struct ConstantOpLowering : public OpConversionPattern<scl::ConstantOp> {
  using OpConversionPattern<scl::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ConstantOp op, scl::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
        op.getValue().cast<TypedAttr>());
    return success();
  }
};

// MARK: DialectCastOpLowering

/// Removes DialectCastOps when they are not necessary any more.
struct DialectCastOpLowering : public OpConversionPattern<scl::DialectCastOp> {
  using OpConversionPattern<scl::DialectCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DialectCastOp op, scl::DialectCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();

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
  matchAndRewrite(scl::EndOp op, scl::EndOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
  }
};

// MARK: FunctionOpLowering

struct FunctionOpLowering : public OpConversionPattern<scl::FunctionOp> {
  using OpConversionPattern<scl::FunctionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::FunctionOp op, scl::FunctionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef name = op.getName();
    auto sclType = op.getFunctionType();

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
                                newFuncOp.getBody().end());
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
  matchAndRewrite(scl::IfThenElseOp op, scl::IfThenElseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loweredOp = rewriter.create<scf::IfOp>(op.getLoc(), adaptor.getCond(),
                                                /*withElseRegion=*/true);
    // inline the block from our then body into the lowered region,
    // then remove the implicitly created one
    rewriter.inlineRegionBefore(op.getThenBody(),
                                &loweredOp.getThenRegion().back());
    rewriter.eraseBlock(&loweredOp.getThenRegion().back());

    // same for the else part
    rewriter.inlineRegionBefore(op.getElseBody(),
                                &loweredOp.getElseRegion().back());
    rewriter.eraseBlock(&loweredOp.getElseRegion().back());

    rewriter.replaceOp(op, loweredOp.getResults());
    return success();
  }
};

// MARK: LoadOpLowering

struct LoadOpLowering : public OpConversionPattern<scl::LoadOp> {
  using OpConversionPattern<scl::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::LoadOp op, scl::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getAddress());
    return success();
  }
};

// MARK: LogicalOpLowering

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredOp>
struct LogicalOpLowering : public OpConversionPattern<SclOp> {
  using OpConversionPattern<SclOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SclOp op, typename SclOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LoweredOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};
using AndOpLowering = LogicalOpLowering<scl::AndOp, AndIOp>;
using OrOpLowering = LogicalOpLowering<scl::OrOp, OrIOp>;
using XOrOpLowering = LogicalOpLowering<scl::XOrOp, XOrIOp>;

// MARK: NumericOpLowering

/// Lower to either a floating point or an integer operation, depending on the
/// type.
template <typename SclOp, typename LoweredFloatOp, typename LoweredIntegerOp>
struct NumericOpLowering : public OpConversionPattern<SclOp> {
  using OpConversionPattern<SclOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SclOp op, typename SclOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    if (all_of_type<FloatType>(operands)) {
      rewriter.replaceOpWithNewOp<LoweredFloatOp>(op, operands[0], operands[1]);
      return success();
    }
    if (all_of_type<IntegerType>(operands)) {
      rewriter.replaceOpWithNewOp<LoweredIntegerOp>(op, operands[0], operands[1]);
      return success();
    }
    return failure();
  }
};
using AddOpLowering = NumericOpLowering<scl::AddOp, AddFOp, AddIOp>;
using SubOpLowering = NumericOpLowering<scl::SubOp, SubFOp, SubIOp>;
using MulOpLowering = NumericOpLowering<scl::MulOp, MulFOp, MulIOp>;
using DivOpLowering = NumericOpLowering<scl::DivOp, DivFOp, DivSIOp>;
using ModOpLowering = NumericOpLowering<scl::ModOp, RemFOp, RemSIOp>;

// MARK: ReturnOpLowering

struct ReturnOpLowering : public OpConversionPattern<scl::ReturnOp> {
  using OpConversionPattern<scl::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ReturnOp op, scl::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

// MARK: ReturnValueOpLowering

struct ReturnValueOpLowering : public OpConversionPattern<scl::ReturnValueOp> {
  using OpConversionPattern<scl::ReturnValueOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ReturnValueOp op, scl::ReturnValueOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto retval = rewriter.create<memref::LoadOp>(op.getLoc(), adaptor.getValue());

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, retval.getResult());
    return success();
  }
};

// MARK: StoreOpLowering

struct StoreOpLowering : public OpConversionPattern<scl::StoreOp> {
  using OpConversionPattern<scl::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::StoreOp op, scl::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getRhs(),
                                                 adaptor.getLhs());
    return success();
  }
};

// MARK: TempVariableOpLowering

struct TempVariableOpLowering
    : public OpConversionPattern<scl::TempVariableOp> {
  using OpConversionPattern<scl::TempVariableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::TempVariableOp op, scl::TempVariableOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto memref = resultType.dyn_cast<MemRefType>();
    if (!memref)
      return failure();
    // alloca requires an identity layout; reinterpret to the address type.
    auto plain = MemRefType::get({}, memref.getElementType());
    auto alloca = rewriter.create<memref::AllocaOp>(op.getLoc(), plain);
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memref, alloca, zero, ValueRange{}, ValueRange{});
    return success();
  }
};

// MARK: UnaryMinusOpLowering

struct UnaryMinusOpLowering
    : public OpConversionPattern<scl::UnaryMinusOp> {
  using OpConversionPattern<scl::UnaryMinusOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UnaryMinusOp op, scl::UnaryMinusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elementType = adaptor.getRhs().getType();
    return TypeSwitch<Type, LogicalResult>(elementType)
        .Case<FloatType>([&](auto elementType) {
          rewriter.replaceOpWithNewOp<NegFOp>(op, adaptor.getRhs());
          return success();
        })
        .Case<IntegerType>([&](auto elementType) {
          auto loc = op.getLoc();
          auto zero = rewriter.create<ConstantIntOp>(loc, 0, elementType);
          rewriter.replaceOpWithNewOp<SubIOp>(op, zero, adaptor.getRhs());
          return success();
        })
        .Default([&](auto elementType) {
          emitError(op.getLoc()) << "invalid type" << elementType << "for UnaryMinusOp";
          return failure();
        });
  }
};

// MARK: UnaryNotOpLowering

struct UnaryNotOpLowering : public OpConversionPattern<scl::UnaryNotOp> {
  using OpConversionPattern<scl::UnaryNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UnaryNotOp op, scl::UnaryNotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i1 = rewriter.getIntegerType(1);
    auto falseVal = rewriter.create<ConstantIntOp>(loc, 0, i1);
    auto trueVal = rewriter.create<ConstantIntOp>(loc, 1, i1);
    rewriter.replaceOpWithNewOp<SelectOp>(op, adaptor.getRhs(), falseVal,
                                          trueVal);
    return success();
  }
};


// MARK: DataBlockOpLowering

/// Lower a data block declaration to a global 0-d memref.
struct DataBlockOpLowering : public OpConversionPattern<scl::DataBlockOp> {
  using OpConversionPattern<scl::DataBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DataBlockOp op, scl::DataBlockOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elemTy = getTypeConverter()->convertType(op.getType());
    if (!elemTy)
      return failure();
    auto memrefTy = MemRefType::get(
        {}, elemTy,
        StridedLayoutAttr::get(op.getContext(), ShapedType::kDynamic, {}));
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymNameAttr(), /*sym_visibility=*/nullptr,
        TypeAttr::get(memrefTy), /*initial_value=*/nullptr,
        /*constant=*/nullptr, /*alignment=*/nullptr);
    return success();
  }
};

// MARK: GetGlobalOpLowering

struct GetGlobalOpLowering : public OpConversionPattern<scl::GetGlobalOp> {
  using OpConversionPattern<scl::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetGlobalOp op, scl::GetGlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, resTy,
                                                     op.getNameAttr());
    return success();
  }
};

// MARK: GetIndexOpLowering

/// Lower array element addressing to a flat offset into the memref.
struct GetIndexOpLowering : public OpConversionPattern<scl::GetIndexOp> {
  using OpConversionPattern<scl::GetIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetIndexOp op, scl::GetIndexOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto arrayTy = op.getAddr().getType()
                       .cast<scl::AddressType>()
                       .getElementType()
                       .cast<scl::ArrayType>();
    auto dims = arrayTy.getDimensions();
    auto indices = adaptor.getIndices();
    if (indices.size() != dims.size())
      return failure();

    // row-major strides
    SmallVector<int64_t, 2> strides(dims.size());
    int64_t s = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= dims[i].second - dims[i].first + 1;
    }

    // Compute the flat offset in index type (row-major).
    auto indexTy = rewriter.getIndexType();
    Value flat;
    for (size_t i = 0; i < dims.size(); ++i) {
      Value idx = indices[i];
      if (!idx.getType().isa<IntegerType, IndexType>())
        return failure();
      Value adj = idx.getType().isa<IndexType>()
                      ? idx
                      : (Value)rewriter.create<arith::IndexCastOp>(loc, indexTy, idx);
      if (dims[i].first != 0) {
        auto minC = rewriter.create<arith::ConstantIndexOp>(loc, dims[i].first);
        adj = rewriter.create<arith::SubIOp>(loc, adj, minC);
      }
      Value contrib = adj;
      if (strides[i] != 1) {
        auto strideC =
            rewriter.create<arith::ConstantIndexOp>(loc, strides[i]);
        contrib = rewriter.create<arith::MulIOp>(loc, adj, strideC);
      }
      flat = flat ? rewriter.create<arith::AddIOp>(loc, flat, contrib).getResult()
                  : contrib;
    }

    auto elemTy = getTypeConverter()->convertType(arrayTy.getElementType());
    if (!elemTy)
      return failure();
    // The address of an array is a 0-d memref of the array memref; load it.
    Value base = adaptor.getAddr();
    auto addrTy = base.getType().cast<MemRefType>();
    if (addrTy.getRank() == 0 && addrTy.getElementType().isa<MemRefType>())
      base = rewriter.create<memref::LoadOp>(loc, base, ValueRange{});
    auto ctx = rewriter.getContext();
    // 0-d view of the element at the flat offset: address of the element.
    auto resTy = MemRefType::get(
        {}, elemTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {}));
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resTy, base, flat, ValueRange{}, ValueRange{});
    return success();
  }
};

// MARK: loop lowerings

/// Replace the loop-body terminators (end/continue/exit) with cf branches.
/// end/continue -> continueB, exit -> exitB.
static void lowerLoopTerminators(ConversionPatternRewriter &rewriter,
                                 Block *bodyB, Block *continueB, Block *exitB) {
  for (auto &o : llvm::make_early_inc_range(bodyB->getOperations())) {
    if (isa<scl::EndOp, scl::ContinueOp>(&o)) {
      rewriter.setInsertionPointToEnd(bodyB);
      rewriter.create<cf::BranchOp>(o.getLoc(), continueB);
      rewriter.eraseOp(&o);
    } else if (isa<scl::ExitOp>(&o)) {
      rewriter.setInsertionPointToEnd(bodyB);
      rewriter.create<cf::BranchOp>(o.getLoc(), exitB);
      rewriter.eraseOp(&o);
    }
  }
}

// MARK: ForDoOpLowering

/// Lower `scl.for` to a cf loop: init; cond; body; incr; cond...
struct ForDoOpLowering : public OpConversionPattern<scl::ForDoOp> {
  using OpConversionPattern<scl::ForDoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ForDoOp op, scl::ForDoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value var = adaptor.getVar();
    Value init = adaptor.getInit();
    Value to = adaptor.getTo();
    Value step = adaptor.getBy();
    if (!step) {
      auto ty = init.getType();
      auto attr = ty.isa<FloatType>()
                      ? (TypedAttr)rewriter.getFloatAttr(ty, 1.0)
                      : (TypedAttr)rewriter.getIntegerAttr(ty, 1);
      step = rewriter.create<arith::ConstantOp>(loc, ty, attr);
    }

    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();
    Block *condB = rewriter.createBlock(&parent, endB->getIterator());
    Block *bodyB = rewriter.createBlock(&parent, endB->getIterator());
    Block *incrB = rewriter.createBlock(&parent, endB->getIterator());

    // head: store init, jump to condition
    rewriter.setInsertionPointToEnd(head);
    rewriter.create<memref::StoreOp>(loc, init, var, ValueRange{});
    rewriter.create<cf::BranchOp>(loc, condB);

    // move the body
    bodyB->getOperations().splice(bodyB->end(),
                                 op.getDoBody().front().getOperations());
    rewriter.eraseOp(op);
    lowerLoopTerminators(rewriter, bodyB, incrB, endB);
    if (bodyB->empty() || !bodyB->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPointToEnd(bodyB);
      rewriter.create<cf::BranchOp>(loc, incrB);
    }

    // cond: iv = load var; cond = step > 0 ? iv <= to : iv >= to
    rewriter.setInsertionPointToStart(condB);
    Value iv = rewriter.create<memref::LoadOp>(loc, var, ValueRange{});
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, step.getType(), rewriter.getIntegerAttr(step.getType(), 0));
    Value pos = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               step, zero);
    Value le = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                              iv, to);
    Value ge = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              iv, to);
    Value cond = rewriter.create<arith::SelectOp>(loc, pos, le, ge);
    rewriter.create<cf::CondBranchOp>(loc, cond, bodyB, ValueRange{}, endB,
                                  ValueRange{});

    // incr: var += step; jump to condition
    rewriter.setInsertionPointToStart(incrB);
    Value iv2 = rewriter.create<memref::LoadOp>(loc, var, ValueRange{});
    Value next = iv2.getType().isa<FloatType>()
                     ? (Value)rewriter.create<arith::AddFOp>(loc, iv2, step)
                     : (Value)rewriter.create<arith::AddIOp>(loc, iv2, step);
    rewriter.create<memref::StoreOp>(loc, next, var, ValueRange{});
    rewriter.create<cf::BranchOp>(loc, condB);
    return success();
  }
};

// MARK: WhileDoOpLowering

/// Lower `scl.while` to: br cond; cond: <whileBody> cond_br c, body, end; body: <doBody>...
struct WhileDoOpLowering : public OpConversionPattern<scl::WhileDoOp> {
  using OpConversionPattern<scl::WhileDoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::WhileDoOp op, scl::WhileDoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();
    Block *condB = rewriter.createBlock(&parent, endB->getIterator());
    Block *bodyB = rewriter.createBlock(&parent, endB->getIterator());

    auto condOp = dyn_cast<scl::ConditionOp>(
        op.getWhileBody().front().getTerminator());
    if (!condOp)
      return failure();
    // The ConditionOp stays as condB's terminator and is lowered by
    // ConditionOpLowering, which uses the converted (i1) condition value.
    condOp->setSuccessor(bodyB, 0);
    condOp->setSuccessor(endB, 1);

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::BranchOp>(loc, condB);

    // move condition region (incl. the ConditionOp terminator) into condB
    condB->getOperations().splice(condB->end(),
                                  op.getWhileBody().front().getOperations());

    // move body
    bodyB->getOperations().splice(bodyB->end(),
                                  op.getDoBody().front().getOperations());
    rewriter.eraseOp(op);
    lowerLoopTerminators(rewriter, bodyB, condB, endB);
    if (bodyB->empty() || !bodyB->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPointToEnd(bodyB);
      rewriter.create<cf::BranchOp>(loc, condB);
    }
    return success();
  }
};

/// Lower `scl.condition %cond, ^true, ^false` to `cf.cond_br`.
/// Runs after the condition producer has been converted, so the adaptor
/// provides the converted i1 value.
struct ConditionOpLowering : public OpConversionPattern<scl::ConditionOp> {
  using OpConversionPattern<scl::ConditionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ConditionOp op, scl::ConditionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCond(), op.getTrueDest(), ValueRange{}, op.getFalseDest(),
        ValueRange{});
    return success();
  }
};

// MARK: RepeatOpLowering

/// Lower `scl.repeat` to: br body; body: <body> br check; check: until c ? exit : body
struct RepeatOpLowering : public OpConversionPattern<scl::RepeatOp> {
  using OpConversionPattern<scl::RepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::RepeatOp op, scl::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();
    Block *bodyB = rewriter.createBlock(&parent, endB->getIterator());
    Block *checkB = rewriter.createBlock(&parent, endB->getIterator());

    auto &repeatBody = op.getRepeatBody().front();
    auto untilOp = dyn_cast<scl::UntilOp>(repeatBody.getTerminator());
    if (!untilOp)
      return failure();
    // The UntilOp moves to checkB and is lowered by UntilOpLowering, which
    // uses the converted (i1) condition value. until cond -> exit, else repeat.
    untilOp->setSuccessor(endB, 0);
    untilOp->setSuccessor(bodyB, 1);

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::BranchOp>(loc, bodyB);

    // move body (all ops except the UntilOp terminator) into bodyB
    bodyB->getOperations().splice(bodyB->end(), repeatBody.getOperations(),
                                  repeatBody.begin(), untilOp->getIterator());
    rewriter.eraseOp(op);
    lowerLoopTerminators(rewriter, bodyB, checkB, endB);
    if (bodyB->empty() || !bodyB->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPointToEnd(bodyB);
      rewriter.create<cf::BranchOp>(loc, checkB);
    }

    // move the UntilOp terminator into checkB
    checkB->getOperations().splice(checkB->end(), repeatBody.getOperations());
    return success();
  }
};

/// Lower `scl.until %cond, ^exit, ^repeat` to `cf.cond_br`.
struct UntilOpLowering : public OpConversionPattern<scl::UntilOp> {
  using OpConversionPattern<scl::UntilOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::UntilOp op, scl::UntilOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCond(), op.getTrueDest(), ValueRange{}, op.getFalseDest(),
        ValueRange{});
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
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<cf::ControlFlowDialect>();
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
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<cf::ControlFlowDialect>();
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<scl::DebugPrintOp>();

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();

  SclTypeConverter converter{};

  RewritePatternSet patterns(context);
  patterns.add<
      AddOpLowering, AndOpLowering, CallFcOpLowering, ConditionOpLowering,
      ConstantOpLowering, DataBlockOpLowering, DialectCastOpLowering,
      DivOpLowering, EndOpLowering, EqualLowering, ForDoOpLowering,
      FunctionOpLowering, GetGlobalOpLowering, GetIndexOpLowering,
      GreaterEqualLowering, GreaterThanLowering, IfThenElseOpLowering,
      LessEqualLowering, LessThanLowering, LoadOpLowering, ModOpLowering,
      MulOpLowering, NotEqualLowering, OrOpLowering, RepeatOpLowering,
      ReturnOpLowering, ReturnValueOpLowering, StoreOpLowering, SubOpLowering,
      TempVariableOpLowering, UnaryMinusOpLowering, UnaryNotOpLowering,
      UntilOpLowering, WhileDoOpLowering, XOrOpLowering>(converter, context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
