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
#include "mlir/Dialect/Math/IR/Math.h"
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

// MARK: FB/struct layout

/// Memory layout of a function block instance (its variables) or struct,
/// computed from the ops before conversion starts (the ops are erased during
/// conversion, so the layout must be cached up front).
struct FbMemberLayout {
  std::string name;
  int64_t offset;
  Type type;
  /// Cold-start initial value (bit pattern) if the initializer is constant.
  std::optional<llvm::APInt> initBits;
};
struct FbLayout {
  int64_t size = 0;
  std::vector<FbMemberLayout> members;
};
using FbLayoutCache = llvm::StringMap<FbLayout>;

/// Size of a type in bytes.
static int64_t byteSize(Type type, const FbLayoutCache &layouts) {
  if (auto intTy = type.dyn_cast<scl::IntegerType>())
    return (intTy.getWidth() + 7) / 8;
  if (auto logTy = type.dyn_cast<scl::LogicalType>())
    return (logTy.getWidth() + 7) / 8;
  if (type.isa<scl::RealType>())
    return 4;
  if (auto arrTy = type.dyn_cast<scl::ArrayType>()) {
    int64_t count = 1;
    for (auto &dim : arrTy.getDimensions())
      count *= dim.second - dim.first + 1;
    return count * byteSize(arrTy.getElementType(), layouts);
  }
  if (auto structTy = type.dyn_cast<scl::StructType>()) {
    int64_t size = 0;
    for (auto elem : structTy.getElementTypes())
      size += byteSize(elem, layouts);
    return size;
  }
  if (auto idbTy = type.dyn_cast<scl::InstanceDbType>()) {
    auto it = layouts.find(idbTy.getFbSymbol());
    return it != layouts.end() ? it->second.size : 0;
  }
  // date/time types
  if (type.isa<scl::S5TimeType>() || type.isa<scl::DateType>())
    return 2;
  if (type.isa<scl::TimeType>() || type.isa<scl::TimeOfDayType>())
    return 4;
  if (type.isa<scl::DateAndTimeType>())
    return 8;
  return 0;
}

/// Compute the layouts of all function blocks in the module.
static FbLayoutCache computeFbLayouts(ModuleOp module) {
  FbLayoutCache layouts;
  // Two passes so that FBs referencing other FBs work regardless of order.
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (auto fb : module.getOps<scl::FunctionBlockOp>()) {
      FbLayout layout;
      for (auto var : fb.getBody().getOps<scl::VariableOp>()) {
        int64_t size = byteSize(var.getType(), layouts);
        FbMemberLayout member{var.getSymName().str(), layout.size,
                              var.getType(), std::nullopt};
        // Capture a constant initializer as a bit pattern; it is baked into
        // the instance DB global's initializer (cold start), NOT emitted in
        // the FB body (which would re-initialize on every call).
        if (Value init = var.getInit()) {
          if (auto c = init.getDefiningOp<scl::ConstantOp>()) {
            Attribute attr = c.getValue();
            if (auto intAttr = attr.dyn_cast<IntegerAttr>())
              member.initBits = intAttr.getValue();
            else if (auto floatAttr = attr.dyn_cast<FloatAttr>())
              member.initBits = floatAttr.getValue().bitcastToAPInt();
          }
        }
        layout.members.push_back(std::move(member));
        layout.size += size;
      }
      layouts[fb.getName()] = layout;
    }
  }
  return layouts;
}

// MARK: SclTypeConverter

class SclTypeConverter : public TypeConverter {
public:
  FbLayoutCache fbLayouts;

  SclTypeConverter() : SclTypeConverter(FbLayoutCache{}) {}
  explicit SclTypeConverter(FbLayoutCache layouts) : fbLayouts(std::move(layouts)) {
    addConversion([&](scl::AddressType type) {
      auto elem = type.getElementType();
      // An instance DB address is the instance buffer itself.
      if (auto idbTy = elem.dyn_cast<scl::InstanceDbType>()) {
        auto it = fbLayouts.find(idbTy.getFbSymbol());
        if (it == fbLayouts.end())
          return Type();
        return (Type)MemRefType::get({it->second.size},
                                     IntegerType::get(type.getContext(), 8));
      }
      auto celem = convertType(elem);
      if (!celem)
        return Type();
      // An address of a composite (array/struct/IDB) IS the buffer memref.
      if (celem.isa<MemRefType>())
        return celem;
      // An i8 address is a rank-1 byte view into a byte buffer (FB/struct
      // members). It must not be chained through another reinterpret_cast:
      // ReinterpretCastOp::fold strips the inner offset when folding
      // reinterpret_cast(reinterpret_cast(x)), so byte accesses index the
      // view directly. Other addresses are 0-d memrefs with dynamic offset,
      // so that array element addresses produced by memref.reinterpret_cast
      // have the same type.
      if (celem.isInteger(8))
        return (Type)mlir::MemRefType::get(
            {ShapedType::kDynamic}, celem,
            StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
                                   {1}));
      return (Type)mlir::MemRefType::get(
          {}, celem,
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
    // A struct or instance DB is a flat byte buffer; members are addressed by
    // byte offset via memref.reinterpret_cast.
    addConversion([&](scl::StructType type) {
      return (Type)MemRefType::get(
          {byteSize(type, fbLayouts)}, IntegerType::get(type.getContext(), 8));
    });
    addConversion([&](scl::InstanceDbType type) {
      auto it = fbLayouts.find(type.getFbSymbol());
      if (it == fbLayouts.end())
        return Type();
      return (Type)MemRefType::get({it->second.size},
                                   IntegerType::get(type.getContext(), 8));
    });
    // date/time types lower to plain integers of their storage width
    addConversion([&](scl::S5TimeType type) {
      return IntegerType::get(type.getContext(), 16);
    });
    addConversion([&](scl::TimeType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([&](scl::DateType type) {
      return IntegerType::get(type.getContext(), 16);
    });
    addConversion([&](scl::TimeOfDayType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([&](scl::DateAndTimeType type) {
      return IntegerType::get(type.getContext(), 64);
    });

    // keep Std types
    addConversion([&](FloatType type) { return type; });
    addConversion([&](IntegerType type) { return type; });
    addConversion([&](IndexType type) { return type; });
    addConversion([&](MemRefType type) { return type; });

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

// MARK: LoadOpLowering

/// Load a value of `resTy` from `addr`; byte-assemble if the address is an
/// i8 view (FB/struct members in a byte buffer).
static Value loadFromByteBuffer(OpBuilder &b, Location loc, Value addr,
                                Type resTy) {
  auto memTy = addr.getType().cast<MemRefType>();
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  if (memTy.getElementType() == resTy) {
    SmallVector<Value, 1> indices(memTy.getRank(), zero);
    return b.create<memref::LoadOp>(loc, addr, indices);
  }
  if (!memTy.getElementType().isInteger(8) || memTy.getRank() != 1)
    return nullptr;
  if (!resTy.isIntOrFloat())
    return nullptr;
  int64_t nbytes = (resTy.getIntOrFloatBitWidth() + 7) / 8;
  if (nbytes == 0 || nbytes > 8)
    return nullptr;
  auto i8 = IntegerType::get(b.getContext(), 8);
  if (resTy.isInteger(1)) {
    // bool: a single byte, true iff non-zero (extui/trunc i8<->i1 are
    // invalid casts).
    auto idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value byte = b.create<memref::LoadOp>(loc, addr, ValueRange{idx0});
    auto zero = b.create<arith::ConstantOp>(loc, i8, b.getIntegerAttr(i8, 0));
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, byte, zero);
  }
  auto accTy = resTy.isa<FloatType>()
                   ? IntegerType::get(b.getContext(), nbytes <= 4 ? 32 : 64)
                   : resTy;
  // addr is a rank-1 i8 view starting at the value (FB/struct member);
  // assemble the value byte by byte. Do not chain another reinterpret_cast:
  // its fold would drop the member offset.
  Value acc = b.create<arith::ConstantOp>(loc, accTy,
                                          b.getIntegerAttr(accTy, 0));
  for (int64_t i = 0; i < nbytes; ++i) {
    auto idx = b.create<arith::ConstantIndexOp>(loc, i);
    Value byte = b.create<memref::LoadOp>(loc, addr, ValueRange{idx});
    Value ext = b.create<arith::ExtUIOp>(loc, accTy, byte);
    if (i > 0) {
      auto sh = b.create<arith::ConstantOp>(loc, accTy,
                                            b.getIntegerAttr(accTy, 8 * i));
      ext = b.create<arith::ShLIOp>(loc, ext, sh);
    }
    acc = b.create<arith::OrIOp>(loc, acc, ext);
  }
  if (resTy.isa<FloatType>())
    return b.create<arith::BitcastOp>(loc, resTy, acc);
  if (accTy != resTy)
    return b.create<arith::TruncIOp>(loc, resTy, acc);
  return acc;
}

struct LoadOpLowering : public OpConversionPattern<scl::LoadOp> {
  using OpConversionPattern<scl::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::LoadOp op, scl::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    Value loaded = loadFromByteBuffer(rewriter, op.getLoc(),
                                       adaptor.getAddress(), resTy);
    if (!loaded)
      return failure();
    rewriter.replaceOp(op, loaded);
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
    auto resTy = getTypeConverter()->convertType(op.getReturnType());
    if (!resTy)
      return failure();
    Value retval = loadFromByteBuffer(rewriter, op.getLoc(), adaptor.getValue(),
                                      resTy);
    if (!retval)
      return failure();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, retval);
    return success();
  }
};

// MARK: StoreOpLowering

/// Store `value` at `addr`, splitting into bytes if the address is an i8 view.
static LogicalResult storeToByteBuffer(OpBuilder &b, Location loc, Value value,
                                       Value addr) {
  auto memTy = addr.getType().cast<MemRefType>();
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  // Whole-buffer assignment (array/struct value stored into its buffer).
  if (value.getType().isa<MemRefType>() && addr.getType() == value.getType()) {
    b.create<memref::CopyOp>(loc, value, addr);
    return success();
  }
  if (memTy.getElementType() == value.getType()) {
    SmallVector<Value, 1> indices(memTy.getRank(), zero);
    b.create<memref::StoreOp>(loc, value, addr, indices);
    return success();
  }
  if (!memTy.getElementType().isInteger(8) || memTy.getRank() != 1)
    return failure();
  Type valTy = value.getType();
  if (!valTy.isIntOrFloat())
    return failure();
  int64_t nbytes = (valTy.getIntOrFloatBitWidth() + 7) / 8;
  if (nbytes == 0 || nbytes > 8)
    return failure();
  auto i8 = IntegerType::get(b.getContext(), 8);
  if (valTy.isInteger(1)) {
    // bool: a single byte, zero-extended from the i1 value.
    Value byte = b.create<arith::ExtUIOp>(loc, i8, value);
    auto idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
    b.create<memref::StoreOp>(loc, byte, addr, ValueRange{idx0});
    return success();
  }
  auto accTy = valTy.isa<FloatType>()
                   ? IntegerType::get(b.getContext(), nbytes <= 4 ? 32 : 64)
                   : valTy;
  Value acc = value;
  if (valTy.isa<FloatType>())
    acc = b.create<arith::BitcastOp>(loc, accTy, value);
  else if (accTy != valTy)
    acc = b.create<arith::ExtSIOp>(loc, accTy, value);
  // addr is a rank-1 i8 view starting at the value (FB/struct member);
  // split the value into bytes and store them directly.
  for (int64_t i = 0; i < nbytes; ++i) {
    Value byte = acc;
    if (i > 0) {
      auto sh = b.create<arith::ConstantOp>(loc, accTy,
                                            b.getIntegerAttr(accTy, 8 * i));
      byte = b.create<arith::ShRUIOp>(loc, acc, sh);
    }
    byte = b.create<arith::TruncIOp>(loc, i8, byte);
    auto idx = b.create<arith::ConstantIndexOp>(loc, i);
    b.create<memref::StoreOp>(loc, byte, addr, ValueRange{idx});
  }
  return success();
}

struct StoreOpLowering : public OpConversionPattern<scl::StoreOp> {
  using OpConversionPattern<scl::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::StoreOp op, scl::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(storeToByteBuffer(rewriter, op.getLoc(), adaptor.getRhs(),
                                 adaptor.getLhs())))
      return failure();
    rewriter.eraseOp(op);
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
    auto loc = op.getLoc();
    // A composite variable (array/struct/IDB) allocates the buffer itself;
    // the address is the buffer memref with identity layout.
    if (memref.getLayout().isIdentity()) {
      auto alloca = rewriter.create<memref::AllocaOp>(loc, memref);
      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }
    // alloca requires an identity layout; reinterpret to the address type.
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    if (memref.getRank() == 1) {
      // i8 address: rank-1 byte view holding a single byte.
      auto plain = MemRefType::get({1}, memref.getElementType());
      auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
          op, memref, alloca, zero, ValueRange{one}, ValueRange{one});
      return success();
    }
    auto plain = MemRefType::get({}, memref.getElementType());
    auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
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

struct DataBlockOpLowering : public OpConversionPattern<scl::DataBlockOp> {
  using OpConversionPattern<scl::DataBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DataBlockOp op, scl::DataBlockOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    mlir::Type memrefTy;
    scl::InstanceDbType idbTyVal;
    if (auto idbTy = op.getType().dyn_cast<scl::InstanceDbType>()) {
      // An instance DB of a FB is a flat byte buffer sized by the FB layout.
      auto it = layouts.find(idbTy.getFbSymbol());
      if (it == layouts.end())
        return failure();
      idbTyVal = idbTy;
      memrefTy = MemRefType::get({it->second.size},
                                 IntegerType::get(op.getContext(), 8));
    } else {
      auto elemTy = getTypeConverter()->convertType(op.getType());
      if (!elemTy)
        return failure();
      // A DB of composite type is the buffer memref itself; a scalar DB is a
      // 0-d strided memref (the address type).
      if (elemTy.isa<MemRefType>())
        memrefTy = elemTy;
      else
        memrefTy = MemRefType::get(
            {}, elemTy,
            StridedLayoutAttr::get(op.getContext(), ShapedType::kDynamic, {}));
    }
    // An initializer makes the LLVM global defined (not external), so the
    // JIT can resolve its symbol. Initial values are stored as a
    // DenseElementsAttr over the equivalent ranked tensor type. For an
    // instance DB, bake the FB members' cold-start initial values into the
    // byte buffer; everything else is zero-filled.
    auto memref = memrefTy.cast<MemRefType>();
    Attribute initValue = nullptr;
    if (memref.getElementType().isIntOrFloat() ||
        memref.getElementType().isa<IndexType>()) {
      auto tensorTy = RankedTensorType::get(memref.getShape(),
                                            memref.getElementType());
      if (memref.getElementType().isInteger(8) && idbTyVal) {
        auto &layout = layouts.find(idbTyVal.getFbSymbol())->second;
        SmallVector<Attribute, 8> bytes;
        bytes.reserve(layout.size);
        std::vector<uint8_t> buf(layout.size, 0);
        for (auto &m : layout.members) {
          if (!m.initBits)
            continue;
          int64_t msize = byteSize(m.type, layouts);
          for (int64_t i = 0; i < msize; ++i)
            buf[m.offset + i] =
                (uint8_t)(m.initBits->getZExtValue() >> (8 * i));
        }
        auto i8 = memref.getElementType();
        for (uint8_t byte : buf)
          bytes.push_back(rewriter.getIntegerAttr(i8, byte));
        initValue = DenseElementsAttr::get(tensorTy, bytes);
      } else {
        initValue = DenseElementsAttr::get(
            tensorTy, rewriter.getZeroAttr(memref.getElementType()));
      }
    }
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymNameAttr(), /*sym_visibility=*/nullptr,
        TypeAttr::get(memrefTy), /*initial_value=*/initValue,
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

/// Walk a chain of memref.reinterpret_cast byte views down to the root
/// buffer, accumulating the offset (constant part into `constOff`, dynamic
/// part into `dynOff`). Returns the root value, or null if the chain bottoms
/// out at something that is not a memref. Used to create a *single* cast from
/// the root: chained reinterpret_casts would lose the inner offset when
/// ReinterpretCastOp::fold fires.
static Value resolveByteViewRoot(Value v, Value &dynOff, int64_t &constOff,
                                 OpBuilder &b, Location loc) {
  constOff = 0;
  dynOff = nullptr;
  auto indexTy = b.getIndexType();
  while (auto rc = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    OpFoldResult ofr = rc.getMixedOffsets()[0];
    if (auto c = mlir::getConstantIntValue(ofr)) {
      constOff += *c;
    } else if (Value o = ofr.dyn_cast<Value>()) {
      if (o.getType() != indexTy)
        o = b.create<arith::IndexCastOp>(loc, indexTy, o);
      dynOff = dynOff ? (Value)b.create<arith::AddIOp>(loc, dynOff, o) : o;
    } else {
      return nullptr;
    }
    v = rc.getSource();
  }
  if (!v.getType().isa<MemRefType>())
    return nullptr;
  return v;
}

/// Combine a constant offset, a dynamic offset value and an extra dynamic
/// offset into a single index value.
static Value combineOffsets(OpBuilder &b, Location loc, int64_t constOff,
                            Value dynOff, Value extra) {
  Value res = extra;
  if (dynOff)
    res = res ? (Value)b.create<arith::AddIOp>(loc, res, dynOff) : dynOff;
  if (constOff != 0) {
    auto c = b.create<arith::ConstantIndexOp>(loc, constOff);
    res = res ? (Value)b.create<arith::AddIOp>(loc, res, c) : c;
  }
  if (!res)
    res = b.create<arith::ConstantIndexOp>(loc, 0);
  return res;
}

/// Create a rank-1 i8 byte view of `size` bytes at `off` from a single cast
/// of the root buffer.
static Value makeByteView(OpBuilder &b, Location loc, Value root, Value off,
                          int64_t size, MLIRContext *ctx) {
  auto i8 = IntegerType::get(ctx, 8);
  auto resTy = MemRefType::get(
      {ShapedType::kDynamic}, i8,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}));
  auto sizeV = b.create<arith::ConstantIndexOp>(loc, size);
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  return b.create<memref::ReinterpretCastOp>(loc, resTy, root, off,
                                             ValueRange{sizeV},
                                             ValueRange{one});
}

// MARK: GetIndexOpLowering

/// Lower array element addressing to a flat offset into the memref.
struct GetIndexOpLowering : public OpConversionPattern<scl::GetIndexOp> {
  using OpConversionPattern<scl::GetIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetIndexOp op, scl::GetIndexOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    // The declared array type: from the attr when the base is a byte view
    // (array member of a FB/struct), else from the address element type.
    scl::ArrayType arrayTy;
    if (op.getArrayType().has_value())
      arrayTy = op.getArrayType()->cast<scl::ArrayType>();
    else
      arrayTy = op.getAddr().getType()
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
    // The address of an array is the array buffer memref itself (or a byte
    // view into a composite buffer).
    Value base = adaptor.getAddr();
    auto ctx = rewriter.getContext();
    auto baseMemTy = base.getType().cast<MemRefType>();
    // Byte path: the base is a rank-1 i8 view (array member of a FB/struct)
    // or a plain char array buffer. Compute the byte offset and create a
    // single cast from the root buffer.
    if (baseMemTy.getRank() == 1 && baseMemTy.getElementType().isInteger(8)) {
      int64_t elemBytes = byteSize(arrayTy.getElementType(), layouts);
      if (elemBytes <= 0)
        return failure();
      Value dynOff;
      int64_t constOff;
      Value root = resolveByteViewRoot(base, dynOff, constOff, rewriter, loc);
      if (!root)
        return failure();
      Value byteFlat = flat;
      if (elemBytes != 1) {
        auto scale = rewriter.create<arith::ConstantIndexOp>(loc, elemBytes);
        byteFlat = rewriter.create<arith::MulIOp>(loc, flat, scale);
      }
      Value off = combineOffsets(rewriter, loc, constOff, dynOff, byteFlat);
      Value view = makeByteView(rewriter, loc, root, off, elemBytes, ctx);
      rewriter.replaceOp(op, view);
      return success();
    }
    // Address of the element at the flat offset as a 0-d view.
    auto resTy = MemRefType::get(
        {}, elemTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {}));
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resTy, base, flat, ValueRange{}, ValueRange{});
    return success();
  }
};

// MARK: region moving helpers

/// Move all blocks of `src` into `dst` before `before`; returns the moved
/// entry block. After the move, `src` is empty.
static Block *moveRegion(ConversionPatternRewriter &rewriter, Region &src,
                         Region &dst, Block *before) {
  if (src.empty())
    return nullptr;
  Block *entry = &src.front();
  rewriter.inlineRegionBefore(src, dst, before->getIterator());
  return entry;
}

/// Replace `scl.end` terminators in `blocks` with a branch to contB. Blocks
/// left without a terminator get a branch to contB. continue/exit are NOT
/// handled here: they are lowered by their own patterns to the innermost
/// enclosing loop's targets (recorded by the loop lowerings).
static void lowerTerminatorsInBlocks(ConversionPatternRewriter &rewriter,
                                     Location loc, ArrayRef<Block *> blocks,
                                     Block *contB, Block *exitB) {
  for (Block *b : blocks) {
    // Collect first: erasing while iterating and re-creating terminators in
    // the same pass can leave duplicate terminators behind.
    SmallVector<Operation *> toRewrite;
    for (auto &o : b->getOperations())
      if (isa<scl::EndOp>(&o))
        toRewrite.push_back(&o);
    for (Operation *o : toRewrite) {
      Location l = o->getLoc();
      rewriter.eraseOp(o);
      rewriter.setInsertionPointToEnd(b);
      rewriter.create<cf::BranchOp>(l, contB);
    }
    if (b->empty() || !b->back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPointToEnd(b);
      rewriter.create<cf::BranchOp>(loc, contB);
    }
  }
}

/// Maps each continue/exit op to the (continue, exit) blocks of the
/// innermost enclosing loop. Filled by the loop lowerings, which are applied
/// before the nested ops (dialect conversion is pre-order), and consumed by
/// ContinueOpLowering/ExitOpLowering when the nested op is converted.
struct LoopTerminatorTargets {
  DenseMap<Operation *, std::pair<Block *, Block *>> map;
};
using LoopTargetsPtr = std::shared_ptr<LoopTerminatorTargets>;

/// Record continue/exit ops in `blocks` and in regions nested inside non-loop
/// ops (if/case bodies), so their lowerings can branch to the loop targets.
/// Nested loop ops are skipped: they record their own bodies.
static void recordLoopTerminators(Operation *op, Block *contB, Block *exitB,
                                  LoopTerminatorTargets &targets) {
  if (isa<scl::ForDoOp, scl::WhileDoOp, scl::RepeatOp>(op))
    return;
  for (Region &r : op->getRegions())
    for (Block &b : r)
      for (Operation &o : b) {
        if (isa<scl::ContinueOp, scl::ExitOp>(&o))
          targets.map[&o] = {contB, exitB};
        else
          recordLoopTerminators(&o, contB, exitB, targets);
      }
}

static void recordLoopTerminators(ArrayRef<Block *> blocks, Block *contB,
                                  Block *exitB, LoopTerminatorTargets &targets) {
  for (Block *b : blocks)
    for (Operation &o : *b) {
      if (isa<scl::ContinueOp, scl::ExitOp>(&o))
        targets.map[&o] = {contB, exitB};
      else
        recordLoopTerminators(&o, contB, exitB, targets);
    }
}

/// Lower `scl.continue` to a branch to the continue block of the innermost
/// enclosing loop (recorded by the loop lowering).
struct ContinueOpLowering : public OpConversionPattern<scl::ContinueOp> {
  LoopTargetsPtr targets;
  ContinueOpLowering(TypeConverter &tc, MLIRContext *ctx,
                     LoopTargetsPtr t)
      : OpConversionPattern(tc, ctx), targets(std::move(t)) {}
  LogicalResult
  matchAndRewrite(scl::ContinueOp op, scl::ContinueOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto it = targets->map.find(op.getOperation());
    if (it == targets->map.end()) {
      emitError(op.getLoc()) << "continue outside of a loop";
      return failure();
    }
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, it->second.first);
    return success();
  }
};

/// Lower `scl.exit` to a branch to the exit block of the innermost enclosing
/// loop.
struct ExitOpLowering : public OpConversionPattern<scl::ExitOp> {
  LoopTargetsPtr targets;
  ExitOpLowering(TypeConverter &tc, MLIRContext *ctx, LoopTargetsPtr t)
      : OpConversionPattern(tc, ctx), targets(std::move(t)) {}
  LogicalResult
  matchAndRewrite(scl::ExitOp op, scl::ExitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto it = targets->map.find(op.getOperation());
    if (it == targets->map.end()) {
      emitError(op.getLoc()) << "exit outside of a loop";
      return failure();
    }
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, it->second.second);
    return success();
  }
};

/// Collect the blocks strictly between `from` (exclusive) and `to`
/// (exclusive).
static SmallVector<Block *> blocksBetween(Block *from, Block *to) {
  SmallVector<Block *> result;
  for (Block *b = from->getNextNode(); b && b != to; b = b->getNextNode())
    result.push_back(b);
  return result;
}

/// Collect the blocks from `from` (inclusive) up to `to` (exclusive).
static SmallVector<Block *> blocksFrom(Block *from, Block *to) {
  SmallVector<Block *> result;
  for (Block *b = from; b && b != to; b = b->getNextNode())
    result.push_back(b);
  return result;
}

// MARK: IfThenElseOpLowering

/// Lower `scl.if` to cf blocks: cond_br cond, thenEntry, elseEntry.
struct IfThenElseOpLowering : public OpConversionPattern<scl::IfThenElseOp> {
  using OpConversionPattern<scl::IfThenElseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::IfThenElseOp op, scl::IfThenElseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();

    Block *thenEntry = moveRegion(rewriter, op.getThenBody(), parent, endB);
    Block *elseEntry = endB;
    if (!op.getElseBody().empty())
      elseEntry = moveRegion(rewriter, op.getElseBody(), parent, endB);

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::CondBranchOp>(loc, adaptor.getCond(), thenEntry,
                                      ValueRange{}, elseEntry, ValueRange{});
    rewriter.eraseOp(op);
    lowerTerminatorsInBlocks(rewriter, loc, blocksBetween(head, endB), endB,
                             endB);
    return success();
  }
};

// MARK: ForDoOpLowering

/// Lower `scl.for` to a cf loop: init; cond; body; incr; cond...
struct ForDoOpLowering : public OpConversionPattern<scl::ForDoOp> {
  LoopTargetsPtr loopTargets;
  ForDoOpLowering(TypeConverter &tc, MLIRContext *ctx,
                  LoopTargetsPtr t)
      : OpConversionPattern(tc, ctx), loopTargets(std::move(t)) {}

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
    Block *incrB = rewriter.createBlock(&parent, endB->getIterator());
    Block *bodyEntry = moveRegion(rewriter, op.getDoBody(), parent, endB);
    auto bodyBlocks = blocksBetween(incrB, endB);

    // head: store init, jump to condition
    rewriter.setInsertionPointToEnd(head);
    if (failed(storeToByteBuffer(rewriter, loc, init, var)))
      return failure();
    rewriter.create<cf::BranchOp>(loc, condB);

    rewriter.eraseOp(op);
    recordLoopTerminators(bodyBlocks, incrB, endB, *loopTargets);
    lowerTerminatorsInBlocks(rewriter, loc, bodyBlocks, incrB, endB);

    // cond: iv = load var; cond = step > 0 ? iv <= to : iv >= to
    rewriter.setInsertionPointToStart(condB);
    Value iv = loadFromByteBuffer(rewriter, loc, var, init.getType());
    if (!iv)
      return failure();
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, step.getType(), rewriter.getIntegerAttr(step.getType(), 0));
    Value pos = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               step, zero);
    Value le = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                              iv, to);
    Value ge = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              iv, to);
    Value cond = rewriter.create<arith::SelectOp>(loc, pos, le, ge);
    rewriter.create<cf::CondBranchOp>(loc, cond, bodyEntry, ValueRange{}, endB,
                                      ValueRange{});

    // incr: var += step; jump to condition
    rewriter.setInsertionPointToStart(incrB);
    Value iv2 = loadFromByteBuffer(rewriter, loc, var, init.getType());
    if (!iv2)
      return failure();
    Value next = iv2.getType().isa<FloatType>()
                     ? (Value)rewriter.create<arith::AddFOp>(loc, iv2, step)
                     : (Value)rewriter.create<arith::AddIOp>(loc, iv2, step);
    if (failed(storeToByteBuffer(rewriter, loc, next, var)))
      return failure();
    rewriter.create<cf::BranchOp>(loc, condB);
    return success();
  }
};

// MARK: WhileDoOpLowering

struct WhileDoOpLowering : public OpConversionPattern<scl::WhileDoOp> {
  LoopTargetsPtr loopTargets;
  WhileDoOpLowering(TypeConverter &tc, MLIRContext *ctx,
                    LoopTargetsPtr t)
      : OpConversionPattern(tc, ctx), loopTargets(std::move(t)) {}

  LogicalResult
  matchAndRewrite(scl::WhileDoOp op, scl::WhileDoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();

    Block *condEntry = moveRegion(rewriter, op.getWhileBody(), parent, endB);
    Block *bodyEntry = moveRegion(rewriter, op.getDoBody(), parent, endB);
    if (!condEntry || !bodyEntry)
      return failure();

    // find the ConditionOp in the moved condition blocks
    scl::ConditionOp condition = nullptr;
    for (Block *b : blocksBetween(head, bodyEntry))
      if (auto *t = b->getTerminator(); isa<scl::ConditionOp>(t))
        condition = cast<scl::ConditionOp>(t);
    if (!condition)
      return failure();
    condition->setSuccessor(bodyEntry, 0);
    condition->setSuccessor(endB, 1);

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::BranchOp>(loc, condEntry);
    rewriter.eraseOp(op);
    auto whileBodyBlocks = blocksFrom(bodyEntry, endB);
    recordLoopTerminators(whileBodyBlocks, condEntry, endB, *loopTargets);
    lowerTerminatorsInBlocks(rewriter, loc, blocksBetween(head, bodyEntry),
                             condEntry, endB);
    lowerTerminatorsInBlocks(rewriter, loc, whileBodyBlocks,
                             condEntry, endB);
    return success();
  }
};

// MARK: ConditionOpLowering

/// Lower `scl.condition %cond, ^true, ^false` to `cf.cond_br`.
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

struct RepeatOpLowering : public OpConversionPattern<scl::RepeatOp> {
  LoopTargetsPtr loopTargets;
  RepeatOpLowering(TypeConverter &tc, MLIRContext *ctx,
                   LoopTargetsPtr t)
      : OpConversionPattern(tc, ctx), loopTargets(std::move(t)) {}

  LogicalResult
  matchAndRewrite(scl::RepeatOp op, scl::RepeatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();

    Block *bodyEntry = moveRegion(rewriter, op.getRepeatBody(), parent, endB);
    if (!bodyEntry)
      return failure();

    // find the UntilOp and its block (the check block)
    scl::UntilOp untilOp = nullptr;
    Block *checkB = nullptr;
    auto bodyBlocks = blocksFrom(bodyEntry, endB);
    for (Block *b : bodyBlocks)
      if (auto *t = b->getTerminator(); isa<scl::UntilOp>(t)) {
        untilOp = cast<scl::UntilOp>(t);
        checkB = b;
      }
    if (!untilOp)
      return failure();
    untilOp->setSuccessor(endB, 0);      // until cond -> exit
    untilOp->setSuccessor(bodyEntry, 1); // else repeat

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::BranchOp>(loc, bodyEntry);
    rewriter.eraseOp(op);
    SmallVector<Block *> rest;
    for (Block *b : bodyBlocks)
      if (b != checkB)
        rest.push_back(b);
    recordLoopTerminators(rest, checkB, endB, *loopTargets);
    lowerTerminatorsInBlocks(rewriter, loc, rest, checkB, endB);
    return success();
  }
};

// MARK: UntilOpLowering

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

// MARK: CaseOpLowering

/// Lower `scl.case` to `cf.switch`. The last body region is the else
/// (default) block.
struct CaseOpLowering : public OpConversionPattern<scl::CaseOp> {
  using OpConversionPattern<scl::CaseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::CaseOp op, scl::CaseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value sel = adaptor.getSelector();
    if (!sel.getType().isa<IntegerType>())
      return failure();
    auto i32 = rewriter.getI32Type();
    if (sel.getType() != i32) {
      auto width = sel.getType().getIntOrFloatBitWidth();
      if (width < 32)
        sel = rewriter.create<arith::ExtSIOp>(loc, i32, sel);
      else if (width > 32)
        sel = rewriter.create<arith::TruncIOp>(loc, i32, sel);
    }

    Block *head = op->getBlock();
    Block *endB = rewriter.splitBlock(head, op->getIterator());
    Region &parent = *head->getParent();

    unsigned numRegions = op.getBody().size();
    if (numRegions == 0)
      return failure();
    SmallVector<Block *> entries;
    for (unsigned i = 0; i < numRegions; ++i)
      entries.push_back(moveRegion(rewriter, op.getBody()[i], parent, endB));
    Block *defaultEntry = entries.back();

    SmallVector<int32_t> caseValues;
    SmallVector<Block *> caseDests;
    SmallVector<ValueRange> caseOperands;
    auto caseValuesAttr = op.getCaseValues();
    for (unsigned i = 0; i + 1 < numRegions; ++i) {
      auto values = caseValuesAttr[i].cast<DenseI32ArrayAttr>();
      for (int32_t v : values.asArrayRef()) {
        caseValues.push_back(v);
        caseDests.push_back(entries[i]);
        caseOperands.push_back(ValueRange{});
      }
    }

    rewriter.setInsertionPointToEnd(head);
    rewriter.create<cf::SwitchOp>(loc, sel, defaultEntry, ValueRange{},
                                  caseValues, caseDests, caseOperands);
    rewriter.eraseOp(op);
    lowerTerminatorsInBlocks(rewriter, loc, blocksBetween(head, endB), endB,
                             endB);
    return success();
  }
};

// MARK: FunctionBlockOpLowering

/// Lower `scl.function_block` to `func.func`; its single argument is the
/// instance DB buffer (a memref of i8).
struct FunctionBlockOpLowering
    : public OpConversionPattern<scl::FunctionBlockOp> {
  using OpConversionPattern<scl::FunctionBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::FunctionBlockOp op, scl::FunctionBlockOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sclType = op.getFunctionType();

    TypeConverter::SignatureConversion signatureConverter(
        sclType.getNumInputs());
    for (auto argType : llvm::enumerate(sclType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }
    auto inputs = signatureConverter.getConvertedTypes();

    SmallVector<Type, 4> results;
    if (failed(typeConverter->convertTypes(sclType.getResults(), results)))
      return failure();

    auto newFuncType = rewriter.getFunctionType(inputs, results);
    auto newFuncOp = rewriter.create<FuncOp>(loc, op.getName(), newFuncType);
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// MARK: VariableOpLowering

/// FB member declarations carry no code; the layout and cold-start
/// initializers live in the IDB global (see computeFbLayouts and
/// DataBlockOpLowering).
struct VariableOpLowering : public OpConversionPattern<scl::VariableOp> {
  using OpConversionPattern<scl::VariableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::VariableOp op, scl::VariableOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// MARK: GetVariableOpLowering

/// Lower `scl.getvariable @name from %idb` to a 0-d i8 view of the instance
/// buffer at the member's byte offset. The value is byte-assembled by
/// Load/Store lowerings.
struct GetVariableOpLowering : public OpConversionPattern<scl::GetVariableOp> {
  using OpConversionPattern<scl::GetVariableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetVariableOp op, scl::GetVariableOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addrTy = op.getAddr().getType().cast<scl::AddressType>();
    auto idbTy = addrTy.getElementType().dyn_cast<scl::InstanceDbType>();
    if (!idbTy)
      return failure();
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    auto it = layouts.find(idbTy.getFbSymbol());
    if (it == layouts.end())
      return failure();
    const FbMemberLayout *member = nullptr;
    for (auto &m : it->second.members)
      if (m.name == op.getSymName())
        member = &m;
    if (!member)
      return failure();

    auto ctx = op.getContext();
    auto i8 = IntegerType::get(ctx, 8);
    // Rank-1 byte view of the member window, created in a single cast from
    // the instance buffer (chaining casts would lose the offset when
    // ReinterpretCastOp::fold fires).
    auto resTy = MemRefType::get(
        {ShapedType::kDynamic}, i8,
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}));
    auto off = rewriter.create<arith::ConstantIndexOp>(loc, member->offset);
    auto size = rewriter.create<arith::ConstantIndexOp>(
        loc, byteSize(member->type, layouts));
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resTy, adaptor.getAddr(), off, ValueRange{size}, ValueRange{one});
    return success();
  }
};

// MARK: GetElementOpLowering

/// Lower `scl.getelement @name from %structAddr` to a 0-d i8 view at the
/// member's byte offset within the struct byte buffer.
struct GetElementOpLowering : public OpConversionPattern<scl::GetElementOp> {
  using OpConversionPattern<scl::GetElementOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetElementOp op, scl::GetElementOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addrTy = op.getAddr().getType().cast<scl::AddressType>();
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    // The declared struct type of the base: from the address element type
    // for plain struct addresses, from the baseType attr when the base is
    // itself a byte view (nested struct member).
    scl::StructType structTy;
    if (auto st = addrTy.getElementType().dyn_cast<scl::StructType>())
      structTy = st;
    else if (op.getBaseType().has_value())
      structTy = op.getBaseType()->cast<scl::StructType>();
    if (!structTy)
      return failure();

    // byte offset of the member within the struct
    int64_t offset = 0;
    Type memberTy;
    for (auto [name, elem] :
         llvm::zip(structTy.getElementNames(), structTy.getElementTypes())) {
      if (name.getValue() == op.getName())
        memberTy = elem;
      if (memberTy)
        break;
      offset += byteSize(elem, layouts);
    }
    if (!memberTy)
      return failure();
    auto ctx = op.getContext();
    // The address of a struct is the struct buffer memref itself (or a byte
    // view into a composite buffer for nested members).
    Value base = adaptor.getAddr();
    // Rank-1 byte view of the member window in a single cast from the root
    // buffer (chained casts lose the offset in ReinterpretCastOp::fold).
    Value dynOff;
    int64_t constOff;
    Value root = resolveByteViewRoot(base, dynOff, constOff, rewriter, loc);
    if (!root)
      return failure();
    Value off = combineOffsets(rewriter, loc, constOff + offset, dynOff,
                               nullptr);
    int64_t size = byteSize(memberTy, layouts);
    if (size <= 0)
      return failure();
    Value view = makeByteView(rewriter, loc, root, off, size, ctx);
    rewriter.replaceOp(op, view);
    return success();
  }
};

// MARK: CallFbOpLowering

struct CallFbOpLowering : public OpConversionPattern<scl::CallFbOp> {
  using OpConversionPattern<scl::CallFbOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::CallFbOp op, scl::CallFbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getFb(),
                                              TypeRange{},
                                              ValueRange{adaptor.getIdb()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// MARK: ExpOpLowering

struct ExpOpLowering : public OpConversionPattern<scl::ExpOp> {
  using OpConversionPattern<scl::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::ExpOp op, scl::ExpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (all_of_type<FloatType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<math::PowFOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
      return success();
    }
    if (all_of_type<IntegerType>(adaptor.getOperands())) {
      rewriter.replaceOpWithNewOp<math::IPowIOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
      return success();
    }
    return failure();
  }
};

// MARK: BitCastOpLowering / IntegerCastOpLowering

struct BitCastOpLowering : public OpConversionPattern<scl::BitCastOp> {
  using OpConversionPattern<scl::BitCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::BitCastOp op, scl::BitCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, resTy,
                                                  adaptor.getValue());
    return success();
  }
};

struct IntegerCastOpLowering : public OpConversionPattern<scl::IntegerCastOp> {
  using OpConversionPattern<scl::IntegerCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::IntegerCastOp op, scl::IntegerCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType()).cast<IntegerType>();
    auto srcTy = adaptor.getValue().getType().cast<IntegerType>();
    if (resTy.getWidth() > srcTy.getWidth())
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resTy, adaptor.getValue());
    else if (resTy.getWidth() < srcTy.getWidth())
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, resTy,
                                                   adaptor.getValue());
    else
      rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

// MARK: date/time constant lowerings

/// Lower a date/time constant op to an integer constant of the converted
/// type, using its value attribute.
template <typename SclOp>
struct TimeConstantOpLowering : public OpConversionPattern<SclOp> {
  using OpConversionPattern<SclOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SclOp op, typename SclOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = this->typeConverter->convertType(op.getType());
    if (!resTy || !resTy.template isa<IntegerType>())
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resTy, rewriter.getIntegerAttr(resTy, op.getValue()));
    return success();
  }
};
using ConstantS5TimeOpLowering = TimeConstantOpLowering<scl::ConstantS5TimeOp>;
using ConstantTimeOpLowering = TimeConstantOpLowering<scl::ConstantTimeOp>;
using ConstantDateOpLowering = TimeConstantOpLowering<scl::ConstantDateOp>;
using ConstantTimeOfDayOpLowering = TimeConstantOpLowering<scl::ConstantTimeOfDayOp>;
using ConstantDateAndTimeOpLowering = TimeConstantOpLowering<scl::ConstantDateAndTimeOp>;

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
    registry.insert<math::MathDialect>();
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
  target.addLegalOp<scl::DebugPrintOp>();

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();

  // Compute FB instance layouts before conversion (the ops are erased).
  SclTypeConverter converter{computeFbLayouts(getOperation())};

  RewritePatternSet patterns(context);
  auto loopTargets = std::make_shared<LoopTerminatorTargets>();
  patterns.add<
      AddOpLowering, AndOpLowering, BitCastOpLowering, CallFbOpLowering,
      CallFcOpLowering, CaseOpLowering, ConditionOpLowering,
      ConstantDateAndTimeOpLowering, ConstantDateOpLowering,
      ConstantOpLowering, ConstantS5TimeOpLowering, ConstantTimeOfDayOpLowering,
      ConstantTimeOpLowering, DataBlockOpLowering, DialectCastOpLowering,
      DivOpLowering, EndOpLowering, EqualLowering, ExpOpLowering,
      FunctionBlockOpLowering, FunctionOpLowering,
      GetElementOpLowering, GetGlobalOpLowering, GetIndexOpLowering,
      GetVariableOpLowering, GreaterEqualLowering, GreaterThanLowering,
      IfThenElseOpLowering, IntegerCastOpLowering, LessEqualLowering,
      LessThanLowering, LoadOpLowering, ModOpLowering, MulOpLowering,
      NotEqualLowering, OrOpLowering, ReturnOpLowering,
      ReturnValueOpLowering, StoreOpLowering, SubOpLowering,
      TempVariableOpLowering, UnaryMinusOpLowering, UnaryNotOpLowering,
      UntilOpLowering, VariableOpLowering,
      XOrOpLowering>(converter, context);
  patterns.add<ForDoOpLowering, WhileDoOpLowering, RepeatOpLowering,
               ContinueOpLowering, ExitOpLowering>(converter, context,
                                                    loopTargets);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sclang::createLowerToStdPass() {
  return std::make_unique<SclToStdLoweringPass>();
}
