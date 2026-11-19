//====- LowerToStdScl.cpp - LowerToStd: scl dialect patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// scl-dialect conversion patterns of the LowerToStd pass (split out of
// LowerToStd.cpp for build parallelism — same patterns, same behavior).
//
//===----------------------------------------------------------------------===//

#include "LowerToStdInternal.h"

using namespace mlir;
using namespace mlir::sclang::lowering;

namespace {


// MARK: CallFcOpLowering

struct CallFcOpLowering : public OpConversionPattern<scl::CallFcOp> {
  using OpConversionPattern<scl::CallFcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::CallFcOp op, scl::CallFcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type returnType = getTypeConverter()->convertType(op.getType());
    // By-reference parameters (VAR_IN_OUT / VAR_OUTPUT of a function) arrive
    // as byte views (address<char> of an IDB member) but the callee's
    // parameter converts to a 0-d view of the element type: adapt with a
    // single reinterpret_cast from the base buffer, preserving the offset.
    SmallVector<Value, 4> args = adaptor.getArguments();
    auto module = moduleOf(op);
    if (auto callee = llvm::dyn_cast_or_null<func::FuncOp>(
            module.lookupSymbol(op.getCallee()))) {
      auto fnTy = callee.getFunctionType();
      for (auto [i, arg] : llvm::enumerate(args)) {
        if (i >= fnTy.getNumInputs())
          break;
        auto target = fnTy.getInput(i);
        auto argMem = llvm::dyn_cast<MemRefType>(arg.getType());
        auto tgtMem = llvm::dyn_cast<MemRefType>(target);
        if (!argMem || !tgtMem || arg.getType() == target)
          continue;
        auto loc = op.getLoc();
        auto meta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, arg);
        SmallVector<OpFoldResult, 1> sizes, strides;
        for (int64_t d : tgtMem.getShape()) {
          sizes.push_back(rewriter.getIndexAttr(d));
          strides.push_back(rewriter.getIndexAttr(1));
        }
        // The offset operand is dynamic, so the result type must carry a
        // dynamic offset too (MLIR verifier); rebuild an identity/static
        // layout as strided with dynamic offset.
        MemRefType rcTy = tgtMem;
        {
          int64_t tyOff = 0;
          SmallVector<int64_t, 4> tyStrides;
          (void)tgtMem.getStridesAndOffset(tyStrides, tyOff);
          if (!ShapedType::isDynamic(tyOff)) {
            SmallVector<int64_t, 4> ones(tgtMem.getRank(), 1);
            rcTy = MemRefType::get(tgtMem.getShape(), tgtMem.getElementType(),
                                   StridedLayoutAttr::get(rewriter.getContext(),
                                                          ShapedType::kDynamic,
                                                          ones));
          }
        }
        args[i] = rewriter.create<memref::ReinterpretCastOp>(
            loc, rcTy, meta.getBaseBuffer(), meta.getOffset(), sizes,
            strides);
      }
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(), returnType, args);
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
      mlir::sclang::withFPFastMath(rewriter.replaceOpWithNewOp<CmpFOp>(
      op, predF, adaptor.getLhs(), adaptor.getRhs()));
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
    // The result type must be the converted type; retyping the attribute
    // accordingly (a bare arith.constant with a !scl.*-typed attribute is not
    // legalizable by the LLVM conversion, e.g. !scl.char char literals).
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();
    auto attr = llvm::cast<TypedAttr>(op.getValue());
    if (attr.getType() != resTy) {
      if (auto ia = llvm::dyn_cast<IntegerAttr>(attr))
        attr = rewriter.getIntegerAttr(resTy, ia.getValue());
      else if (auto fa = llvm::dyn_cast<FloatAttr>(attr))
        attr = rewriter.getFloatAttr(resTy, fa.getValue());
      else
        return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resTy, attr);
    return success();
  }
};


// MARK: DialectCastOpLowering

/// Removes DialectCastOps when they are not necessary any more; performs a
/// scalar conversion (ext/trunc/sitofp/fptosi) when the types still differ.
struct DialectCastOpLowering : public OpConversionPattern<scl::DialectCastOp> {
  using OpConversionPattern<scl::DialectCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DialectCastOp op, scl::DialectCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Type targetTy = typeConverter->convertType(op.getType());

    if (value.getType() != targetTy) {
      // A byte view (converted !scl.address) cast to i64 yields its raw data
      // pointer: aligned base plus the view offset (same extraction the
      // builtin-FB lowerings use to pass instance DBs to the runtime).
      if (llvm::isa<MemRefType>(value.getType()) && targetTy.isInteger(64)) {
        auto loc = op.getLoc();
        auto meta =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, value);
        Value basePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, meta.getBaseBuffer());
        Value ptrIdx =
            rewriter.create<arith::AddIOp>(loc, basePtr, meta.getOffset());
        Value ptr = rewriter.create<arith::IndexCastOp>(loc, targetTy, ptrIdx);
        rewriter.replaceOp(op, ptr);
        return success();
      }
      // real -> integer with `round` (IEC 61131-3 ed.3 table 22): round to
      // the nearest integer first, then fptosi
      if (op.getRoundAttr() && llvm::isa<FloatType>(value.getType()) &&
          llvm::isa<IntegerType>(targetTy)) {
        Value rounded =
            rewriter.create<math::RoundOp>(op.getLoc(), value).getResult();
        rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, targetTy, rounded);
        return success();
      }
      if (Value converted = ::convertScalarTo(rewriter, op.getLoc(), value, targetTy)) {
        rewriter.replaceOp(op, converted);
        return success();
      }
      return failure();
    }
    rewriter.replaceOp(op, value);
    return success();
  }
};


// MARK: DebugPrintOpConversion

// scl.debug.print stays until the LLVM lowering (it needs a char* for the
// message); here we only convert its optional value operand to a std type,
// so the LLVM-stage lowering sees i16/f32/... instead of !scl.* types.
struct DebugPrintOpConversion
    : public OpConversionPattern<scl::DebugPrintOp> {
  using OpConversionPattern<scl::DebugPrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::DebugPrintOp op, scl::DebugPrintOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scl::DebugPrintOp>(op, op.getMsgAttr(),
                                                   adaptor.getValue());
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

    // Create the converted func op, preserving the visibility: external
    // stubs (empty body) are private, and a public func declaration is
    // rejected by the verifier.
    auto newFuncType = rewriter.getFunctionType(inputs, results);
    auto newFuncOp = rewriter.create<FuncOp>(loc, name, newFuncType);
    newFuncOp.setVisibility(op.getVisibility());

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();
    // An organization block models one scan cycle: read the process image
    // of the inputs on entry, write the outputs on exit (ADR 002).
    if (op->hasAttr("scl.organization_block")) {
      auto module = moduleOf(op);
      auto voidFn = rewriter.getFunctionType({}, {});
      auto updIn = getOrInsertRtFunc(rewriter, module, "scl_rt_update_inputs",
                                     voidFn);
      auto updOut = getOrInsertRtFunc(rewriter, module, "scl_rt_update_outputs",
                                      voidFn);
      rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
      rewriter.create<func::CallOp>(loc, updIn, TypeRange{}, ValueRange{});
      newFuncOp.getBody().walk([&](scl::ReturnOp ret) {
        rewriter.setInsertionPoint(ret);
        rewriter.create<func::CallOp>(ret.getLoc(), updOut, TypeRange{},
                                      ValueRange{});
      });
    }
    rewriter.eraseOp(op);
    return success();
  }
};


struct LoadOpLowering : public OpConversionPattern<scl::LoadOp> {
  using OpConversionPattern<scl::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::LoadOp op, scl::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    // Direct address (memory area / DB): runtime C ABI call (ADR 002).
    if (auto dt = llvm::dyn_cast<scl::DirectType>(addrElemTy(op.getAddress()))) {
      Value loaded = lowerDirectLoad(rewriter, op.getLoc(), dt, resTy,
                                     moduleOf(op));
      rewriter.replaceOp(op, loaded);
      return success();
    }
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
      mlir::sclang::withFPFastMath(rewriter.replaceOpWithNewOp<LoweredFloatOp>(
          op, operands[0], operands[1]));
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


struct StoreOpLowering : public OpConversionPattern<scl::StoreOp> {
  using OpConversionPattern<scl::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::StoreOp op, scl::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Direct address (memory area / DB): runtime C ABI call (ADR 002).
    if (auto dt = llvm::dyn_cast<scl::DirectType>(addrElemTy(op.getLhs()))) {
      if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getRhs(),
                                  moduleOf(op), op)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    }
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
    auto memref = llvm::dyn_cast<MemRefType>(resultType);
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
    auto zero = constIndex(rewriter, loc, 0);
    if (memref.getRank() == 1) {
      if (memref.getShape()[0] != ShapedType::kDynamic) {
        // struct window: alloca the whole buffer and cast to the window type
        auto plain = MemRefType::get({memref.getShape()[0]},
                                     memref.getElementType());
        auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
        rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
            op, memref, alloca, rcOffsetOfr(rewriter, memref, zero),
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(memref.getShape()[0])},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
        return success();
      }
      // i8 address: rank-1 byte view holding a single byte. An instance-DB
      // address allocates the FB's whole instance buffer instead.
      int64_t allocSize = 1;
      if (auto idbTy = llvm::dyn_cast<scl::InstanceDbType>(addrElemTy(op.getResult()))) {
        auto &layouts =
            static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
        auto it = layouts.find(idbTy.getFbSymbol());
        if (it != layouts.end())
          allocSize = it->second.size;
      }
      auto plain = MemRefType::get({allocSize}, memref.getElementType());
      auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
      rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
          op, memref, alloca, rcOffsetOfr(rewriter, memref, zero),
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(allocSize)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
      return success();
    }
    auto plain = MemRefType::get({}, memref.getElementType());
    auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memref, alloca, rcOffsetOfr(rewriter, memref, zero),
        ArrayRef<OpFoldResult>{}, ArrayRef<OpFoldResult>{});
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
          mlir::sclang::withFPFastMath(rewriter.replaceOpWithNewOp<NegFOp>(op, adaptor.getRhs()));
          return success();
        })
        .Case<IntegerType>([&](auto elementType) {
          auto loc = op.getLoc();
          auto zero = rewriter.create<ConstantIntOp>(loc, elementType, 0);
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
    Value v = adaptor.getRhs();
    if (v.getType().isInteger(1)) {
      // logical NOT on a bool
      auto i1 = rewriter.getIntegerType(1);
      auto falseVal = rewriter.create<ConstantIntOp>(loc, i1, 0);
      auto trueVal = rewriter.create<ConstantIntOp>(loc, i1, 1);
      rewriter.replaceOpWithNewOp<SelectOp>(op, v, falseVal, trueVal);
      return success();
    }
    // bitwise NOT on a bit string (BYTE/WORD/DWORD): x XOR all-ones
    auto allOnes = rewriter.create<ConstantIntOp>(
        loc, -1, llvm::cast<IntegerType>(v.getType()).getWidth());
    rewriter.replaceOpWithNewOp<XOrIOp>(op, v, allOnes);
    return success();
  }
};



// MARK: ConfigOpLowering (IEC CONFIGURATION metadata, ADR 004)

/// Erased: the CONFIGURATION's VAR_GLOBAL already became a global DataBlockOp
/// and the TASK/PROGRAM bindings are metadata only (the driver/harness reads
/// them for scheduling).
struct ConfigOpLowering : public OpConversionPattern<scl::ConfigOp> {
  using OpConversionPattern<scl::ConfigOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scl::ConfigOp op, scl::ConfigOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
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
    if (auto idbTy = llvm::dyn_cast<scl::InstanceDbType>(op.getType())) {
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
      // 0-d memref. NB: the global always keeps a plain identity-layout type
      // because the LLVM lowering of memref.global requires identity maps;
      // memref.get_global results that need a strided window type are cast in
      // GetGlobalOpLowering.
      if (llvm::isa<MemRefType>(elemTy))
        memrefTy = elemTy;
      else
        memrefTy = MemRefType::get({}, elemTy);
    }
    // An initializer makes the LLVM global defined (not external), so the
    // JIT can resolve its symbol. Initial values are stored as a
    // DenseElementsAttr over the equivalent ranked tensor type. For an
    // instance DB, bake the FB members' cold-start initial values into the
    // byte buffer; everything else is zero-filled.
    auto memref = llvm::cast<MemRefType>(memrefTy);
    Attribute initValue = nullptr;
    if (memref.getElementType().isIntOrFloat() ||
        llvm::isa<IndexType>(memref.getElementType())) {
      auto tensorTy = RankedTensorType::get(memref.getShape(),
                                            memref.getElementType());
      if (memref.getElementType().isInteger(8) && idbTyVal) {
        auto &layout = layouts.find(idbTyVal.getFbSymbol())->second;
        SmallVector<Attribute, 8> bytes;
        bytes.reserve(layout.size);
        std::vector<uint8_t> buf(layout.size, 0);
        // Bake cold-start initializers into the byte buffer, recursing into
        // FB-typed members (their builtin/user layouts may carry their own
        // initializers, e.g. F_TRIG's edge memory starts TRUE).
        std::function<void(const FbLayout &, int64_t)> bake =
            [&](const FbLayout &lay, int64_t baseOff) {
          for (auto &m : lay.members) {
            if (auto idbM = llvm::dyn_cast<scl::InstanceDbType>(m.type)) {
              auto it = layouts.find(idbM.getFbSymbol());
              if (it != layouts.end())
                bake(it->second, baseOff + m.offset);
            }
            if (m.initBits) {
              int64_t msize = byteSize(m.type, layouts);
              // big-endian (ADR 001): byte i of the member window holds
              // bits 8*(msize-1-i).. of the value. Extract byte-wise from
              // the raw words (the pattern may be wider than 64 bits for
              // array/struct initializers).
              auto words = m.initBits->getRawData();
              unsigned nwords = m.initBits->getNumWords();
              for (int64_t i = 0; i < msize; ++i) {
                unsigned bitPos = (unsigned)(8 * (msize - 1 - i));
                unsigned w = bitPos / 64, off = bitPos % 64;
                uint64_t v = w < nwords ? (words[w] >> off) : 0;
                if (off > 56 && w + 1 < nwords)
                  v |= words[w + 1] << (64 - off);
                buf[baseOff + m.offset + i] = (uint8_t)(v & 0xFF);
              }
            } else if (m.initStr) {
              // String member (flat char array): store the literal chars,
              // zero-padded (the buffer starts zeroed).
              if (auto arrTy = llvm::dyn_cast<scl::ArrayType>(m.type))
                if (llvm::isa<scl::IntegerType>(arrTy.getElementType()) &&
                    llvm::cast<scl::IntegerType>(arrTy.getElementType()).getWidth() == 8) {
                  int64_t count = 1;
                  for (auto &dim : arrTy.getDimensions())
                    count *= dim.second - dim.first + 1;
                  for (int64_t i = 0;
                       i < count && i < (int64_t)m.initStr->size(); ++i)
                    buf[baseOff + m.offset + i] = (uint8_t)(*m.initStr)[i];
                }
            }
          }
        };
        bake(layout, 0);
        // DB-level initial values (scl.db `init` attr) override the FB
        // members' cold-start values.
        if (auto initOpt = op.getInit()) {
          if (auto initDict = llvm::dyn_cast<mlir::DictionaryAttr>(*initOpt)) {
            for (auto na : initDict.getValue()) {
              std::string mname = na.getName().str();
              auto ia = llvm::dyn_cast<IntegerAttr>(na.getValue());
              if (!ia)
                continue;
              for (auto &m : layout.members) {
                if (m.name != mname)
                  continue;
                int64_t msize = byteSize(m.type, layouts);
                uint64_t v = ia.getValue().getZExtValue();
                for (int64_t i = 0; i < msize; ++i)
                  buf[m.offset + i] =
                      (uint8_t)(v >> (8 * (msize - 1 - i)));
              }
            }
          }
        }
        auto i8 = memref.getElementType();
        for (uint8_t byte : buf)
          bytes.push_back(rewriter.getIntegerAttr(i8, byte));
        initValue = DenseElementsAttr::get(tensorTy, bytes);
      } else {
        initValue = DenseElementsAttr::get(
            tensorTy, rewriter.getZeroAttr(memref.getElementType()));
        // A struct-typed DB (e.g. an IEC CONFIGURATION's VAR_GLOBAL block,
        // ADR 004) can carry `init` entries too: bake them big-endian into
        // the zero-filled byte buffer at the members' struct offsets.
        if (auto initOpt = op.getInit())
          if (auto st = llvm::dyn_cast<scl::StructType>(op.getType()))
            if (auto initDict = llvm::dyn_cast<mlir::DictionaryAttr>(*initOpt)) {
              SmallVector<Attribute, 8> bytes;
              int64_t total = memref.getShape().empty() ? 0 : memref.getShape()[0];
              std::vector<uint8_t> buf((size_t)total, 0);
              int64_t off = 0;
              for (auto [mname, mty] :
                   llvm::zip(st.getElementNames(), st.getElementTypes())) {
                int64_t msize = byteSize(mty, layouts);
                for (auto na : initDict.getValue()) {
                  if (na.getName().str() != mname.str())
                    continue;
                  auto ia = llvm::dyn_cast<IntegerAttr>(na.getValue());
                  if (!ia)
                    continue;
                  uint64_t v = ia.getValue().getZExtValue();
                  for (int64_t i = 0; i < msize; ++i)
                    buf[off + i] = (uint8_t)(v >> (8 * (msize - 1 - i)));
                }
                off += msize;
              }
              auto i8 = memref.getElementType();
              for (uint8_t byte : buf)
                bytes.push_back(rewriter.getIntegerAttr(i8, byte));
              initValue = DenseElementsAttr::get(tensorTy, bytes);
            }
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
    // The global itself is stored with a plain identity-layout memref type
    // (the LLVM lowering of memref.global requires identity maps), so
    // memref.get_global must use that plain type; cast the result to the
    // (possibly strided) address type when it differs.
    Value result;
    if (auto memrefRes = llvm::dyn_cast<MemRefType>(resTy);
        memrefRes && memrefRes.getLayout()) {
      // An instance-DB address converts to a dynamic-size byte view; the
      // global itself must use the FB layout's static size.
      SmallVector<int64_t, 1> shape(memrefRes.getShape());
      if (auto addrTy = llvm::dyn_cast<scl::AddressType>(op.getResult().getType()
                            )) {
        if (auto idbTy =
                llvm::dyn_cast<scl::InstanceDbType>(addrTy.getElementType())) {
          auto &layouts = static_cast<const SclTypeConverter *>(typeConverter)
                              ->fbLayouts;
          auto it = layouts.find(idbTy.getFbSymbol());
          if (it == layouts.end())
            return failure();
          shape[0] = it->second.size;
        }
      }
      auto plainTy = MemRefType::get(shape, memrefRes.getElementType());
      Value global = rewriter.create<memref::GetGlobalOp>(op.getLoc(), plainTy,
                                                          op.getNameAttr());
      result = rewriter.create<memref::CastOp>(op.getLoc(), resTy, global);
    } else {
      result = rewriter.create<memref::GetGlobalOp>(op.getLoc(), resTy,
                                                    op.getNameAttr());
    }
    rewriter.replaceOp(op, result);
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
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    // The declared array type: from the attr when the base is a byte view
    // (array member of a FB/struct), else from the address element type.
    scl::ArrayType arrayTy;
    if (op.getArrayType().has_value())
      arrayTy = llvm::cast<scl::ArrayType>(op.getArrayType());
    else
      arrayTy = llvm::cast<scl::ArrayType>(
          addrElemTy(op.getAddr()));
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
      if (!llvm::isa<IntegerType, IndexType>(idx.getType()))
        return failure();
      Value adj = llvm::isa<IndexType>(idx.getType())
                      ? idx
                      : (Value)rewriter.create<arith::IndexCastOp>(loc, indexTy, idx);
      if (dims[i].first != 0) {
        auto minC = constIndex(rewriter, loc, dims[i].first);
        adj = rewriter.create<arith::SubIOp>(loc, adj, minC);
      }
      Value contrib = adj;
      if (strides[i] != 1) {
        auto strideC =
            constIndex(rewriter, loc, strides[i]);
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
    auto baseMemTy = llvm::cast<MemRefType>(base.getType());
    // Byte path: the base is a rank-1 i8 view (array member of a FB/struct)
    // or a plain char array buffer. Compute the byte offset and create a
    // single cast from the root buffer.
    if (baseMemTy.getRank() == 1 && baseMemTy.getElementType().isInteger(8)) {
      int64_t elemBytes = byteSize(arrayTy.getElementType(), layouts);
      if (elemBytes <= 0)
        return failure();
      Value byteFlat = flat;
      if (elemBytes != 1) {
        auto scale = constIndex(rewriter, loc, elemBytes);
        byteFlat = rewriter.create<arith::MulIOp>(loc, flat, scale);
      }
      bool compositeElem =
          llvm::isa<scl::StructType, scl::ArrayType>(arrayTy.getElementType());
      // A static-size window is only correct when the result is an
      // address<composite> (plain array variable, no arrayType attr); the
      // byte-view path (array member of a FB/struct) yields address<i8>,
      // which converts to a dynamic-size view.
      bool staticSize = compositeElem && !op.getArrayType().has_value();
      // Fold in the byte index and the base view's own runtime offset.
      Value dynOff;
      int64_t constOff;
      Value root = resolveByteViewRoot(base, dynOff, constOff, rewriter, loc);
      if (!root || !llvm::isa<MemRefType>(root.getType()))
        return failure();
      auto meta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, root);
      Value off = combineOffsets(rewriter, loc, constOff, dynOff, nullptr);
      off = rewriter.create<arith::AddIOp>(loc, off, byteFlat);
      off = rewriter.create<arith::AddIOp>(loc, off, meta.getOffset());
      Value view = makeByteView(rewriter, loc, meta.getBaseBuffer(), off,
                                elemBytes, ctx, staticSize);
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
    Value toAddr = adaptor.getTo(); // address of the bound cell (re-read below)
    Value step = adaptor.getBy();
    if (!step) {
      auto ty = init.getType();
      auto attr = llvm::isa<FloatType>(ty)
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

    // cond: iv = load var; to = reload bound cell (the body may have
    // re-stored it); cond = step > 0 ? iv <= to : iv >= to
    rewriter.setInsertionPointToStart(condB);
    Value iv = loadFromByteBuffer(rewriter, loc, var, init.getType());
    if (!iv)
      return failure();
    Value to = loadFromByteBuffer(rewriter, loc, toAddr, init.getType());
    if (!to)
      return failure();
    auto zero = constInt(rewriter, loc, step.getType(), 0);
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
    Value next = llvm::isa<FloatType>(iv2.getType())
                     ? (Value)mlir::sclang::withFPFastMath(rewriter.create<arith::AddFOp>(loc, iv2, step))
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
    if (!llvm::isa<IntegerType>(sel.getType()))
      return failure();
    auto i32 = rewriter.getI32Type();
    if (sel.getType() != i32) {
      auto width = sel.getType().getIntOrFloatBitWidth();
      if (width < 32)
        // width 1 (bool): extsi would sign-extend true to -1; case values
        // are 1/0, so zero-extend
        if (width == 1)
          sel = rewriter.create<arith::ExtUIOp>(loc, i32, sel);
        else
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
      auto values = llvm::cast<DenseI32ArrayAttr>(caseValuesAttr[i]);
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
    newFuncOp.setVisibility(op.getVisibility());
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


// MARK: GetDirectOpLowering

/// Lower `scl.getdirect` to a packed i64 placeholder. Loads/stores through a
/// direct address lower to runtime C ABI calls (ADR 002) and never use this
/// value; it only exists so the type conversion has a concrete type.
struct GetDirectOpLowering : public OpConversionPattern<scl::GetDirectOp> {
  using OpConversionPattern<scl::GetDirectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scl::GetDirectOp op, scl::GetDirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    auto dt = llvm::cast<scl::DirectType>(
        addrElemTy(op.getResult()));
    int64_t packed = ((int64_t)(uint8_t)dt.getArea() << 56) |
                     ((int64_t)(uint32_t)dt.getDb() << 40) |
                     ((int64_t)(uint32_t)dt.getByte() << 16) |
                     ((int64_t)(uint32_t)dt.getBit() << 8) | (uint32_t)dt.getWidth();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resTy, rewriter.getIntegerAttr(resTy, packed));
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
    auto addrTy = llvm::cast<scl::AddressType>(op.getAddr().getType());
    auto idbTy = llvm::dyn_cast<scl::InstanceDbType>(addrTy.getElementType());
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
    // Rank-1 byte view of the member window. makeByteViewFromSource folds in
    // the address's own runtime offset (an instance-DB parameter passed as a
    // local-member view), which a direct reinterpret_cast would drop.
    int64_t size = byteSize(member->type, layouts);
    Value view = makeByteViewFromSource(rewriter, loc, adaptor.getAddr(),
                                        member->offset, size, ctx);
    if (!view)
      return failure();
    rewriter.replaceOp(op, view);
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
    auto addrTy = llvm::cast<scl::AddressType>(op.getAddr().getType());
    auto &layouts =
        static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
    // Byte offset and declared type of the member: from the FB's layout when
    // the base is a byte view of an instance DB (fb attr; also covers builtin
    // standard FBs), otherwise from the declared struct type of the base.
    int64_t offset = 0;
    Type memberTy;
    if (op.getFb().has_value()) {
      auto it = layouts.find(*op.getFb());
      if (it == layouts.end())
        return failure();
      for (auto &m : it->second.members) {
        // IEC 61131-3 identifiers are case-insensitive
        if (llvm::StringRef(m.name).equals_insensitive(op.getName())) {
          offset = m.offset;
          memberTy = m.type;
          break;
        }
      }
      if (!memberTy)
        return failure();
    } else {
      scl::StructType structTy;
      if (auto st = llvm::dyn_cast<scl::StructType>(addrTy.getElementType()))
        structTy = st;
      else if (op.getBaseType().has_value())
        structTy = llvm::cast<scl::StructType>(op.getBaseType());
      if (!structTy)
        return failure();
      for (auto [name, elem] :
           llvm::zip(structTy.getElementNames(), structTy.getElementTypes())) {
        // IEC 61131-3 identifiers are case-insensitive
        if (name.getValue().equals_insensitive(op.getName()))
          memberTy = elem;
        if (memberTy)
          break;
        offset += byteSize(elem, layouts);
      }
      if (!memberTy)
        return failure();
    }
    auto ctx = op.getContext();
    // The address of a struct is the struct buffer memref itself (or a byte
    // view into a composite buffer for nested members).
    int64_t size = byteSize(memberTy, layouts);
    if (size <= 0)
      return failure();
    Value view = makeByteViewFromSource(rewriter, loc, adaptor.getAddr(),
                                        offset, size, ctx);
    if (!view)
      return failure();
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
    // Builtin standard FBs (timers TON/TOF/TP, counters CTU/CTD/CTUD, edges
    // R_TRIG/F_TRIG/SR/RS) lower to a runtime call on the instance DB
    // buffer, not a func.call (no FunctionBlockOp needed; fixed layouts in
    // the FbLayoutCache match the runtime's interpretation). A user-defined
    // FunctionBlockOp with the same name wins (e.g. an IEC PROGRAM named
    // "counter" — the TIMER/COUNTER aliases are an S7-ism).
    std::string fbUp = llvm::StringRef(op.getFb()).upper();
    if (op->getParentOfType<ModuleOp>().lookupSymbol<scl::FunctionBlockOp>(
            op.getFb()))
      fbUp = "";
    const char *rtName = nullptr;
    int mode = -1;
    if (fbUp == "TON") { rtName = "scl_rt_timer_fb"; mode = 0;
    } else if (fbUp == "TOF") { rtName = "scl_rt_timer_fb"; mode = 1;
    } else if (fbUp == "TP") { rtName = "scl_rt_timer_fb"; mode = 2;
    } else if (fbUp == "CTU") { rtName = "scl_rt_counter_fb"; mode = 0;
    } else if (fbUp == "CTD") { rtName = "scl_rt_counter_fb"; mode = 1;
    } else if (fbUp == "CTUD") { rtName = "scl_rt_counter_fb"; mode = 2;
    } else if (fbUp == "R_TRIG") { rtName = "scl_rt_edge_fb"; mode = 0;
    } else if (fbUp == "F_TRIG") { rtName = "scl_rt_edge_fb"; mode = 1;
    } else if (fbUp == "SR") { rtName = "scl_rt_edge_fb"; mode = 2;
    } else if (fbUp == "RS") { rtName = "scl_rt_edge_fb"; mode = 3;
    }
    if (rtName) {
      auto loc = op.getLoc();
      ModuleOp module = moduleOf(op);
      auto i8 = rewriter.getI8Type();
      auto i64 = rewriter.getI64Type();
      auto fnTy = rewriter.getFunctionType({i8, i64}, {});
      auto rtFn = getOrInsertRtFunc(rewriter, module, rtName, fnTy);
      // The instance view's data pointer as i64: aligned base pointer plus
      // the view's own offset (nonzero for local FB-typed members).
      Value idb = adaptor.getIdb();
      auto meta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, idb);
      Value basePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, meta.getBaseBuffer());
      Value ptrIdx =
          rewriter.create<arith::AddIOp>(loc, basePtr, meta.getOffset());
      Value ptr = rewriter.create<arith::IndexCastOp>(loc, i64, ptrIdx);
      Value modeC = constInt(rewriter, loc, i8, (int8_t)mode);
      rewriter.create<func::CallOp>(loc, rtFn, TypeRange{}, ValueRange{modeC, ptr});
      rewriter.eraseOp(op);
      return success();
    }
    Value idb = adaptor.getIdb();
    // The callee's instance-DB parameter and the caller's instance address
    // both convert to the same strided byte-view type, so they match
    // directly (global DBs arrive via memref.cast in GetGlobalOpLowering,
    // local FB-typed members are already byte views).
    auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getFb(),
                                              TypeRange{}, ValueRange{idb});
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
    auto resTy = llvm::cast<IntegerType>(typeConverter->convertType(op.getType()));
    auto srcTy = llvm::cast<IntegerType>(adaptor.getValue().getType());
    if (resTy.getWidth() > srcTy.getWidth()) {
      // zext: zero-extend (unsigned source, e.g. byte/word); default is
      // sign-extension (signed int/dint).
      if (op.getZextAttr())
        rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, resTy,
                                                    adaptor.getValue());
      else
        rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resTy,
                                                    adaptor.getValue());
    } else if (resTy.getWidth() < srcTy.getWidth())
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
    if (!resTy || !llvm::isa<IntegerType>(resTy))
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

namespace mlir::sclang::lowering {

void populateSclToStdPatterns(RewritePatternSet &patterns,
                              SclTypeConverter &converter, MLIRContext *context,
                              LoopTargetsPtr loopTargets) {
  patterns.add<
      AddOpLowering, AndOpLowering, BitCastOpLowering, CallFbOpLowering,
      CallFcOpLowering, CaseOpLowering, ConditionOpLowering,
      ConstantDateAndTimeOpLowering, ConstantDateOpLowering,
      ConstantOpLowering, ConstantS5TimeOpLowering, ConstantTimeOfDayOpLowering,
      ConfigOpLowering, ConstantTimeOpLowering, DataBlockOpLowering,
      DebugPrintOpConversion, DialectCastOpLowering,
      DivOpLowering, EndOpLowering, EqualLowering, ExpOpLowering,
      FunctionBlockOpLowering, FunctionOpLowering,
      GetDirectOpLowering, GetElementOpLowering, GetGlobalOpLowering,
      GetIndexOpLowering,
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
}

} // namespace mlir::sclang::lowering
