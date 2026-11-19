//====- LowerToStdAwl.cpp - LowerToStd: awl dialect patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// awl-dialect conversion patterns of the LowerToStd pass (ADR 003), split
// out of LowerToStd.cpp for build parallelism — same patterns, same
// behavior.
//
//===----------------------------------------------------------------------===//

#include "LowerToStdInternal.h"

using namespace mlir;
using namespace mlir::sclang::lowering;

namespace {


struct AwlDirectOpLowering : public OpConversionPattern<awl::DirectOp> {
  using OpConversionPattern<awl::DirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::DirectOp op, awl::DirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    auto dt = llvm::cast<scl::DirectType>(op.getResult().getType());
    int64_t packed = ((int64_t)(uint8_t)dt.getArea() << 56) |
                     ((int64_t)(uint32_t)dt.getDb() << 40) |
                     ((int64_t)(uint32_t)dt.getByte() << 16) |
                     ((int64_t)(uint32_t)dt.getBit() << 8) | (uint32_t)dt.getWidth();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resTy, rewriter.getIntegerAttr(resTy, packed));
    return success();
  }
};


struct AwlConstOpLowering : public OpConversionPattern<awl::ConstOp> {
  using OpConversionPattern<awl::ConstOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ConstOp op, awl::ConstOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
        llvm::cast<TypedAttr>(op.getValue()));
    return success();
  }
};


// LC T n -> @scl_rt_timer_lc(timer: i32) -> i64
struct AwlTimerLcOpLowering : public OpConversionPattern<awl::TimerLcOp> {
  using OpConversionPattern<awl::TimerLcOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::TimerLcOp op, awl::TimerLcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_lc",
                                rewriter.getFunctionType({i32}, {rewriter.getI64Type()}));
    auto tC = constInt(rewriter, loc, i32, op.getTimer());
    auto v = rewriter.create<func::CallOp>(loc, fn, TypeRange{rewriter.getI64Type()},
                                           ValueRange{tC})
                 .getResult(0);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, rewriter.getI32Type(), v);
    return success();
  }
};


struct AwlLoadOpLowering : public OpConversionPattern<awl::LoadOp> {
  using OpConversionPattern<awl::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::LoadOp op, awl::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = llvm::cast<scl::DirectType>(op.getAddr().getType());
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    Value v = lowerDirectLoad(rewriter, op.getLoc(), dt, resTy,
                              moduleOf(op), op);
    rewriter.replaceOp(op, v);
    return success();
  }
};


struct AwlStoreOpLowering : public OpConversionPattern<awl::StoreOp> {
  using OpConversionPattern<awl::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StoreOp op, awl::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = llvm::cast<scl::DirectType>(op.getAddr().getType());
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getValue(),
                                moduleOf(op), op)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};


template <typename RloOp, bool isOr>
struct AwlRloLowering : public OpConversionPattern<RloOp> {
  using OpConversionPattern<RloOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(RloOp op, typename RloOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = llvm::cast<scl::DirectType>(op.getAddr().getType());
    auto i1 = rewriter.getI1Type();
    Value bit = lowerDirectLoad(rewriter, op.getLoc(), dt, i1,
                                moduleOf(op));
    if (op.getNegate()) {
      auto trueC = constInt(rewriter, op.getLoc(), i1, 1);
      bit = rewriter.create<arith::XOrIOp>(op.getLoc(), bit, trueC);
    }
    Value r = isOr ? (Value)rewriter.create<arith::OrIOp>(op.getLoc(),
                                                          adaptor.getRlo(), bit)
                   : (Value)rewriter.create<arith::AndIOp>(op.getLoc(),
                                                           adaptor.getRlo(), bit);
    rewriter.replaceOp(op, r);
    return success();
  }
};


struct AwlNotOpLowering : public OpConversionPattern<awl::NotOp> {
  using OpConversionPattern<awl::NotOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::NotOp op, awl::NotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    auto trueC = constInt(rewriter, op.getLoc(), i1, 1);
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getRlo(), trueC);
    return success();
  }
};


struct AwlAssignOpLowering : public OpConversionPattern<awl::AssignOp> {
  using OpConversionPattern<awl::AssignOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AssignOp op, awl::AssignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = llvm::cast<scl::DirectType>(op.getAddr().getType());
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getRlo(),
                                moduleOf(op), op)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};


template <typename SetResetOp, bool set>
struct AwlSetResetLowering : public OpConversionPattern<SetResetOp> {
  using OpConversionPattern<SetResetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SetResetOp op, typename SetResetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = llvm::cast<scl::DirectType>(op.getAddr().getType());
    auto i1 = rewriter.getI1Type();
    auto v = constInt(rewriter, op.getLoc(), i1, set ? 1 : 0);
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, v,
                                moduleOf(op), op)))
      return failure();
    rewriter.replaceOp(op, v);
    return success();
  }
};


template <typename AwlOp, typename LoweredFloatOp, typename LoweredIntOp>
struct AwlArithLowering : public OpConversionPattern<AwlOp> {
  using OpConversionPattern<AwlOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AwlOp op, typename AwlOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = this->typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    Value l = convertScalarTo(rewriter, op.getLoc(), adaptor.getLhs(), resTy);
    Value r = convertScalarTo(rewriter, op.getLoc(), adaptor.getRhs(), resTy);
    if (!l || !r)
      return failure();
    if (llvm::isa<FloatType>(resTy)) {
      // awlsim computes real arith in double and rounds to f32 on store,
      // with inf/zero-divisor special cases in *R and /R that also set the
      // STW — route Add/Sub/Mul/Div on floats through the runtime helper.
      if constexpr (std::is_same_v<LoweredFloatOp, arith::AddFOp> ||
                    std::is_same_v<LoweredFloatOp, arith::SubFOp> ||
                    std::is_same_v<LoweredFloatOp, arith::MulFOp> ||
                    std::is_same_v<LoweredFloatOp, arith::DivFOp>) {
        constexpr uint8_t opc =
            std::is_same_v<LoweredFloatOp, arith::AddFOp> ? 0 :
            std::is_same_v<LoweredFloatOp, arith::SubFOp> ? 1 :
            std::is_same_v<LoweredFloatOp, arith::MulFOp> ? 2 : 3;
        auto f64 = rewriter.getF64Type();
        auto i8 = rewriter.getI8Type();
        Value ld = convertScalarTo(rewriter, op.getLoc(), adaptor.getLhs(), f64);
        Value rd = convertScalarTo(rewriter, op.getLoc(), adaptor.getRhs(), f64);
        if (!ld || !rd)
          return failure();
        ModuleOp module = moduleOf(op);
        auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_arith_f",
                                    rewriter.getFunctionType({i8, f64, f64}, {f64}));
        Value opcC = constInt(rewriter, op.getLoc(), i8, opc);
        Value res = rewriter.create<func::CallOp>(op.getLoc(), fn,
                                                  TypeRange{f64}, ValueRange{opcC, ld, rd})
                        .getResult(0);
        res = convertScalarTo(rewriter, op.getLoc(), res, resTy); // round to f32
        rewriter.replaceOp(op, res);
        return success();
      }
      mlir::sclang::withFPFastMath(
          rewriter.replaceOpWithNewOp<LoweredFloatOp>(op, resTy, l, r));
    } else {
      rewriter.replaceOpWithNewOp<LoweredIntOp>(op, resTy, l, r);
    }
    return success();
  }
};

using AwlAddLowering = AwlArithLowering<awl::AddOp, arith::AddFOp, arith::AddIOp>;

using AwlSubLowering = AwlArithLowering<awl::SubOp, arith::SubFOp, arith::SubIOp>;

using AwlMulLowering = AwlArithLowering<awl::MulOp, arith::MulFOp, arith::MulIOp>;

using AwlDivLowering = AwlArithLowering<awl::DivOp, arith::DivFOp, arith::DivSIOp>;

using AwlAndLowering = AwlArithLowering<awl::AndOp, arith::AndIOp, arith::AndIOp>;

using AwlOrLowering = AwlArithLowering<awl::OrOp, arith::OrIOp, arith::OrIOp>;

using AwlXOrLowering = AwlArithLowering<awl::XOrOp, arith::XOrIOp, arith::XOrIOp>;


/// SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA: shift/rotate ACCU1 (i32).
/// Word variants touch only the low 16 bits and keep the high word; ssi/ssd
/// are sign-filling; the amount is clamped to the operand width (S7: counts
/// >= width behave like the width; rotates use amount % 32). rlda/rrda are
/// approximated as rld/rrd (the carry bit is not tracked by the sim).
struct AwlShiftOpLowering : public OpConversionPattern<awl::ShiftOp> {
  using OpConversionPattern<awl::ShiftOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ShiftOp op, awl::ShiftOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i32 = rewriter.getI32Type();
    Value v = adaptor.getValue();
    Value n = adaptor.getAmount();
    auto c = [&](int32_t x) {
      return constInt(rewriter, loc, i32, x);
    };
    auto bin = [&](Value a, Value b, auto mk) -> Value {
      return rewriter.create<decltype(mk)>(loc, i32, a, b);
    };
    std::string k = op.getKind().str();
    Value r;
    if (k == "sld" || k == "srd") { // 32-bit logical, count >= 32 -> 0
      Value cnt = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, n, c(31));
      Value amt = rewriter.create<arith::SelectOp>(loc, cnt, n, c(31));
      Value s = k == "sld" ? bin(v, amt, arith::ShLIOp{})
                           : bin(v, amt, arith::ShRUIOp{});
      Value ge32 = rewriter.create<arith::CmpIOp>(loc,
          arith::CmpIPredicate::uge, n, c(32));
      r = rewriter.create<arith::SelectOp>(loc, ge32, c(0), s);
    } else if (k == "ssd") { // 32-bit arithmetic, count >= 32 -> sign fill
      Value cnt = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, n, c(31));
      Value amt = rewriter.create<arith::SelectOp>(loc, cnt, n, c(31));
      Value s = rewriter.create<arith::ShRSIOp>(loc, i32, v, amt);
      Value fill = rewriter.create<arith::ShRSIOp>(loc, i32, v, c(31));
      Value ge32 = rewriter.create<arith::CmpIOp>(loc,
          arith::CmpIPredicate::uge, n, c(32));
      r = rewriter.create<arith::SelectOp>(loc, ge32, fill, s);
    } else if (k == "slw" || k == "srw" || k == "ssi") { // 16-bit low word
      Value hi = rewriter.create<arith::AndIOp>(loc, i32, v, c(-65536));
      Value cnt = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, n, c(16));
      Value amt = rewriter.create<arith::SelectOp>(loc, cnt, n, c(16));
      Value lo = rewriter.create<arith::AndIOp>(loc, i32, v, c(0xFFFF));
      if (k == "slw") {
        Value s = rewriter.create<arith::ShLIOp>(loc, i32, v, amt);
        r = rewriter.create<arith::OrIOp>(loc, i32, hi,
               rewriter.create<arith::AndIOp>(loc, i32, s, c(0xFFFF)));
      } else if (k == "srw") {
        Value s = rewriter.create<arith::ShRUIOp>(loc, i32, lo, amt);
        r = rewriter.create<arith::OrIOp>(loc, i32, hi, s);
      } else { // ssi: arithmetic shift right of the sign-extended low word
        auto i16 = rewriter.getI16Type();
        Value t = rewriter.create<arith::TruncIOp>(loc, i16, lo);
        Value s = rewriter.create<arith::ExtSIOp>(loc, i32, t);
        Value sh = rewriter.create<arith::ShRSIOp>(loc, i32, s, amt);
        r = rewriter.create<arith::OrIOp>(loc, i32, hi,
               rewriter.create<arith::AndIOp>(loc, i32, sh, c(0xFFFF)));
      }
    } else if (k == "rlda" || k == "rrda") {
      // rotate by exactly 1 through the carry bit; the sim does not track
      // the STW carry, so the incoming carry is approximated as 0
      r = k == "rlda" ? bin(v, c(1), arith::ShLIOp{})
                      : bin(v, c(1), arith::ShRUIOp{});
    } else { // rld/rrd
      bool left = k == "rld";
      Value m = rewriter.create<arith::AndIOp>(loc, i32, n, c(31));
      // (32 - m) & 31: 0 when m == 0, so a rotate by 0 is the identity and
      // never shifts by the poisonous 32
      Value om = rewriter.create<arith::AndIOp>(
          loc, i32, rewriter.create<arith::SubIOp>(loc, i32, c(32), m), c(31));
      Value a, b;
      if (left) {
        a = rewriter.create<arith::ShLIOp>(loc, i32, v, m);
        b = rewriter.create<arith::ShRUIOp>(loc, i32, v, om);
      } else {
        a = rewriter.create<arith::ShRUIOp>(loc, i32, v, m);
        b = rewriter.create<arith::ShLIOp>(loc, i32, v, om);
      }
      r = rewriter.create<arith::OrIOp>(loc, i32, a, b);
    }
    rewriter.replaceOp(op, r);
    return success();
  }
};


/// ABS/NEGR/SIN/COS/TAN/ASIN/ACOS/ATAN/SQRT/EXP/LN: unary real math on f32.
struct AwlRealMathOpLowering : public OpConversionPattern<awl::RealMathOp> {
  using OpConversionPattern<awl::RealMathOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::RealMathOp op, awl::RealMathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto f32 = rewriter.getF32Type();
    Value v = adaptor.getValue();
    std::string k = op.getKind().str();
    Value r;
    if (k == "abs") {
      // awlsim insn_abs: pure bit op (dword & 0x7FFFFFFF), no STW change
      r = rewriter.create<math::AbsFOp>(loc, f32, v);
    } else if (k == "negr") {
      // awlsim insn_negr: pure sign-bit XOR, no STW change
      r = mlir::sclang::withFPFastMath(rewriter.create<arith::NegFOp>(loc, f32, v));
    } else {
      // sin/cos/tan/asin/acos/atan/sqrt/exp/ln/sqr: route through the
      // runtime, which replicates the awlsim instruction exactly (NaN/inf
      // special cases, epsilon snapping, double math rounded to f32) and
      // sets the STW (setForFloatingPoint).
      uint8_t opc;
      if (k == "sin") opc = 0;
      else if (k == "cos") opc = 1;
      else if (k == "tan") opc = 2;
      else if (k == "asin") opc = 3;
      else if (k == "acos") opc = 4;
      else if (k == "atan") opc = 5;
      else if (k == "sqrt") opc = 6;
      else if (k == "exp") opc = 7;
      else if (k == "ln") opc = 8;
      else if (k == "sqr") opc = 9;
      else return failure();
      auto mod = moduleOf(op);
      auto f64 = rewriter.getF64Type();
      auto i8 = rewriter.getI8Type();
      auto fn = getOrInsertRtFunc(rewriter, mod, "scl_rt_realmath_f",
                                  rewriter.getFunctionType({i8, f64}, {f64}));
      Value opcC = constInt(rewriter, loc, i8, opc);
      Value d = mlir::sclang::withFPFastMath(rewriter.create<arith::ExtFOp>(loc, f64, v));
      Value rd = rewriter.create<func::CallOp>(loc, fn, TypeRange{f64},
                                               ValueRange{opcC, d}).getResult(0);
      r = mlir::sclang::withFPFastMath(rewriter.create<arith::TruncFOp>(loc, f32, rd));
    }
    rewriter.replaceOp(op, r);
    return success();
  }
};


struct AwlNegOpLowering : public OpConversionPattern<awl::NegOp> {
  using OpConversionPattern<awl::NegOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::NegOp op, awl::NegOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType());
    Value v = convertScalarTo(rewriter, op.getLoc(), adaptor.getRhs(), resTy);
    if (!v)
      return failure();
    if (llvm::isa<FloatType>(resTy)) {
      mlir::sclang::withFPFastMath(rewriter.replaceOpWithNewOp<arith::NegFOp>(op, resTy, v));
    } else {
      auto zero = constInt(rewriter, op.getLoc(), resTy, 0);
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, resTy, zero, v);
    }
    return success();
  }
};


struct AwlCastOpLowering : public OpConversionPattern<awl::CastOp> {
  using OpConversionPattern<awl::CastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::CastOp op, awl::CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    if (op.getBitcast().has_value()) {
      // reinterpret the bits (DTR, TRUNC input): int <-> float, same width
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, resTy,
                                                    adaptor.getValue());
      return success();
    }
    Value v = convertScalarTo(rewriter, op.getLoc(), adaptor.getValue(), resTy,
                              op.getZext().has_value());
    if (!v)
      return failure();
    rewriter.replaceOp(op, v);
    return success();
  }
};


template <typename AwlOp, CmpIPredicate predI, CmpFPredicate predF>
struct AwlCmpLowering : public OpConversionPattern<AwlOp> {
  using OpConversionPattern<AwlOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AwlOp op, typename AwlOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value l = adaptor.getLhs();
    Value r = adaptor.getRhs();
    if (l.getType() != r.getType()) {
      // normalize to the wider type
      Type ty = llvm::isa<FloatType>(l.getType()) || llvm::isa<FloatType>(r.getType())
                    ? (Type)rewriter.getF32Type()
                    : (l.getType().getIntOrFloatBitWidth() >=
                               r.getType().getIntOrFloatBitWidth()
                           ? l.getType()
                           : r.getType());
      l = convertScalarTo(rewriter, op.getLoc(), l, ty);
      r = convertScalarTo(rewriter, op.getLoc(), r, ty);
      if (!l || !r)
        return failure();
    }
    if (llvm::isa<FloatType>(l.getType()))
      mlir::sclang::withFPFastMath(rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, predF, l, r));
    else
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, predI, l, r);
    return success();
  }
};

using AwlCmpEqLowering = AwlCmpLowering<awl::CmpEqOp, CmpIPredicate::eq, CmpFPredicate::OEQ>;

using AwlCmpNeLowering = AwlCmpLowering<awl::CmpNeOp, CmpIPredicate::ne, CmpFPredicate::UNE>;

using AwlCmpLtLowering = AwlCmpLowering<awl::CmpLtOp, CmpIPredicate::slt, CmpFPredicate::OLT>;

using AwlCmpLeLowering = AwlCmpLowering<awl::CmpLeOp, CmpIPredicate::sle, CmpFPredicate::OLE>;

using AwlCmpGtLowering = AwlCmpLowering<awl::CmpGtOp, CmpIPredicate::sgt, CmpFPredicate::OGT>;

using AwlCmpGeLowering = AwlCmpLowering<awl::CmpGeOp, CmpIPredicate::sge, CmpFPredicate::OGE>;


struct AwlCallOpLowering : public OpConversionPattern<awl::CallOp> {
  using OpConversionPattern<awl::CallOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::CallOp op, awl::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> args;
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    // awlsim insn_call/insn_cc/insn_uc: after the call OS (bit 4) is cleared
    auto clearOs = [&]() {
      auto i8 = rewriter.getI8Type();
      auto fn = getOrInsertRtFunc(rewriter, mod, "scl_rt_stw_set_bit",
                                  rewriter.getFunctionType({i8, i8}, {}));
      auto bitC = constInt(rewriter, op.getLoc(), i8, 4);
      auto zero = constInt(rewriter, op.getLoc(), i8, 0);
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{bitC, zero});
    };
    clearOs(); // OS (bit 4) is cleared when a block is called
    // CALL FB n, DB m: pass the instance DB number as the FB's i32 argument
    mlir::Value savedDi, savedDb;
    if (auto dbAttr = op.getDb()) {
      auto i32 = rewriter.getI32Type();
      // awlsim saves prevDbRegister/prevDiRegister in the call stack element
      // and restores them when the callee returns (run_BE)
      auto curDiFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_current_di",
                                       rewriter.getFunctionType({}, {i32}));
      savedDi = rewriter.create<func::CallOp>(op.getLoc(), curDiFn,
                                              TypeRange{i32}, ValueRange{})
                    .getResult(0);
      auto curDbFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_current_db",
                                       rewriter.getFunctionType({}, {i32}));
      savedDb = rewriter.create<func::CallOp>(op.getLoc(), curDbFn,
                                              TypeRange{i32}, ValueRange{})
                    .getResult(0);
      if (dbAttr.value() == -1) {
        // multi-instance call (CALL #var): the instance DB is the current DI
        // register; awlsim also sets the DB register to it (cpu.py
        // __call_MULTI_FB: dbRegister = diRegister)
        auto diFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_current_di",
                                      rewriter.getFunctionType({}, {i32}));
        auto diVal = rewriter.create<func::CallOp>(op.getLoc(), diFn,
                                                   TypeRange{i32}, ValueRange{})
                         .getResult(0);
        args.push_back(diVal);
        auto dbFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_set_current_db",
                                      rewriter.getFunctionType({i32}, {}));
        rewriter.create<func::CallOp>(op.getLoc(), dbFn, TypeRange{},
                                      ValueRange{diVal});
      } else {
        args.push_back(constInt(rewriter, op.getLoc(), i32, dbAttr.value()));
        // the instance DB becomes the DI register for the callee (AUF DI)
        auto diFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_set_current_di",
                                      rewriter.getFunctionType({i32}, {}));
        rewriter.create<func::CallOp>(op.getLoc(), diFn, TypeRange{},
                                      ValueRange{args.back()});
      }
    }
    // SFB0-5 (multi-instance or with instance DB): run the system FB
    // semantics in the runtime (after the AR2 base is set below), using the
    // DI register + AR2 base
    bool sfbRt = false;
    int32_t sfbN = 0;
    if (auto sfcAttr = op.getSfc()) {
      if (op.getCallee().starts_with("SFB")) {
        sfbRt = true;
        sfbN = sfcAttr.value();
      }
    }
    // SFC n: run in the runtime (params staged in the marker area by the
    // frontend at kLocalBase)
    bool sfcRt = false;
    int32_t sfcN = 0;
    if (auto sfcAttr = op.getSfc()) {
      if (op.getCallee().starts_with("SFC")) {
        sfcRt = true;
        sfcN = sfcAttr.value();
      }
    }
    // IEC library FC called by name (CALL "LEN"/"GE_DT"/...): run in the
    // runtime (params staged in the marker area by the frontend)
    bool iecRt = false;
    int32_t iecN = 0;
    if (auto iecAttr = op.getIec()) {
      iecRt = true;
      iecN = iecAttr.value();
    }
    // Cross-file / system block callee (not defined in this module): create
    // an empty stub so single files of a multi-file project still lower and
    // JIT-run (the callee body is not modelled — documented approximation).
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!sfbRt && !sfcRt && !iecRt && !module.lookupSymbol(op.getCallee())) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      SmallVector<Type, 4> argTys;
      for (Value a : args)
        argTys.push_back(a.getType());
      auto fn = rewriter.create<func::FuncOp>(
          op.getLoc(), op.getCallee(), rewriter.getFunctionType(argTys, {}));
      fn.setVisibility(SymbolTable::Visibility::Private);
      auto *entry = fn.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      rewriter.create<func::ReturnOp>(op.getLoc());
    }
    // explicit instance DB (CALL FB n, DB m): AR2 := P#DBX 0.0 for the
    // duration of the call, restored afterwards (S7 saves the ARs)
    mlir::Value savedAr2;
    if (op.getInstdb().has_value()) {
      auto i32 = rewriter.getI32Type();
      auto i64 = rewriter.getI64Type();
      auto i8 = rewriter.getI8Type();
      auto getFn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_get",
                                     rewriter.getFunctionType({i8}, {i64}));
      auto setFn = getOrInsertRtFunc(
          rewriter, module, "scl_rt_ar_set",
          rewriter.getFunctionType({i8, i64}, {}));
      auto packFn = getOrInsertRtFunc(rewriter, module, "scl_rt_pack_s7",
                                      rewriter.getFunctionType(
                                          {i32, i32, i32, i32}, {i64}));
      auto arC = constInt(rewriter, op.getLoc(), i8, 2);
      savedAr2 = rewriter.create<func::CallOp>(op.getLoc(), getFn,
                                               TypeRange{i64},
                                               ValueRange{arC})
                     .getResult(0);
      int32_t ar2off = op.getAr2off() ? (int32_t)op.getAr2off().value() : 0;
      Value ptr;
      if (ar2off > 0) {
        // multi-instance base: AR2 += ar2off (relative to the caller's AR2,
        // so nested multi-instances accumulate correctly)
        auto addFn = getOrInsertRtFunc(
            rewriter, module, "scl_rt_ar_add",
            rewriter.getFunctionType({i8, i32}, {i64}));
        auto arC2 = constInt(rewriter, op.getLoc(), i8, 2);
        auto offC = constInt(rewriter, op.getLoc(), i32, ar2off * 8);
        ptr = rewriter
                  .create<func::CallOp>(op.getLoc(), addFn, TypeRange{i64},
                                        ValueRange{arC2, offC})
                  .getResult(0);
      } else {
        // awlsim sets AR2 = AREA_DB_S | base offset (P#DBX 0.0 style, db
        // field 0 = current DI register at use time), not the DI area code
        auto areaC = constInt(rewriter, op.getLoc(), i32, 0x84);
        auto zero32 = constInt(rewriter, op.getLoc(), i32, 0);
        ptr = rewriter
                  .create<func::CallOp>(op.getLoc(), packFn, TypeRange{i64},
                                        ValueRange{areaC, zero32, zero32,
                                                   zero32})
                  .getResult(0);
      }
      rewriter.create<func::CallOp>(op.getLoc(), setFn, TypeRange{},
                                    ValueRange{arC, ptr});
    }
    if (sfbRt) {
      auto i32 = rewriter.getI32Type();
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_sfb",
                                  rewriter.getFunctionType({i32}, {}));
      auto nC = constInt(rewriter, op.getLoc(), i32, sfbN);
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{nC});
      if (savedAr2) {
        auto i8 = rewriter.getI8Type();
        auto i64 = rewriter.getI64Type();
        auto setFn = getOrInsertRtFunc(
            rewriter, module, "scl_rt_ar_set",
            rewriter.getFunctionType({i8, i64}, {}));
        auto arC = constInt(rewriter, op.getLoc(), i8, 2);
        rewriter.create<func::CallOp>(op.getLoc(), setFn, TypeRange{},
                                      ValueRange{arC, savedAr2});
      }
      rewriter.eraseOp(op);
      return success();
    }
    if (sfcRt) {
      auto i32 = rewriter.getI32Type();
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_sfc",
                                  rewriter.getFunctionType({i32}, {}));
      auto nC = constInt(rewriter, op.getLoc(), i32, sfcN);
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{nC});
      rewriter.eraseOp(op);
      return success();
    }
    if (iecRt) {
      auto i32 = rewriter.getI32Type();
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_iec",
                                  rewriter.getFunctionType({i32}, {}));
      auto nC = constInt(rewriter, op.getLoc(), i32, iecN);
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{nC});
      rewriter.eraseOp(op);
      return success();
    }
    auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(),
                                              TypeRange{}, args);
    // awlsim insn_CALL.run clears these status bits after the call returns:
    // OS=0, OR=0, STA=1, NER=0 (OV is preserved). Bits: NER=0 STA=2 OR=3 OS=4.
    {
      auto i8 = rewriter.getI8Type();
      auto fn = getOrInsertRtFunc(rewriter, mod, "scl_rt_stw_set_bit",
                                  rewriter.getFunctionType({i8, i8}, {}));
      for (auto bv : {std::pair<int, int>{4, 0}, {3, 0}, {2, 1}, {0, 0}}) {
        auto bitC = constInt(rewriter, op.getLoc(), i8, bv.first);
        auto valC = constInt(rewriter, op.getLoc(), i8, bv.second);
        rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                      ValueRange{bitC, valC});
      }
    }
    if (savedAr2) {
      auto i8 = rewriter.getI8Type();
      auto i64 = rewriter.getI64Type();
      auto setFn = getOrInsertRtFunc(
          rewriter, module, "scl_rt_ar_set",
          rewriter.getFunctionType({i8, i64}, {}));
      auto arC = constInt(rewriter, op.getLoc(), i8, 2);
      rewriter.create<func::CallOp>(op.getLoc(), setFn, TypeRange{},
                                    ValueRange{arC, savedAr2});
    }
    if (savedDi) { // restore the DB/DI registers saved before the call
      auto i32 = rewriter.getI32Type();
      auto diFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_set_current_di",
                                    rewriter.getFunctionType({i32}, {}));
      rewriter.create<func::CallOp>(op.getLoc(), diFn, TypeRange{},
                                    ValueRange{savedDi});
      auto dbFn = getOrInsertRtFunc(rewriter, mod, "scl_rt_set_current_db",
                                    rewriter.getFunctionType({i32}, {}));
      rewriter.create<func::CallOp>(op.getLoc(), dbFn, TypeRange{},
                                    ValueRange{savedDb});
    }
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};


// __ASSERT oracle -> scl_rt_assert_cmp / scl_rt_assert_cmp_r
struct AwlAssertOpLowering : public OpConversionPattern<awl::AssertOp> {
  using OpConversionPattern<awl::AssertOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AssertOp op, awl::AssertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i8 = rewriter.getI8Type();
    auto i64 = rewriter.getI64Type();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_assert_cmp",
                                rewriter.getFunctionType({i32, i8, i64, i64}, {}));
    auto idC = constInt(rewriter, op.getLoc(), i32, op.getId());
    auto opC = constInt(rewriter, op.getLoc(), i8, op.getOp());
    rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                  ValueRange{idC, opC, adaptor.getActual(),
                                             adaptor.getExpected()});
    rewriter.eraseOp(op);
    return success();
  }
};


struct AwlAssertROpLowering : public OpConversionPattern<awl::AssertROp> {
  using OpConversionPattern<awl::AssertROp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AssertROp op, awl::AssertROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i8 = rewriter.getI8Type();
    auto f64 = rewriter.getF64Type();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_assert_cmp_r",
                                rewriter.getFunctionType({i32, i8, f64, f64}, {}));
    auto idC = constInt(rewriter, op.getLoc(), i32, op.getId());
    auto opC = constInt(rewriter, op.getLoc(), i8, op.getOp());
    rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                  ValueRange{idC, opC, adaptor.getActual(),
                                             adaptor.getExpected()});
    rewriter.eraseOp(op);
    return success();
  }
};


struct AwlNopOpLowering : public OpConversionPattern<awl::NopOp> {
  using OpConversionPattern<awl::NopOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::NopOp op, awl::NopOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};


// MARK: status word lowerings (awlsim statusword; bits NER=0 VKE/RLO=1
// STA=2 OR=3 OS=4 OV=5 A0/CC0=6 A1/CC1=7 BIE/BR=8)

// L STW -> @scl_rt_stw_get() -> i32
struct AwlStwGetLowering : public OpConversionPattern<awl::StwGetOp> {
  using OpConversionPattern<awl::StwGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwGetOp op, awl::StwGetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_get",
                                rewriter.getFunctionType({}, {rewriter.getI32Type()}));
    auto word = rewriter.create<func::CallOp>(loc, fn, TypeRange{rewriter.getI32Type()},
                                              ValueRange{})
                    .getResult(0);
    rewriter.replaceOp(op, word);
    return success();
  }
};


// ACCU get -> @scl_rt_accu_get(index: i8) -> i32
struct AwlAccuGetLowering : public OpConversionPattern<awl::AccuGetOp> {
  using OpConversionPattern<awl::AccuGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AccuGetOp op, awl::AccuGetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value idx = constInt(rewriter, loc, i8, op.getIndex());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_accu_get",
                                rewriter.getFunctionType({i8}, {rewriter.getI32Type()}));
    auto v = rewriter.create<func::CallOp>(loc, fn, TypeRange{rewriter.getI32Type()},
                                           ValueRange{idx}).getResult(0);
    rewriter.replaceOp(op, v);
    return success();
  }
};


// ACCU set -> @scl_rt_accu_set(index: i8, value: i32)
struct AwlAccuSetLowering : public OpConversionPattern<awl::AccuSetOp> {
  using OpConversionPattern<awl::AccuSetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AccuSetOp op, awl::AccuSetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    Value idx = constInt(rewriter, loc, i8, op.getIndex());
    Value v = convertScalarTo(rewriter, loc, adaptor.getValue(), i32);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_accu_set",
                                rewriter.getFunctionType({i8, i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{idx, v});
    rewriter.eraseOp(op);
    return success();
  }
};


// T STW -> @scl_rt_stw_set(word: i32)
struct AwlStwSetLowering : public OpConversionPattern<awl::StwSetOp> {
  using OpConversionPattern<awl::StwSetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwSetOp op, awl::StwSetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value word = convertScalarTo(rewriter, loc, adaptor.getWord(), i32);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_set",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{word});
    rewriter.eraseOp(op);
    return success();
  }
};


// __STW bit -> (stw >> bit) & 1, computed from @scl_rt_stw_get()
struct AwlStwBitLowering : public OpConversionPattern<awl::StwBitOp> {
  using OpConversionPattern<awl::StwBitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwBitOp op, awl::StwBitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_get",
                                rewriter.getFunctionType({}, {i32}));
    Value word = rewriter.create<func::CallOp>(loc, fn, TypeRange{i32}, ValueRange{})
                     .getResult(0);
    Value bitC = constInt(rewriter, loc, i32, op.getBit());
    Value shifted = rewriter.create<arith::ShRUIOp>(loc, word, bitC);
    Value one = constInt(rewriter, loc, i32, 1);
    Value bit = rewriter.create<arith::AndIOp>(loc, shifted, one);
    // arith.cmpi returns i1 for any integer operands
    Value zero = constInt(rewriter, loc, i32, 0);
    Value res = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                bit, zero);
    rewriter.replaceOp(op, res);
    return success();
  }
};


// SAVE -> @scl_rt_stw_set_bit(bit: i8, value: i8)
struct AwlStwSetBitLowering : public OpConversionPattern<awl::StwSetBitOp> {
  using OpConversionPattern<awl::StwSetBitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwSetBitOp op, awl::StwSetBitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value bitC = constInt(rewriter, loc, i8, op.getBit());
    Value val = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getValue());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_set_bit",
                                rewriter.getFunctionType({i8, i8}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{bitC, val});
    rewriter.eraseOp(op);
    return success();
  }
};


// U >0 / ==0 / <>0 / <0 / >=0 / <=0 / UO -> A0/A1 bits of the STW.
// awlsim cpu.__fetchSTW_*: 0=Z(!A0&!A1), 1=NZ(A0|A1), 2=POS(!A0&A1),
// 3=NEG(A0&!A1), 4=GE(!A0), 5=LE(!A1), 6=UO(A0&A1).
struct AwlStwCondLowering : public OpConversionPattern<awl::StwCondOp> {
  using OpConversionPattern<awl::StwCondOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwCondOp op, awl::StwCondOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i1 = rewriter.getI1Type();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_get",
                                rewriter.getFunctionType({}, {i32}));
    Value word = rewriter.create<func::CallOp>(loc, fn, TypeRange{i32}, ValueRange{})
                     .getResult(0);
    auto bitOf = [&](int bit) -> Value {
      Value c = constInt(rewriter, loc, i32, bit);
      Value sh = rewriter.create<arith::ShRUIOp>(loc, word, c);
      Value one = constInt(rewriter, loc, i32, 1);
      return rewriter.create<arith::AndIOp>(loc, sh, one);
    };
    Value a0 = bitOf(6); // CC0: result < 0
    Value a1 = bitOf(7); // CC1: result > 0
    Value zero = constInt(rewriter, loc, i32, 0);
    Value isZ = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                rewriter.create<arith::OrIOp>(loc, a0, a1),
                                                zero);
    Value isNZ = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 rewriter.create<arith::OrIOp>(loc, a0, a1),
                                                 zero);
    Value a0b = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, a0, zero);
    Value a1b = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, a1, zero);
    Value trueC = rewriter.create<arith::ConstantOp>(loc, i1, rewriter.getBoolAttr(true));
    Value notA0 = rewriter.create<arith::XOrIOp>(loc, a0b, trueC);
    Value notA1 = rewriter.create<arith::XOrIOp>(loc, a1b, trueC);
    Value pos = rewriter.create<arith::AndIOp>(loc, a1b, notA0); // > 0
    Value neg = rewriter.create<arith::AndIOp>(loc, a0b, notA1); // < 0
    // awlsim: >=0 is !A0 and <=0 is !A1 (so unordered A0=A1=1 gives 0)
    Value ge = notA0;
    Value le = notA1;
    Value uo = rewriter.create<arith::AndIOp>(loc, a0b, a1b); // unordered
    Value res;
    switch (op.getCond()) {
    case 0: res = isZ; break;
    case 1: res = isNZ; break;
    case 2: res = pos; break;
    case 3: res = neg; break;
    case 4: res = ge; break;
    case 5: res = le; break;
    case 6: res = uo; break;
    default: return rewriter.notifyMatchFailure(op, "bad stw cond");
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};


// __STWRST -> @scl_rt_stw_reset()
struct AwlStwResetLowering : public OpConversionPattern<awl::StwResetOp> {
  using OpConversionPattern<awl::StwResetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwResetOp op, awl::StwResetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_reset",
                                rewriter.getFunctionType({}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};


// comparison -> record into A0/A1: @scl_rt_stw_cmp_i/f(pred: i8, lhs, rhs)
struct AwlStwCmpLowering : public OpConversionPattern<awl::StwCmpOp> {
  using OpConversionPattern<awl::StwCmpOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwCmpOp op, awl::StwCmpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value predC = constInt(rewriter, loc, i8, op.getPred());
    if (llvm::isa<FloatType>(adaptor.getLhs().getType())) {
      auto f64 = rewriter.getF64Type();
      Value l = convertScalarTo(rewriter, loc, adaptor.getLhs(), f64);
      Value r = convertScalarTo(rewriter, loc, adaptor.getRhs(), f64);
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_cmp_f",
                                  rewriter.getFunctionType({i8, f64, f64}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{predC, l, r});
    } else {
      auto i64 = rewriter.getI64Type();
      Value l = convertScalarTo(rewriter, loc, adaptor.getLhs(), i64);
      Value r = convertScalarTo(rewriter, loc, adaptor.getRhs(), i64);
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_cmp_i",
                                  rewriter.getFunctionType({i8, i64, i64}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{predC, l, r});
    }
    rewriter.eraseOp(op);
    return success();
  }
};


// +/-/mul I/D/R -> @scl_rt_stw_sign_i/f(value, lhs, rhs, op, range)
struct AwlStwSignLowering : public OpConversionPattern<awl::StwSignOp> {
  using OpConversionPattern<awl::StwSignOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwSignOp op, awl::StwSignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (llvm::isa<FloatType>(adaptor.getValue().getType())) {
      auto f64 = rewriter.getF64Type();
      Value v = convertScalarTo(rewriter, loc, adaptor.getValue(), f64);
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_sign_f",
                                  rewriter.getFunctionType({f64}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{v});
    } else {
      auto i64 = rewriter.getI64Type();
      auto i32 = rewriter.getI32Type();
      auto i8 = rewriter.getI8Type();
      Value v = convertScalarTo(rewriter, loc, adaptor.getValue(), i64);
      Value l = convertScalarTo(rewriter, loc, adaptor.getLhs(), i64);
      Value r = convertScalarTo(rewriter, loc, adaptor.getRhs(), i64);
      Value opC = constInt(rewriter, loc, i8, op.getOp());
      Value rangeC = constInt(rewriter, loc, i32, op.getRange());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_sign_i",
                                  rewriter.getFunctionType({i64, i64, i64, i8, i32}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{v, l, r, opC, rangeC});
    }
    rewriter.eraseOp(op);
    return success();
  }
};


// / I/D -> @scl_rt_stw_div_i/f(divisor, result)
struct AwlStwDivLowering : public OpConversionPattern<awl::StwDivOp> {
  using OpConversionPattern<awl::StwDivOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StwDivOp op, awl::StwDivOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (llvm::isa<FloatType>(adaptor.getResult().getType())) {
      auto f64 = rewriter.getF64Type();
      Value d = convertScalarTo(rewriter, loc, adaptor.getDivisor(), f64);
      Value r = convertScalarTo(rewriter, loc, adaptor.getResult(), f64);
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_div_f",
                                  rewriter.getFunctionType({f64, f64}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{d, r});
    } else {
      auto i64 = rewriter.getI64Type();
      Value d = convertScalarTo(rewriter, loc, adaptor.getDivisor(), i64);
      Value r = convertScalarTo(rewriter, loc, adaptor.getResult(), i64);
      Value rng = constInt(rewriter, loc, rewriter.getI32Type(), op.getRange());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_stw_div_i",
                                  rewriter.getFunctionType({i64, i64, rewriter.getI32Type()}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{d, r, rng});
    }
    rewriter.eraseOp(op);
    return success();
  }
};


// TRUNC/RND/RND+/RND- -> @scl_rt_f2i(value: f64, orig: i64, mode: i8) -> i64 -> i32
struct AwlF2ILowering : public OpConversionPattern<awl::F2IOp> {
  using OpConversionPattern<awl::F2IOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::F2IOp op, awl::F2IOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto f64 = rewriter.getF64Type();
    auto i64 = rewriter.getI64Type();
    auto i8 = rewriter.getI8Type();
    Value v = convertScalarTo(rewriter, loc, adaptor.getValue(), f64);
    Value orig = convertScalarTo(rewriter, loc, adaptor.getOrig(), i64);
    Value modeC = constInt(rewriter, loc, i8, op.getMode());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_f2i",
                                rewriter.getFunctionType({f64, i64, i8}, {i64}));
    Value r = rewriter.create<func::CallOp>(loc, fn, TypeRange{i64},
                                            ValueRange{v, orig, modeC})
                  .getResult(0);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, rewriter.getI32Type(), r);
    return success();
  }
};


// SP/SE/SS/SV/SI/SA T n -> @scl_rt_timer_start(timer: i32, mode: i8, dur: i64, in: i8)
struct AwlTimerStartLowering : public OpConversionPattern<awl::TimerStartOp> {
  using OpConversionPattern<awl::TimerStartOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::TimerStartOp op, awl::TimerStartOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getTimer().getType().isInteger(64)) {
      // dynamic T pointer (TIMER-typed variable holding the number)
      auto loc = op.getLoc();
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto i8 = rewriter.getI8Type();
      auto i64 = rewriter.getI64Type();
      auto modeC = constInt(rewriter, loc, i8, op.getMode());
      Value durC = convertScalarTo(rewriter, loc, adaptor.getDur(), i64);
      Value inC = constInt(rewriter, loc, i8, 1);
      if (adaptor.getIn())
        inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_start_ptr",
                                  rewriter.getFunctionType({i64, i8, i64, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getTimer(), modeC, durC, inC});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = llvm::dyn_cast<scl::DirectType>(op.getTimer().getType());
    if (!dt)
      return rewriter.notifyMatchFailure(op, "timer_start needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = moduleOf(op);
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value timerC = constInt(rewriter, loc, i32, dt.getByte());
    Value modeC = constInt(rewriter, loc, i8, op.getMode());
    Value dur = convertScalarTo(rewriter, loc, adaptor.getDur(), i64);
    Value inC = constInt(rewriter, loc, i8, 1);
    if (adaptor.getIn())
      inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_start",
                                rewriter.getFunctionType({i32, i8, i64, i8}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                  ValueRange{timerC, modeC, dur, inC});
    rewriter.eraseOp(op);
    return success();
  }
};


// R T n -> @scl_rt_timer_reset(timer: i32)
struct AwlTimerResetLowering : public OpConversionPattern<awl::TimerResetOp> {
  using OpConversionPattern<awl::TimerResetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::TimerResetOp op, awl::TimerResetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getTimer().getType().isInteger(64)) {
      auto loc = op.getLoc();
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_reset_ptr",
                                  rewriter.getFunctionType({rewriter.getI64Type()}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getTimer()});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = llvm::dyn_cast<scl::DirectType>(op.getTimer().getType());
    if (!dt)
      return rewriter.notifyMatchFailure(op, "timer_reset needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value timerC = constInt(rewriter, loc, i32, dt.getByte());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_reset",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{timerC});
    rewriter.eraseOp(op);
    return success();
  }
};


// FR T n / FR C n -> @scl_rt_timer_fr(t: i32, in: i8) / counter_op FR
struct AwlFrOpLowering : public OpConversionPattern<awl::FrOp> {
  using OpConversionPattern<awl::FrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::FrOp op, awl::FrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperand().getType().isInteger(64)) {
      auto loc = op.getLoc();
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto i8 = rewriter.getI8Type();
      Value inC = constInt(rewriter, loc, i8, 1);
      if (adaptor.getIn())
        inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_fr_ptr",
                                  rewriter.getFunctionType({rewriter.getI64Type(), i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getOperand(), inC});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = llvm::dyn_cast<scl::DirectType>(op.getOperand().getType());
    if (!dt)
      return rewriter.notifyMatchFailure(op, "fr needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value idxC = constInt(rewriter, loc, i32, dt.getByte());
    Value inC = constInt(rewriter, loc, i8, 1);
    if (adaptor.getIn())
      inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
    if (dt.getArea() == scl::DirectArea::Timer) {
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_fr",
                                  rewriter.getFunctionType({i32, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{idxC, inC});
    } else if (dt.getArea() == scl::DirectArea::Counter) {
      Value opC = constInt(rewriter, loc, i8, 4 /* SCL_RT_COUNTER_FR */);
      Value zero = constInt(rewriter, loc, i64, 0);
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_counter_op",
                                  rewriter.getFunctionType({i32, i8, i64, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{idxC, opC, zero, inC});
    }
    // FR DB: nothing to do
    rewriter.eraseOp(op);
    return success();
  }
};


// __SLEEP n -> @scl_rt_sleep(ms: i32)
struct AwlSleepOpLowering : public OpConversionPattern<awl::SleepOp> {
  using OpConversionPattern<awl::SleepOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SleepOp op, awl::SleepOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value msC = constInt(rewriter, loc, i32, op.getMs());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_sleep",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{msC});
    rewriter.eraseOp(op);
    return success();
  }
};


// ZV/ZR/S/R C n -> @scl_rt_counter_op(counter: i32, op: i8, value: i64, in: i8)
// pack_area(area, byte, bit) -> @scl_rt_pack_area(area: i8, byte: i32, bit: i8) -> i64
struct AwlPackAreaOpLowering : public OpConversionPattern<awl::PackAreaOp> {
  using OpConversionPattern<awl::PackAreaOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::PackAreaOp op, awl::PackAreaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto areaC = constInt(rewriter, loc, i8, op.getArea());
    auto byteC = convertScalarTo(rewriter, loc, adaptor.getByte(), i32);
    auto bitC = constInt(rewriter, loc, i8, op.getBit());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_pack_area",
                                rewriter.getFunctionType({i8, i32, i8}, {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()},
        ValueRange{areaC, byteC, bitC});
    return success();
  }
};


struct AwlCounterOpLowering : public OpConversionPattern<awl::CounterOp> {
  using OpConversionPattern<awl::CounterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::CounterOp op, awl::CounterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = moduleOf(op);
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    auto opC0 = constInt(rewriter, loc, i8, op.getOp());
    Value val0 = convertScalarTo(rewriter, loc, adaptor.getValue(), i64);
    Value inC0 = constInt(rewriter, loc, i8, 0);
    if (adaptor.getIn())
      inC0 = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
    if (adaptor.getCounter().getType().isInteger(64)) {
      // dynamic T/C pointer (TIMER/COUNTER-typed variable holding the number)
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_counter_op_ptr",
                                  rewriter.getFunctionType({i64, i8, i64, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getCounter(), opC0,
                                               val0, inC0});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = llvm::dyn_cast<scl::DirectType>(op.getCounter().getType());
    if (!dt)
      return rewriter.notifyMatchFailure(op, "counter needs a direct operand");
    Value cntC = constInt(rewriter, loc, i32, dt.getByte());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_counter_op",
                                rewriter.getFunctionType({i32, i8, i64, i8}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                  ValueRange{cntC, opC0, val0, inC0});
    rewriter.eraseOp(op);
    return success();
  }
};


// MARK: ABI v5 lowerings (DB register / address registers / indirect access)

// set the DB size -> @scl_rt_sim_set_db_size(db: i32, bytes: i64)
struct AwlSetDbSizeOpLowering : public OpConversionPattern<awl::SetDbSizeOp> {
  using OpConversionPattern<awl::SetDbSizeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDbSizeOp op, awl::SetDbSizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value dbC = constInt(rewriter, loc, i32, op.getDb());
    Value szC = constInt(rewriter, loc, i64, op.getBytes());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_sim_set_db_size",
                                rewriter.getFunctionType({i32, i64}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbC, szC});
    rewriter.eraseOp(op);
    return success();
  }
};


// register a DB's highest referenced byte -> @scl_rt_sim_db_ref(db: i32, bytes: i64)
struct AwlSetDbRefOpLowering : public OpConversionPattern<awl::SetDbRefOp> {
  using OpConversionPattern<awl::SetDbRefOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDbRefOp op, awl::SetDbRefOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value dbC = constInt(rewriter, loc, i32, op.getDb());
    Value szC = constInt(rewriter, loc, i64, op.getBytes());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_sim_db_ref",
                                rewriter.getFunctionType({i32, i64}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbC, szC});
    rewriter.eraseOp(op);
    return success();
  }
};


// __FEATURE t [, v] -> @scl_rt_feature(target: i32, value: i32) -> i32
struct AwlFeatureOpLowering : public OpConversionPattern<awl::FeatureOp> {
  using OpConversionPattern<awl::FeatureOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::FeatureOp op, awl::FeatureOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value tC = constInt(rewriter, loc, i32, op.getTarget());
    Value vC = constInt(rewriter, loc, i32, op.getValue());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_feature",
                                rewriter.getFunctionType({i32, i32}, {i32}));
    auto call = rewriter.create<func::CallOp>(loc, fn, TypeRange{i32},
                                              ValueRange{tC, vC});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};


// DBLG/DBNO/DILG/DINO -> @scl_rt_db_info(kind: i32, db: i32) -> i64
struct AwlDbInfoOpLowering : public OpConversionPattern<awl::DbInfoOp> {
  using OpConversionPattern<awl::DbInfoOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::DbInfoOp op, awl::DbInfoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value kindC = constInt(rewriter, loc, i32, op.getKind());
    Value dbC = constInt(rewriter, loc, i32, op.getDb());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_db_info",
                                rewriter.getFunctionType({i32, i32}, {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()}, ValueRange{kindC, dbC});
    return success();
  }
};


// AUF/OPN DB n -> @scl_rt_set_current_db(db: i32)
struct AwlSetDbOpLowering : public OpConversionPattern<awl::SetDbOp> {
  using OpConversionPattern<awl::SetDbOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDbOp op, awl::SetDbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value dbC = constInt(rewriter, loc, i32, op.getDb());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_set_current_db",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbC});
    rewriter.eraseOp(op);
    return success();
  }
};


// AUF/OPN DI n -> @scl_rt_set_current_di(db: i32)
struct AwlSetDiOpLowering : public OpConversionPattern<awl::SetDiOp> {
  using OpConversionPattern<awl::SetDiOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDiOp op, awl::SetDiOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value dbC = constInt(rewriter, loc, i32, op.getDb());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_set_current_di",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbC});
    rewriter.eraseOp(op);
    return success();
  }
};


// pack_ptr -> @scl_rt_pack_s7(area: i32, db: i32, byte: i32, bit: i32) -> i64
struct AwlPackPtrOpLowering : public OpConversionPattern<awl::PackPtrOp> {
  using OpConversionPattern<awl::PackPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::PackPtrOp op, awl::PackPtrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value dbV = adaptor.getDb();
    if (!dbV.getType().isInteger(32))
      dbV = rewriter.create<arith::TruncIOp>(loc, i32, dbV);
    auto mk = [&](int32_t v) {
      return constInt(rewriter, loc, i32, v);
    };
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_pack_s7",
                                rewriter.getFunctionType({i32, i32, i32, i32},
                                                         {i64}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{i64},
        ValueRange{mk(op.getArea()), dbV, mk(op.getByte()), mk(op.getBit())});
    return success();
  }
};


// AUF <value> -> @scl_rt_set_current_db(db: i32)
struct AwlSetDbvOpLowering : public OpConversionPattern<awl::SetDbvOp> {
  using OpConversionPattern<awl::SetDbvOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDbvOp op, awl::SetDbvOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value dbV = adaptor.getDb();
    if (!dbV.getType().isInteger(32))
      dbV = rewriter.create<arith::TruncIOp>(loc, i32, dbV);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_set_current_db",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbV});
    rewriter.eraseOp(op);
    return success();
  }
};


// AUF DI <value> -> @scl_rt_set_current_di(db: i32)
struct AwlSetDivOpLowering : public OpConversionPattern<awl::SetDivOp> {
  using OpConversionPattern<awl::SetDivOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::SetDivOp op, awl::SetDivOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value dbV = adaptor.getDb();
    if (!dbV.getType().isInteger(32))
      dbV = rewriter.create<arith::TruncIOp>(loc, i32, dbV);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_set_current_di",
                                rewriter.getFunctionType({i32}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{dbV});
    rewriter.eraseOp(op);
    return success();
  }
};


// LAR1/LAR2 <packed ptr> -> @scl_rt_ar_set(ar: i8, ptr: i64)
struct AwlArSetOpLowering : public OpConversionPattern<awl::ArSetOp> {
  using OpConversionPattern<awl::ArSetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ArSetOp op, awl::ArSetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i64 = rewriter.getI64Type();
    Value arC = constInt(rewriter, loc, i8, op.getAr());
    // a !scl.direct operand arrives already converted to the packed i64
    Value ptr = adaptor.getPtr();
    if (ptr.getType() != i64)
      ptr = convertScalarTo(rewriter, loc, ptr, i64);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_set",
                                rewriter.getFunctionType({i8, i64}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{arC, ptr});
    rewriter.eraseOp(op);
    return success();
  }
};


// read AR -> @scl_rt_ar_get(ar: i8) -> i64
struct AwlArGetOpLowering : public OpConversionPattern<awl::ArGetOp> {
  using OpConversionPattern<awl::ArGetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ArGetOp op, awl::ArGetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value arC = constInt(rewriter, loc, i8, op.getAr());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_get",
                                rewriter.getFunctionType({i8}, {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()}, ValueRange{arC});
    return success();
  }
};


// TARn <memory>: AR as raw 32-bit S7 pointer -> @scl_rt_ar_raw(ar: i8) -> i32
struct AwlArRawOpLowering : public OpConversionPattern<awl::ArRawOp> {
  using OpConversionPattern<awl::ArRawOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ArRawOp op, awl::ArRawOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value arC = constInt(rewriter, loc, i8, op.getAr());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_raw",
                                rewriter.getFunctionType({i8}, {rewriter.getI32Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI32Type()}, ValueRange{arC});
    return success();
  }
};


// [AR1, P#x.y] -> @scl_rt_ar_add(ar: i8, offBits: i32) -> i64 packed ptr
struct AwlIndirectOpLowering : public OpConversionPattern<awl::IndirectOp> {
  using OpConversionPattern<awl::IndirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::IndirectOp op, awl::IndirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    Value arC = constInt(rewriter, loc, i8, op.getAr());
    Value offC = constInt(rewriter, loc, i32, op.getOffBits());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_add",
                                rewriter.getFunctionType({i8, i32}, {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()}, ValueRange{arC, offC});
    return success();
  }
};


// +ARn off -> @scl_rt_ar_add(ar: i8, offBits: i32) -> i64
struct AwlArAddOpLowering : public OpConversionPattern<awl::ArAddOp> {
  using OpConversionPattern<awl::ArAddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::ArAddOp op, awl::ArAddOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    Value arC = constInt(rewriter, loc, i8, op.getAr());
    Value offC = rewriter.create<arith::TruncIOp>(loc, i32, adaptor.getOff());
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_ar_add",
                                rewriter.getFunctionType({i8, i32}, {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()}, ValueRange{arC, offC});
    return success();
  }
};


// L [AR1, ...] -> @scl_rt_read_indirect(ptr: i64, widthBytes: i8) -> i64
struct AwlLoadIndirectOpLowering
    : public OpConversionPattern<awl::LoadIndirectOp> {
  using OpConversionPattern<awl::LoadIndirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::LoadIndirectOp op,
                  awl::LoadIndirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    Value widthC = constInt(rewriter, loc, i8, op.getWidthBits() / 8);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_read_indirect",
                                rewriter.getFunctionType({rewriter.getI64Type(), i8},
                                                         {rewriter.getI64Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI64Type()},
        ValueRange{adaptor.getPtr(), widthC});
    return success();
  }
};


// T [AR1, ...] -> @scl_rt_write_indirect(ptr: i64, value: i64, widthBytes: i8)
struct AwlStoreIndirectOpLowering
    : public OpConversionPattern<awl::StoreIndirectOp> {
  using OpConversionPattern<awl::StoreIndirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StoreIndirectOp op,
                  awl::StoreIndirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i64 = rewriter.getI64Type();
    Value v = convertScalarTo(rewriter, loc, adaptor.getValue(), i64);
    Value widthC = constInt(rewriter, loc, i8, op.getWidthBits() / 8);
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_write_indirect",
                                rewriter.getFunctionType({i64, i64, i8}, {}));
    rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                  ValueRange{adaptor.getPtr(), v, widthC});
    rewriter.eraseOp(op);
    return success();
  }
};


// U/O [AR1, ...] -> @scl_rt_read_bit_indirect(ptr: i64) -> i1
struct AwlLoadBitIndirectOpLowering
    : public OpConversionPattern<awl::LoadBitIndirectOp> {
  using OpConversionPattern<awl::LoadBitIndirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::LoadBitIndirectOp op,
                  awl::LoadBitIndirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_read_bit_indirect",
                                rewriter.getFunctionType({rewriter.getI64Type()},
                                                         {rewriter.getI1Type()}));
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn, TypeRange{rewriter.getI1Type()}, ValueRange{adaptor.getPtr()});
    return success();
  }
};


// S/R [AR1, ...] -> @scl_rt_write_bit_indirect(ptr: i64, value: i1)
struct AwlWriteBitIndirectOpLowering
    : public OpConversionPattern<awl::WriteBitIndirectOp> {
  using OpConversionPattern<awl::WriteBitIndirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::WriteBitIndirectOp op,
                  awl::WriteBitIndirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_write_bit_indirect",
                                rewriter.getFunctionType({rewriter.getI64Type(),
                                                          rewriter.getI1Type()},
                                                         {}));
    rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                  ValueRange{adaptor.getPtr(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};


/// Lower `awl.block` to `func.func`: split the flat block at labels/jumps,
/// wire cf branches, `awl.be` -> `func.return`. The remaining awl ops are
/// converted by their own patterns afterwards (conversion is pre-order).
struct AwlBlockOpLowering : public OpConversionPattern<awl::BlockOp> {
  using OpConversionPattern<awl::BlockOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::BlockOp op, awl::BlockOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // FB blocks take the instance DB number (from `CALL FB n, DB m`) as an
    // i32 argument; DB operands without a number inside the FB use it.
    bool isFb = op.getSymName().starts_with("FB") || op.getFb().has_value();
    auto funcType = isFb ? rewriter.getFunctionType({rewriter.getI32Type()}, {})
                         : rewriter.getFunctionType({}, {});
    auto func = rewriter.create<func::FuncOp>(loc, op.getSymName(), funcType);
    rewriter.inlineRegionBefore(op.getBody(), func.getBody(),
                                func.getBody().end());
    Block *flat = &func.getBody().front();
    // func::FuncOp::create does not add an entry block here; the inlined awl
    // block is the entry, so give it the instance-DB argument of the FB type
    if (isFb)
      flat->addArgument(rewriter.getI32Type(), loc);

    // Collect cut ops (labels, jumps, conditional calls, be) and split the
    // flat block before each of them, so every cut starts a new block.
    SmallVector<Operation *, 16> cuts;
    for (auto &o : flat->getOperations())
      if (isa<awl::LabelOp, awl::JumpOp, awl::JumpIfOp, awl::JumpIfNotOp,
              awl::BeOp, awl::BeIfOp>(&o) ||
          (isa<awl::CallOp>(&o) && cast<awl::CallOp>(&o).getRlo()))
        cuts.push_back(&o);
    DenseMap<Operation *, Block *> cutBlock;
    for (auto *cut : cuts)
      cutBlock[cut] = rewriter.splitBlock(cut->getBlock(), cut->getIterator());

    // Pass 1: register labels.
    llvm::StringMap<Block *> labels;
    for (auto *cut : cuts)
      if (auto l = dyn_cast<awl::LabelOp>(cut)) {
        // a duplicate label keeps the first occurrence (some corpus files
        // reuse a label name; jumps resolve to the first definition)
        labels.try_emplace(l.getName(), cutBlock[cut]);
        rewriter.eraseOp(l);
      }

    // Pass 2: wire jumps and be.
    for (auto *cut : cuts) {
      Block *b = cutBlock[cut];
      if (auto j = dyn_cast<awl::JumpOp>(cut)) {
        auto it = labels.find(j.getLabel());
        if (it == labels.end()) {
          emitError(j.getLoc()) << "undefined label '" << j.getLabel() << "'";
          return failure();
        }
        // ops after an unconditional jump are unreachable: move them to a
        // fresh dead block (kept valid by the fallthrough pass below)
        auto nextIt = std::next(j->getIterator());
        if (nextIt != b->end())
          rewriter.splitBlock(b, nextIt);
        rewriter.setInsertionPointToEnd(b);
        rewriter.create<cf::BranchOp>(loc, it->second);
        rewriter.eraseOp(j);
      } else if (auto j = dyn_cast<awl::JumpIfOp>(cut)) {
        auto it = labels.find(j.getLabel());
        if (it == labels.end()) {
          emitError(j.getLoc()) << "undefined label '" << j.getLabel() << "'";
          return failure();
        }
        auto nextIt = std::next(j->getIterator());
        Block *fall =
            nextIt != b->end() ? rewriter.splitBlock(b, nextIt) : b->getNextNode();
        if (!fall) {
          emitError(j.getLoc()) << "jump_if without fallthrough block";
          return failure();
        }
        rewriter.setInsertionPointToEnd(b);
        rewriter.create<cf::CondBranchOp>(loc, j.getRlo(), it->second,
                                          ValueRange{}, fall, ValueRange{});
        rewriter.eraseOp(j);
      } else if (auto j = dyn_cast<awl::JumpIfNotOp>(cut)) {
        auto it = labels.find(j.getLabel());
        if (it == labels.end()) {
          emitError(j.getLoc()) << "undefined label '" << j.getLabel() << "'";
          return failure();
        }
        auto nextIt = std::next(j->getIterator());
        Block *fall =
            nextIt != b->end() ? rewriter.splitBlock(b, nextIt) : b->getNextNode();
        if (!fall) {
          emitError(j.getLoc()) << "jump_if_not without fallthrough block";
          return failure();
        }
        rewriter.setInsertionPointToEnd(b);
        rewriter.create<cf::CondBranchOp>(loc, j.getRlo(), fall, ValueRange{},
                                          it->second, ValueRange{});
        rewriter.eraseOp(j);
      } else if (auto cc = dyn_cast<awl::CallOp>(cut)) {
        // CC: call only if RLO = 1. Split an empty head block in front of
        // the call block; the head gets cond_br(rlo, callBlock, fall) and
        // the call itself stays in callBlock for AwlCallOpLowering.
        Block *callB = rewriter.splitBlock(b, b->begin());
        auto nextIt = std::next(cc->getIterator());
        Block *fall =
            nextIt != callB->end() ? rewriter.splitBlock(callB, nextIt)
                                   : callB->getNextNode();
        if (!fall) {
          emitError(cc.getLoc()) << "conditional call without fallthrough block";
          return failure();
        }
        rewriter.setInsertionPointToEnd(b);
        rewriter.create<cf::CondBranchOp>(loc, cc.getRlo(), callB,
                                          ValueRange{}, fall, ValueRange{});
      } else if (auto be = dyn_cast<awl::BeOp>(cut)) {
        rewriter.setInsertionPointToEnd(b);
        rewriter.create<func::ReturnOp>(loc);
        rewriter.eraseOp(be);
      } else if (auto be = dyn_cast<awl::BeIfOp>(cut)) {
        // BEB/BEC: conditional block end -> cond_br(rlo, retBlock, fall)
        auto nextIt = std::next(be->getIterator());
        Block *fall =
            nextIt != b->end() ? rewriter.splitBlock(b, nextIt) : b->getNextNode();
        if (!fall) {
          emitError(be.getLoc()) << "be_if without fallthrough block";
          return failure();
        }
        Block *retB = rewriter.createBlock(&func.getBody());
        rewriter.setInsertionPointToEnd(retB);
        rewriter.create<func::ReturnOp>(loc);
        rewriter.setInsertionPointToEnd(b);
        mlir::Value cond = be.getRlo();
        if (be.getNegate().has_value()) {
          auto one = constInt(rewriter, loc, rewriter.getI1Type(), 1);
          cond = rewriter.create<arith::XOrIOp>(loc, cond, one);
        }
        rewriter.create<cf::CondBranchOp>(loc, cond, retB, ValueRange{}, fall,
                                          ValueRange{});
        rewriter.eraseOp(be);
      }
    }

    // Pass 3: fallthrough branches for blocks left without a terminator.
    for (Block &b : func.getBody()) {
      if (&b == &func.getBody().back())
        continue;
      if (b.empty() || !b.back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(&b);
        rewriter.create<cf::BranchOp>(loc, b.getNextNode());
      }
    }
    // The last block must end with a return (from awl.be).
    Block &last = func.getBody().back();
    if (last.empty() || !last.back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPointToEnd(&last);
      rewriter.create<func::ReturnOp>(loc);
    }

    // L-stack frame (S7 local data): push the block's frame at entry, pop it
    // before every return, so L-area accesses of nested calls stay separate.
    {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto i32 = rewriter.getI32Type();
      auto pushFn = getOrInsertRtFunc(rewriter, module, "scl_rt_l_push",
                                      rewriter.getFunctionType({i32}, {}));
      auto popFn = getOrInsertRtFunc(rewriter, module, "scl_rt_l_pop",
                                     rewriter.getFunctionType({}, {}));
      int32_t frameSize = op.getLframe() ? (int32_t)op.getLframe().value() : 0;
      int32_t paramBytes =
          op.getParamBytes() ? (int32_t)op.getParamBytes().value() : 0;
      rewriter.setInsertionPointToStart(&func.getBody().front());
      auto frameC = constInt(rewriter, loc, i32, frameSize);
      auto pushCall =
          rewriter.create<func::CallOp>(loc, pushFn, TypeRange{}, ValueRange{frameC});
      // awlsim adds the AR2 base to DI-area accesses only inside FBs
      {
        auto i8 = rewriter.getI8Type();
        auto fbFn = getOrInsertRtFunc(rewriter, module, "scl_rt_frame_set_fb",
                                      rewriter.getFunctionType({i8}, {}));
        auto fbC = constInt(rewriter, loc, i8, op.getFb() ? 1 : 0);
        rewriter.create<func::CallOp>(loc, fbFn, TypeRange{}, ValueRange{fbC});
      }
      // OB entry: write the S7 OB-temp preset header into the new frame
      if (op.getSymName().starts_with("OB")) {
        int32_t obNum = atoi(op.getSymName().substr(2).str().c_str());
        auto obFn = getOrInsertRtFunc(rewriter, module, "scl_rt_ob_start",
                                      rewriter.getFunctionType({i32}, {}));
        rewriter.setInsertionPointAfter(pushCall);
        auto obC = constInt(rewriter, loc, i32, obNum);
        rewriter.create<func::CallOp>(loc, obFn, TypeRange{}, ValueRange{obC});
      }
      // FC interface parameters: copy the caller frame's parameter area into
      // this frame on entry (and back on exit), so recursive calls keep
      // per-invocation parameters and output params propagate to the caller
      func::CallOp inCall;
      if (paramBytes > 0) {
        auto inFn = getOrInsertRtFunc(rewriter, module, "scl_rt_param_in",
                                      rewriter.getFunctionType({i32}, {}));
        rewriter.setInsertionPointAfter(pushCall);
        auto pC = constInt(rewriter, loc, i32, paramBytes);
        inCall = rewriter.create<func::CallOp>(loc, inFn, TypeRange{},
                                               ValueRange{pC});
      }
      for (Block &b : func.getBody())
        for (auto &o : llvm::make_early_inc_range(b.getOperations()))
          if (auto ret = dyn_cast<func::ReturnOp>(&o)) {
            rewriter.setInsertionPoint(ret);
            if (paramBytes > 0) {
              auto outFn = getOrInsertRtFunc(
                  rewriter, module, "scl_rt_param_out",
                  rewriter.getFunctionType({i32}, {}));
              auto pC = constInt(rewriter, loc, i32, paramBytes);
              rewriter.create<func::CallOp>(loc, outFn, TypeRange{},
                                            ValueRange{pC});
            }
            rewriter.create<func::CallOp>(loc, popFn, TypeRange{},
                                          ValueRange{});
          }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace.

namespace mlir::sclang::lowering {

void populateAwlToStdPatterns(RewritePatternSet &patterns,
                              SclTypeConverter &converter, MLIRContext *context) {
  patterns.add<
      AwlAddLowering, AwlAndLowering, AwlAssignOpLowering, AwlBlockOpLowering,
      AwlAccuGetLowering, AwlAccuSetLowering,
      AwlCallOpLowering, AwlCastOpLowering, AwlCmpEqLowering, AwlCmpGeLowering,
      AwlAssertOpLowering, AwlAssertROpLowering,
      AwlCmpGtLowering, AwlCmpLeLowering, AwlCmpLtLowering, AwlCmpNeLowering,
      AwlConstOpLowering, AwlCounterOpLowering, AwlDirectOpLowering,
      AwlDivLowering, AwlFrOpLowering, AwlLoadOpLowering, AwlMulLowering,
      AwlNegOpLowering,
      AwlNopOpLowering, AwlNotOpLowering, AwlOrLowering, AwlPackAreaOpLowering,
      AwlPackPtrOpLowering,
      AwlArGetOpLowering, AwlArRawOpLowering, AwlArSetOpLowering,
      AwlSetDivOpLowering, AwlDbInfoOpLowering,
      AwlArAddOpLowering, AwlIndirectOpLowering, AwlTimerLcOpLowering,
      AwlLoadBitIndirectOpLowering,
      AwlLoadIndirectOpLowering, AwlRealMathOpLowering, AwlSetDbOpLowering,
      AwlSetDbvOpLowering, AwlSetDbSizeOpLowering, AwlSetDbRefOpLowering,
      AwlSetDiOpLowering, AwlFeatureOpLowering,
      AwlShiftOpLowering, AwlSleepOpLowering, AwlStoreIndirectOpLowering,
      AwlF2ILowering,
      AwlStwBitLowering, AwlStwCmpLowering, AwlStwCondLowering,
      AwlStwDivLowering, AwlStwGetLowering, AwlStwResetLowering,
      AwlStwSetBitLowering, AwlStwSetLowering, AwlStwSignLowering,
      AwlTimerResetLowering, AwlWriteBitIndirectOpLowering,
      AwlRloLowering<awl::RloAndOp, false>,
      AwlRloLowering<awl::RloOrOp, true>, AwlSetResetLowering<awl::ResetOp, false>,
      AwlSetResetLowering<awl::SetOp, true>, AwlStoreOpLowering, AwlSubLowering,
      AwlTimerStartLowering, AwlXOrLowering>(converter, context);
}

} // namespace mlir::sclang::lowering
