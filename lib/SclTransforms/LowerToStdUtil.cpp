//====- LowerToStdUtil.cpp - Shared LowerToStd helpers ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Definitions of the shared helpers declared in LowerToStdInternal.h
// (layouts, runtime-ABI access, byte-buffer load/store, block utilities).
// Split out of LowerToStd.cpp purely for build parallelism — see AGENTS.md.
//
//===----------------------------------------------------------------------===//

#include "LowerToStdInternal.h"

namespace mlir::sclang::lowering {


/// Size of a type in bytes.
int64_t byteSize(Type type, const FbLayoutCache &layouts) {
  if (auto intTy = llvm::dyn_cast<scl::IntegerType>(type))
    return (intTy.getWidth() + 7) / 8;
  if (auto logTy = llvm::dyn_cast<scl::LogicalType>(type))
    return (logTy.getWidth() + 7) / 8;
  if (llvm::isa<scl::RealType>(type))
    return 4;
  if (llvm::isa<scl::LrealType>(type))
    return 8;
  if (auto arrTy = llvm::dyn_cast<scl::ArrayType>(type)) {
    int64_t count = 1;
    for (auto &dim : arrTy.getDimensions())
      count *= dim.second - dim.first + 1;
    return count * byteSize(arrTy.getElementType(), layouts);
  }
  if (auto structTy = llvm::dyn_cast<scl::StructType>(type)) {
    int64_t size = 0;
    for (auto elem : structTy.getElementTypes())
      size += byteSize(elem, layouts);
    return size;
  }
  if (auto idbTy = llvm::dyn_cast<scl::InstanceDbType>(type)) {
    auto it = layouts.find(idbTy.getFbSymbol());
    return it != layouts.end() ? it->second.size : 0;
  }
  // date/time types
  if (llvm::isa<scl::S5TimeType>(type) || llvm::isa<scl::DateType>(type))
    return 2;
  if (llvm::isa<scl::TimeType>(type) || llvm::isa<scl::TimeOfDayType>(type))
    return 4;
  if (llvm::isa<scl::DateAndTimeType>(type))
    return 8;
  return 0;
}


/// Fold a constant expression (literals + integer arithmetic) to a bit
/// pattern, used for FB member initializers (cold-start values baked into
/// the instance DB global).
std::optional<llvm::APInt> foldConstBits(Value v, unsigned width) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto c = llvm::dyn_cast<scl::ConstantOp>(def)) {
    if (auto ia = llvm::dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getValue().sextOrTrunc(width);
    if (auto fa = llvm::dyn_cast<FloatAttr>(c.getValue()))
      return fa.getValue().bitcastToAPInt().sextOrTrunc(width);
    return std::nullopt;
  }
  // date/time literals carry their storage value directly (ms / day count)
  if (auto ct = llvm::dyn_cast<scl::ConstantTimeOp>(def))
    return llvm::APInt(width, (uint64_t)(int64_t)ct.getValue(), true);
  if (auto cd = llvm::dyn_cast<scl::ConstantDateOp>(def))
    return llvm::APInt(width, (uint64_t)cd.getValue());
  if (auto ctod = llvm::dyn_cast<scl::ConstantTimeOfDayOp>(def))
    return llvm::APInt(width, (uint64_t)ctod.getValue());
  if (auto cdt = llvm::dyn_cast<scl::ConstantDateAndTimeOp>(def))
    return llvm::APInt(width, cdt.getValue());
  if (llvm::isa<scl::AddOp, scl::SubOp, scl::MulOp, scl::DivOp, scl::ModOp>(
          def)) {
    if (def->getNumOperands() != 2)
      return std::nullopt;
    auto lv = foldConstBits(def->getOperand(0), width);
    auto rv = foldConstBits(def->getOperand(1), width);
    if (!lv || !rv)
      return std::nullopt;
    if (llvm::isa<scl::AddOp>(def))
      return *lv + *rv;
    if (llvm::isa<scl::SubOp>(def))
      return *lv - *rv;
    if (llvm::isa<scl::MulOp>(def))
      return *lv * *rv;
    if (rv->isZero())
      return std::nullopt;
    if (llvm::isa<scl::DivOp>(def))
      return lv->sdiv(*rv);
    return lv->srem(*rv);
  }
  if (auto neg = llvm::dyn_cast<scl::UnaryMinusOp>(def)) {
    auto ov = foldConstBits(neg.getRhs(), width);
    if (!ov)
      return std::nullopt;
    return llvm::APInt(width, 0) - *ov;
  }
  return std::nullopt;
}


FbLayoutCache computeFbLayouts(ModuleOp module) {
  FbLayoutCache layouts;
  forceBuiltinFbLayouts(module, layouts);
  // Two passes so that FBs referencing other FBs work regardless of order.
  // Builtin standard FBs are already present (forced) and not overridden —
  // even if the frontend emitted an empty stub op for them.
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (auto fb : module.getOps<scl::FunctionBlockOp>()) {
      if (layouts.contains(fb.getName()))
        continue; // builtin standard FB: keep the forced layout
      FbLayout layout;
      for (auto var : fb.getBody().getOps<scl::VariableOp>()) {
        int64_t size = byteSize(var.getType(), layouts);
        FbMemberLayout member{var.getSymName().str(), layout.size,
                              var.getType(), std::nullopt};
        // Capture a constant initializer as a bit pattern; it is baked into
        // the instance DB global's initializer (cold start), NOT emitted in
        // the FB body (which would re-initialize on every call).
        // Array/struct literals arrive pre-folded as scl.init_bits (see
        // MLIRGen foldConstCompositeAST).
        if (auto ia = var->getAttrOfType<IntegerAttr>("scl.init_bits")) {
          member.initBits = ia.getValue();
        } else if (Value init = var.getInit()) {
          unsigned width = 64;
          if (auto it = llvm::dyn_cast<scl::IntegerType>(var.getType()))
            width = it.getWidth();
          else if (auto lt = llvm::dyn_cast<scl::LogicalType>(var.getType()))
            width = lt.getWidth();
          else if (llvm::isa<scl::RealType>(var.getType()))
            width = 32;
          else if (llvm::isa<scl::TimeType>(var.getType()) ||
                   llvm::isa<scl::TimeOfDayType>(var.getType()))
            width = 32;
          else if (llvm::isa<scl::S5TimeType>(var.getType()) ||
                   llvm::isa<scl::DateType>(var.getType()))
            width = 16;
          if (auto bits = foldConstBits(init, width))
            member.initBits = *bits;
          else if (auto sa = var->getAttrOfType<StringAttr>("scl.init_str"))
            member.initStr = sa.getValue().str();
          else if (pass == 0)
            var.emitWarning()
                << "non-constant initializer of FB member '"
                << var.getSymName() << "' in FB '" << fb.getName()
                << "' is ignored (member starts zeroed)";
        } else if (auto sa = var->getAttrOfType<StringAttr>("scl.init_str")) {
          member.initStr = sa.getValue().str();
        }
        layout.members.push_back(std::move(member));
        layout.size += size;
      }
      layouts[fb.getName()] = layout;
    }
  }
  return layouts;
}


/// Fixed layouts of the builtin standard FBs (lowered to runtime calls on
/// the instance DB buffer, not func.call), forced even if user-declared (or
/// stub-declared by the frontend), so the runtime's interpretation matches
/// (see Runtime.h scl_rt_*_fb). Bool members occupy a full byte; ints are
/// big-endian; timers use IN @0, PT @4 (i32 ms), Q @8, ET @12 (i32 ms).
void forceBuiltinFbLayouts(ModuleOp module, FbLayoutCache &layouts) {
  auto ctx = module.getContext();
  auto boolTy = scl::LogicalType::get(ctx, 1);
  auto intTy = scl::IntegerType::get(ctx, 16);
  auto timeTy = scl::TimeType::get(ctx);
  struct BuiltinMember {
    const char *name;
    int64_t off;
    Type ty;
    std::optional<llvm::APInt> initBits = std::nullopt;
  };
  struct BuiltinFb { const char *name; std::initializer_list<BuiltinMember> members; int64_t size; };
  const BuiltinFb builtins[] = {
      {"TON", {{"IN", 0, boolTy}, {"PT", 4, timeTy}, {"Q", 8, boolTy}, {"ET", 12, timeTy}}, 16},
      {"TOF", {{"IN", 0, boolTy}, {"PT", 4, timeTy}, {"Q", 8, boolTy}, {"ET", 12, timeTy}}, 16},
      {"TP", {{"IN", 0, boolTy}, {"PT", 4, timeTy}, {"Q", 8, boolTy}, {"ET", 12, timeTy}}, 16},
      // CTU: CU @0, R @1, PV @2, Q @4, CV @6, CU_OLD @8
      {"CTU", {{"CU", 0, boolTy}, {"R", 1, boolTy}, {"PV", 2, intTy}, {"Q", 4, boolTy}, {"CV", 6, intTy}, {"CU_OLD", 8, boolTy}}, 10},
      // CTD: CD @0, LD @1, PV @2, Q @4, CV @6, CD_OLD @8
      {"CTD", {{"CD", 0, boolTy}, {"LD", 1, boolTy}, {"PV", 2, intTy}, {"Q", 4, boolTy}, {"CV", 6, intTy}, {"CD_OLD", 8, boolTy}}, 10},
      // CTUD: CU @0, CD @1, R @2, LD @3, PV @4, QU @6, QD @7, CV @8, CU_OLD @10, CD_OLD @11
      {"CTUD", {{"CU", 0, boolTy}, {"CD", 1, boolTy}, {"R", 2, boolTy}, {"LD", 3, boolTy}, {"PV", 4, intTy}, {"QU", 6, boolTy}, {"QD", 7, boolTy}, {"CV", 8, intTy}, {"CU_OLD", 10, boolTy}, {"CD_OLD", 11, boolTy}}, 12},
      // R_TRIG / F_TRIG: CLK @0, Q @1, CLK_OLD @2
      {"R_TRIG", {{"CLK", 0, boolTy}, {"Q", 1, boolTy}, {"CLK_OLD", 2, boolTy}}, 4},
      // F_TRIG's edge memory starts TRUE (IEC ed.3 table 45 NOTE), so a
      // CLK=FALSE first scan already yields Q=TRUE; baked as cold-start bit.
      {"F_TRIG", {{"CLK", 0, boolTy}, {"Q", 1, boolTy}, {"CLK_OLD", 2, boolTy, llvm::APInt(1, 1)}}, 4},
      // SR: S @0, R @1, Q @2; RS: R @0, S @1, Q @2 (IEC also spells the
      // inputs S1/R1 and the output Q1 — aliased to the same offsets)
      {"SR", {{"S", 0, boolTy}, {"S1", 0, boolTy}, {"R", 1, boolTy}, {"Q", 2, boolTy}, {"Q1", 2, boolTy}}, 4},
      {"RS", {{"R", 0, boolTy}, {"R1", 0, boolTy}, {"S", 1, boolTy}, {"Q", 2, boolTy}, {"Q1", 2, boolTy}}, 4},
  };
  for (const auto &b : builtins) {
    FbLayout layout;
    for (const auto &m : b.members)
      layout.members.push_back(
          FbMemberLayout{m.name, m.off, m.ty, m.initBits});
    layout.size = b.size;
    layouts[b.name] = layout;
  }
}


/// Lower a load from a direct address (memory area / DB operand) to a call
/// of the runtime C ABI (ADR 002): `scl_rt_read_bit` for bit access, else
/// `scl_rt_read` with the width in bytes.
Value lowerDirectLoad(OpBuilder &b, Location loc, scl::DirectType dt,
                             Type resTy, ModuleOp module, Operation *op ) {
  auto i8 = b.getI8Type();
  auto i32 = b.getI32Type();
  auto i64 = b.getI64Type();
  auto areaC = constInt(b, loc, i8, (uint8_t)dt.getArea());
  auto dbC = directDbValue(b, loc, dt, module, op);
  auto byteC = constInt(b, loc, i32, dt.getByte());
  if (dt.getWidth() == 1) {
    auto bitC = constInt(b, loc, i8, dt.getBit());
    auto fn = getOrInsertRtFunc(b, module, "scl_rt_read_bit",
                                b.getFunctionType({i8, i32, i32, i8},
                                                  {b.getI1Type()}));
    return b.create<func::CallOp>(loc, fn, TypeRange{b.getI1Type()},
                                  ValueRange{areaC, dbC, byteC, bitC})
        .getResult(0);
  }
  auto widthC = constInt(b, loc, i8, dt.getWidth() / 8);
  auto fn = getOrInsertRtFunc(b, module, "scl_rt_read",
                              b.getFunctionType({i8, i32, i32, i8}, {i64}));
  Value v = b.create<func::CallOp>(loc, fn, TypeRange{i64},
                                   ValueRange{areaC, dbC, byteC, widthC})
                .getResult(0);
  if (resTy == i64)
    return v;
  if (llvm::isa<FloatType>(resTy)) {
    // REAL load: reinterpret the bits, do not convert the value
    Value iv = b.create<arith::TruncIOp>(
        loc, b.getIntegerType(resTy.getIntOrFloatBitWidth()), v);
    return b.create<arith::BitcastOp>(loc, resTy, iv);
  }
  if (resTy.isInteger(1))
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                   constInt(b, loc, i64, 0));
  return b.create<arith::TruncIOp>(loc, resTy, v);
}


/// Lower a store to a direct address (memory area / DB operand) to a call of
/// the runtime C ABI (ADR 002): `scl_rt_write_bit` for bit access, else
/// `scl_rt_write` with the value zero/sign-extended to i64 and the width in
/// bytes.
LogicalResult lowerDirectStore(OpBuilder &b, Location loc,
                                      scl::DirectType dt, Value value,
                                      ModuleOp module, Operation *op ) {
  auto i8 = b.getI8Type();
  auto i32 = b.getI32Type();
  auto i64 = b.getI64Type();
  auto areaC = constInt(b, loc, i8, (uint8_t)dt.getArea());
  auto dbC = directDbValue(b, loc, dt, module, op);
  auto byteC = constInt(b, loc, i32, dt.getByte());
  if (dt.getWidth() == 1) {
    Value bitVal = value;
    if (!bitVal.getType().isInteger(1)) {
      if (!bitVal.getType().isIntOrIndex())
        return failure();
      bitVal = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, bitVal,
          constInt(b, loc, bitVal.getType(), 0));
    }
    auto bitC = constInt(b, loc, i8, dt.getBit());
    auto fn = getOrInsertRtFunc(b, module, "scl_rt_write_bit",
                                b.getFunctionType({i8, i32, i32, i8, b.getI1Type()},
                                                  {}));
    b.create<func::CallOp>(loc, fn, TypeRange{},
                           ValueRange{areaC, dbC, byteC, bitC, bitVal});
    return success();
  }
  Value v = value;
  if (llvm::isa<FloatType>(v.getType())) {
    auto intTy = b.getIntegerType(v.getType().getIntOrFloatBitWidth());
    v = b.create<arith::BitcastOp>(loc, intTy, v);
  }
  if (!v.getType().isIntOrIndex())
    return failure();
  if (v.getType() != i64)
    v = v.getType().getIntOrFloatBitWidth() < 64
            ? (Value)b.create<arith::ExtUIOp>(loc, i64, v)
            : (Value)b.create<arith::TruncIOp>(loc, i64, v);
  auto widthC = constInt(b, loc, i8, dt.getWidth() / 8);
  auto fn = getOrInsertRtFunc(b, module, "scl_rt_write",
                              b.getFunctionType({i8, i32, i32, i8, i64}, {}));
  b.create<func::CallOp>(loc, fn, TypeRange{},
                         ValueRange{areaC, dbC, byteC, widthC, v});
  return success();
}


// MARK: LoadOpLowering

/// Load a value of `resTy` from `addr`; byte-assemble if the address is an
/// i8 view (FB/struct members in a byte buffer).
Value loadFromByteBuffer(OpBuilder &b, Location loc, Value addr,
                                Type resTy) {
  auto memTy = llvm::cast<MemRefType>(addr.getType());
  // Whole-buffer load (array value): the address of an array IS the buffer
  // memref, so the load is the address itself.
  if (llvm::isa<MemRefType>(resTy)) {
    if (resTy == memTy)
      return addr;
    // Byte view (array/struct member of an FB/IDB): cast the view to the
    // requested buffer type using the result's static size.
    auto rm = llvm::cast<MemRefType>(resTy);
    auto zero = constIndex(b, loc, 0);
    if (memTy.getRank() == 1 && memTy.getElementType().isInteger(8) &&
        rm.getRank() == 1 && rm.getElementType().isInteger(8) &&
        rm.getShape()[0] != ShapedType::kDynamic) {
      return b.create<memref::ReinterpretCastOp>(
          loc, rm, addr, rcOffsetOfr(b, rm, zero),
          ArrayRef<OpFoldResult>{b.getIndexAttr(rm.getShape()[0])},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    }
    return nullptr;
  }
  auto zero = constIndex(b, loc, 0);
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
    auto idx0 = constIndex(b, loc, 0);
    Value byte = b.create<memref::LoadOp>(loc, addr, ValueRange{idx0});
    auto zero = constInt(b, loc, i8, 0);
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, byte, zero);
  }
  auto accTy = llvm::isa<FloatType>(resTy)
                   ? IntegerType::get(b.getContext(), nbytes <= 4 ? 32 : 64)
                   : resTy;
  // addr is a rank-1 i8 view starting at the value (FB/struct member);
  // assemble the value byte by byte, big-endian (ADR 001): byte i of the
  // value lives at offset i from the start, most significant byte first.
  // Do not chain another reinterpret_cast: its fold would drop the member
  // offset.
  Value acc = constInt(b, loc, accTy, 0);
  for (int64_t i = 0; i < nbytes; ++i) {
    auto idx = constIndex(b, loc, i);
    Value byte = b.create<memref::LoadOp>(loc, addr, ValueRange{idx});
    Value ext = b.create<arith::ExtUIOp>(loc, accTy, byte);
    if (i < nbytes - 1) {
      auto sh = constInt(b, loc, accTy, 8 * (nbytes - 1 - i));
      ext = b.create<arith::ShLIOp>(loc, ext, sh);
    }
    acc = b.create<arith::OrIOp>(loc, acc, ext);
  }
  if (llvm::isa<FloatType>(resTy))
    return b.create<arith::BitcastOp>(loc, resTy, acc);
  if (accTy != resTy)
    return b.create<arith::TruncIOp>(loc, resTy, acc);
  return acc;
}


// MARK: StoreOpLowering

/// Store `value` at `addr`, splitting into bytes if the address is an i8 view.
LogicalResult storeToByteBuffer(OpBuilder &b, Location loc, Value value,
                                       Value addr) {
  auto memTy = llvm::cast<MemRefType>(addr.getType());
  auto zero = constIndex(b, loc, 0);
  // Whole-buffer assignment (array/struct value stored into its buffer).
  if (llvm::isa<MemRefType>(value.getType()) && llvm::isa<MemRefType>(addr.getType())) {
    Value src = value;
    auto am = llvm::cast<MemRefType>(addr.getType());
    if (src.getType() != addr.getType()) {
      auto sm = llvm::cast<MemRefType>(src.getType());
      if (sm.getRank() != 1 || sm.getElementType() != am.getElementType())
        return failure();
      if (am.getRank() == 1 && am.getElementType().isInteger(8) &&
          am.getShape()[0] != ShapedType::kDynamic) {
        // struct window address: cast the buffer to the window type first
        src = b.create<memref::ReinterpretCastOp>(
            loc, am, value, rcOffsetOfr(b, am, zero),
            ArrayRef<OpFoldResult>{b.getIndexAttr(am.getShape()[0])},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      } else if (am.getRank() == 1 && am.getElementType().isInteger(8) &&
                 am.getShape()[0] == ShapedType::kDynamic &&
                 sm.getShape()[0] != ShapedType::kDynamic) {
        // dynamic byte view (array/struct member of an FB/IDB): cast the
        // source buffer to the view type using the source's static size
        src = b.create<memref::ReinterpretCastOp>(
            loc, am, src, rcOffsetOfr(b, am, zero),
            ArrayRef<OpFoldResult>{b.getIndexAttr(sm.getShape()[0])},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      } else {
        return failure();
      }
    }
    b.create<memref::CopyOp>(loc, src, addr);
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
    auto idx0 = constIndex(b, loc, 0);
    b.create<memref::StoreOp>(loc, byte, addr, ValueRange{idx0});
    return success();
  }
  auto accTy = llvm::isa<FloatType>(valTy)
                   ? IntegerType::get(b.getContext(), nbytes <= 4 ? 32 : 64)
                   : valTy;
  Value acc = value;
  if (llvm::isa<FloatType>(valTy))
    acc = b.create<arith::BitcastOp>(loc, accTy, value);
  else if (accTy != valTy)
    acc = b.create<arith::ExtSIOp>(loc, accTy, value);
  // Storing a wider value into a narrower byte view (e.g. int -> byte var)
  // truncates to the view's width, keeping the low-order bytes.
  int64_t viewSize = memTy.getShape()[0];
  if (viewSize != ShapedType::kDynamic && viewSize < nbytes) {
    accTy = IntegerType::get(b.getContext(), (unsigned)viewSize * 8);
    acc = b.create<arith::TruncIOp>(loc, accTy, acc);
    nbytes = viewSize;
  }
  // addr is a rank-1 i8 view starting at the value (FB/struct member);
  // split the value into bytes and store them directly, big-endian
  // (ADR 001): byte i of the value lives at offset i, most significant
  // byte first.
  for (int64_t i = 0; i < nbytes; ++i) {
    Value byte = acc;
    if (i < nbytes - 1) {
      auto sh = constInt(b, loc, accTy, 8 * (nbytes - 1 - i));
      byte = b.create<arith::ShRUIOp>(loc, acc, sh);
    }
    byte = b.create<arith::TruncIOp>(loc, i8, byte);
    auto idx = constIndex(b, loc, i);
    b.create<memref::StoreOp>(loc, byte, addr, ValueRange{idx});
  }
  return success();
}


/// Walk a chain of memref.reinterpret_cast byte views down to the root
/// buffer, accumulating the offset (constant part into `constOff`, dynamic
/// part into `dynOff`). Returns the root value, or null if the chain bottoms
/// out at something that is not a memref. Used to create a *single* cast from
/// the root: chained reinterpret_casts would lose the inner offset when
/// ReinterpretCastOp::fold fires.
Value resolveByteViewRoot(Value v, Value &dynOff, int64_t &constOff,
                                 OpBuilder &b, Location loc) {
  constOff = 0;
  dynOff = nullptr;
  auto indexTy = b.getIndexType();
  while (auto rc = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    OpFoldResult ofr = rc.getMixedOffsets()[0];
    if (auto c = mlir::getConstantIntValue(ofr)) {
      constOff += *c;
    } else if (Value o = llvm::dyn_cast<Value>(ofr)) {
      if (o.getType() != indexTy)
        o = b.create<arith::IndexCastOp>(loc, indexTy, o);
      dynOff = dynOff ? (Value)b.create<arith::AddIOp>(loc, dynOff, o) : o;
    } else {
      return nullptr;
    }
    v = rc.getSource();
  }
  if (!llvm::isa<MemRefType>(v.getType()))
    return nullptr;
  return v;
}


/// Combine a constant offset, a dynamic offset value and an extra dynamic
/// offset into a single index value.
Value combineOffsets(OpBuilder &b, Location loc, int64_t constOff,
                            Value dynOff, Value extra) {
  Value res = extra;
  if (dynOff)
    res = res ? (Value)b.create<arith::AddIOp>(loc, res, dynOff) : dynOff;
  if (constOff != 0) {
    auto c = constIndex(b, loc, constOff);
    res = res ? (Value)b.create<arith::AddIOp>(loc, res, c) : c;
  }
  if (!res)
    res = constIndex(b, loc, 0);
  return res;
}


/// Create a rank-1 i8 byte view of `size` bytes at `off` from a single cast
/// of the root buffer.
Value makeByteView(OpBuilder &b, Location loc, Value root, Value off,
                          int64_t size, MLIRContext *ctx,
                          bool staticSize ) {
  auto i8 = IntegerType::get(ctx, 8);
  auto resTy = MemRefType::get(
      {staticSize ? size : ShapedType::kDynamic}, i8,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                             {staticSize ? 1 : ShapedType::kDynamic}));
  OpFoldResult sizeOfr =
      staticSize ? OpFoldResult(b.getIndexAttr(size))
                 : OpFoldResult(constIndex(b, loc, size));
  OpFoldResult strideOfr =
      staticSize ? OpFoldResult(b.getIndexAttr(1))
                 : OpFoldResult(constIndex(b, loc, 1));
  return b.create<memref::ReinterpretCastOp>(loc, resTy, root, OpFoldResult(off),
                                             ArrayRef<OpFoldResult>{sizeOfr},
                                             ArrayRef<OpFoldResult>{strideOfr});
}


/// Create a rank-1 i8 byte view at byte offset `memberOff` within `src`,
/// folding in the runtime offset of `src` itself (e.g. an FB instance-DB
/// parameter that is a view with a nonzero offset). reinterpret_cast offsets
/// are absolute w.r.t. the base buffer, so resolve the root of any cast chain,
/// take the root's metadata (base buffer + its own offset), and emit a single
/// cast from the base buffer with the total offset.
Value makeByteViewFromSource(OpBuilder &b, Location loc, Value src,
                                    int64_t memberOff, int64_t size,
                                    MLIRContext *ctx, bool staticSize ) {
  Value dynOff;
  int64_t constOff;
  Value root = resolveByteViewRoot(src, dynOff, constOff, b, loc);
  if (!root || !llvm::isa<MemRefType>(root.getType()))
    return nullptr;
  auto meta = b.create<memref::ExtractStridedMetadataOp>(loc, root);
  Value off =
      combineOffsets(b, loc, constOff + memberOff, dynOff, meta.getOffset());
  return makeByteView(b, loc, meta.getBaseBuffer(), off, size, ctx,
                      staticSize);
}


// MARK: region moving helpers

/// Move all blocks of `src` into `dst` before `before`; returns the moved
/// entry block. After the move, `src` is empty.
Block *moveRegion(ConversionPatternRewriter &rewriter, Region &src,
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
void lowerTerminatorsInBlocks(ConversionPatternRewriter &rewriter,
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


/// Record continue/exit ops in `blocks` and in regions nested inside non-loop
/// ops (if/case bodies), so their lowerings can branch to the loop targets.
/// Nested loop ops are skipped: they record their own bodies.
void recordLoopTerminators(Operation *op, Block *contB, Block *exitB,
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


void recordLoopTerminators(ArrayRef<Block *> blocks, Block *contB,
                                  Block *exitB, LoopTerminatorTargets &targets) {
  for (Block *b : blocks)
    for (Operation &o : *b) {
      if (isa<scl::ContinueOp, scl::ExitOp>(&o))
        targets.map[&o] = {contB, exitB};
      else
        recordLoopTerminators(&o, contB, exitB, targets);
    }
}


/// Collect the blocks strictly between `from` (exclusive) and `to`
/// (exclusive).
SmallVector<Block *> blocksBetween(Block *from, Block *to) {
  SmallVector<Block *> result;
  for (Block *b = from->getNextNode(); b && b != to; b = b->getNextNode())
    result.push_back(b);
  return result;
}


/// Collect the blocks from `from` (inclusive) up to `to` (exclusive).
SmallVector<Block *> blocksFrom(Block *from, Block *to) {
  SmallVector<Block *> result;
  for (Block *b = from; b && b != to; b = b->getNextNode())
    result.push_back(b);
  return result;
}



// MARK: AWL dialect lowering (ADR 003)
//
// The awl dialect (AWL/STL frontend output and SCL->AWL backend target)
// converges at std IR together with scl: !scl.direct operands lower to the
// runtime C ABI calls (ADR 002), accumulator/RLO ops to arith, jumps to cf
// blocks.

/// Convert an int/float value to `ty` (ext/trunc/sitofp/fptosi).
Value convertScalarTo(OpBuilder &b, Location loc, Value v, Type ty,
                             bool zext) {
  Type src = v.getType();
  if (src == ty)
    return v;
  if (llvm::isa<FloatType>(src) && llvm::isa<FloatType>(ty)) {
    if (src.getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
      return mlir::sclang::withFPFastMath(b.create<arith::ExtFOp>(loc, ty, v));
    if (src.getIntOrFloatBitWidth() > ty.getIntOrFloatBitWidth())
      return mlir::sclang::withFPFastMath(b.create<arith::TruncFOp>(loc, ty, v));
    return v;
  }
  if (llvm::isa<FloatType>(src) && llvm::isa<IntegerType>(ty))
    return b.create<arith::FPToSIOp>(loc, ty, v);
  if (llvm::isa<IntegerType>(src) && llvm::isa<FloatType>(ty))
    return b.create<arith::SIToFPOp>(loc, ty, v);
  if (llvm::isa<IntegerType>(src) && llvm::isa<IntegerType>(ty)) {
    if (src.getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth()) {
      // a bit (i1) is an unsigned 0/1, not a sign bit; zext forces
      // zero-extension for unsigned loads (BYTE/WORD into the ACCU)
      if (src.getIntOrFloatBitWidth() == 1 || zext)
        return b.create<arith::ExtUIOp>(loc, ty, v);
      return b.create<arith::ExtSIOp>(loc, ty, v);
    }
    return b.create<arith::TruncIOp>(loc, ty, v);
  }
  return nullptr;
}

} // namespace mlir::sclang::lowering
