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
#include "sclang/AwlDialect/AwlOps.h"
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

/// Convert an int/float value to `ty` (ext/trunc/sitofp/fptosi).
/// Defined further below; declared here for DialectCastOpLowering.
static Value convertScalarTo(OpBuilder &b, Location loc, Value v, Type ty,
                             bool zext = false);

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
  /// Cold-start string initializer (chars, zero-padded) for STRING members.
  std::optional<std::string> initStr;
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

/// Fold a constant expression (literals + integer arithmetic) to a bit
/// pattern, used for FB member initializers (cold-start values baked into
/// the instance DB global).
static std::optional<llvm::APInt> foldConstBits(Value v, unsigned width) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto c = llvm::dyn_cast<scl::ConstantOp>(def)) {
    if (auto ia = c.getValue().dyn_cast<IntegerAttr>())
      return ia.getValue().sextOrTrunc(width);
    if (auto fa = c.getValue().dyn_cast<FloatAttr>())
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

/// Compute the layouts of all function blocks in the module.
static void forceBuiltinFbLayouts(ModuleOp module, FbLayoutCache &layouts);

static FbLayoutCache computeFbLayouts(ModuleOp module) {
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
          if (auto it = var.getType().dyn_cast<scl::IntegerType>())
            width = it.getWidth();
          else if (auto lt = var.getType().dyn_cast<scl::LogicalType>())
            width = lt.getWidth();
          else if (var.getType().isa<scl::RealType>())
            width = 32;
          else if (var.getType().isa<scl::TimeType>() ||
                   var.getType().isa<scl::TimeOfDayType>())
            width = 32;
          else if (var.getType().isa<scl::S5TimeType>() ||
                   var.getType().isa<scl::DateType>())
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
static void forceBuiltinFbLayouts(ModuleOp module, FbLayoutCache &layouts) {
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

// MARK: SclTypeConverter

class SclTypeConverter : public TypeConverter {
public:
  FbLayoutCache fbLayouts;

  SclTypeConverter() : SclTypeConverter(FbLayoutCache{}) {}
  explicit SclTypeConverter(FbLayoutCache layouts) : fbLayouts(std::move(layouts)) {
    addConversion([&](scl::AddressType type) {
      auto elem = type.getElementType();
      // A direct address (memory area / DB operand) converts to a packed i64
      // placeholder; loads/stores through it lower to runtime C ABI calls
      // (ADR 002) and never touch this value.
      if (elem.isa<scl::DirectType>())
        return (Type)IntegerType::get(type.getContext(), 64);
      // An instance DB address is a rank-1 i8 view of the instance buffer
      // with a dynamic offset (like any other byte view). Keeping the view
      // type here means FB calls on local instances (FB-typed members, which
      // are byte views into the parent IDB) pass the view directly — no cast,
      // and no reinterpret_cast chaining (whose fold would drop the inner
      // offset). Global instance DBs are get_global'ed as plain memrefs and
      // memref.cast to this view type in GetGlobalOpLowering.
      if (auto idbTy = elem.dyn_cast<scl::InstanceDbType>()) {
        auto it = fbLayouts.find(idbTy.getFbSymbol());
        if (it == fbLayouts.end())
          return Type();
        return (Type)mlir::MemRefType::get(
            {ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8),
            StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
                                   {1}));
      }
      auto celem = convertType(elem);
      if (!celem)
        return Type();
      // An address of a struct is a rank-1 i8 window of the struct buffer
      // (static size, dynamic offset), so that getelement/getindex on
      // struct array elements chain through resolveByteViewRoot with a
      // dynamic offset.
      if (auto st = elem.dyn_cast<scl::StructType>())
        return (Type)mlir::MemRefType::get(
            {byteSize(st, fbLayouts)}, IntegerType::get(type.getContext(), 8),
            StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
                                   {1}));
      // An address of a composite (array/IDB) IS the buffer memref.
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
    addConversion([&](scl::DirectType type) {
      return IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](scl::ArrayType type) {
      SmallVector<int64_t, 2> shape;
      for (auto &dim : type.getDimensions())
        shape.push_back(dim.second - dim.first + 1);
      auto elem = convertType(type.getElementType());
      if (!elem)
        return Type();
      // An array of composites (struct/array elements) is a flat byte
      // buffer, so that element addresses take the byte-view path in
      // GetIndexOpLowering (a memref<N x memref<…>> would not).
      if (elem.isa<MemRefType>())
        return (Type)MemRefType::get(
            {byteSize(type, fbLayouts)}, IntegerType::get(type.getContext(), 8));
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

// MARK: runtime C ABI (ADR 002)

/// Get or insert a private declaration of a runtime ABI function (ADR 002)
/// into the module, returning its symbol reference.
static FlatSymbolRefAttr getOrInsertRtFunc(OpBuilder &b, ModuleOp module,
                                           StringRef name, FunctionType type) {
  if (module.lookupSymbol<func::FuncOp>(name))
    return SymbolRefAttr::get(b.getContext(), name);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setVisibility(SymbolTable::Visibility::Private);
  return SymbolRefAttr::get(b.getContext(), name);
}

/// The db operand for a direct access: inside an FB (awl.block "FBn" lowered
/// to func @FBn(i32)), DB operands without a number (db = 0) address the
/// instance DB, passed as the function's i32 argument.
static Value directDbValue(OpBuilder &b, Location loc, scl::DirectType dt,
                           ModuleOp module, Operation *op) {
  auto i32 = b.getI32Type();
  if (dt.getArea() == scl::DirectArea::DataBlock && dt.getDb() == -1) {
    // DI-area operand: pass the DI marker 0xFFFF through, so the runtime
    // resolves it to the DI-register DB *and* adds the AR2 multi-instance
    // base offset (awlsim cpu.py getArea_DI/setArea_DI)
    return b.create<arith::ConstantOp>(
        loc, i32, b.getIntegerAttr(i32, (int32_t)0xFFFF));
  }
  if (op && dt.getArea() == scl::DirectArea::DataBlock && dt.getDb() == 0) {
    if (auto func = op->template getParentOfType<func::FuncOp>()) {
      if (func.getSymName().startswith("FB") && func.getNumArguments() == 1 &&
          func.getBody().front().getNumArguments() == 1)
        return func.getBody().front().getArgument(0);
    }
  }
  return b.create<arith::ConstantOp>(loc, i32,
                                     b.getIntegerAttr(i32, dt.getDb()));
}

/// Lower a load from a direct address (memory area / DB operand) to a call
/// of the runtime C ABI (ADR 002): `scl_rt_read_bit` for bit access, else
/// `scl_rt_read` with the width in bytes.
static Value lowerDirectLoad(OpBuilder &b, Location loc, scl::DirectType dt,
                             Type resTy, ModuleOp module, Operation *op = nullptr) {
  auto i8 = b.getI8Type();
  auto i32 = b.getI32Type();
  auto i64 = b.getI64Type();
  auto areaC = b.create<arith::ConstantOp>(
      loc, i8, b.getIntegerAttr(i8, (uint8_t)dt.getArea()));
  auto dbC = directDbValue(b, loc, dt, module, op);
  auto byteC = b.create<arith::ConstantOp>(loc, i32,
                                           b.getIntegerAttr(i32, dt.getByte()));
  if (dt.getWidth() == 1) {
    auto bitC = b.create<arith::ConstantOp>(
        loc, i8, b.getIntegerAttr(i8, dt.getBit()));
    auto fn = getOrInsertRtFunc(b, module, "scl_rt_read_bit",
                                b.getFunctionType({i8, i32, i32, i8},
                                                  {b.getI1Type()}));
    return b.create<func::CallOp>(loc, fn, TypeRange{b.getI1Type()},
                                  ValueRange{areaC, dbC, byteC, bitC})
        .getResult(0);
  }
  auto widthC = b.create<arith::ConstantOp>(
      loc, i8, b.getIntegerAttr(i8, dt.getWidth() / 8));
  auto fn = getOrInsertRtFunc(b, module, "scl_rt_read",
                              b.getFunctionType({i8, i32, i32, i8}, {i64}));
  Value v = b.create<func::CallOp>(loc, fn, TypeRange{i64},
                                   ValueRange{areaC, dbC, byteC, widthC})
                .getResult(0);
  if (resTy == i64)
    return v;
  if (resTy.isa<FloatType>()) {
    // REAL load: reinterpret the bits, do not convert the value
    Value iv = b.create<arith::TruncIOp>(
        loc, b.getIntegerType(resTy.getIntOrFloatBitWidth()), v);
    return b.create<arith::BitcastOp>(loc, resTy, iv);
  }
  if (resTy.isInteger(1))
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                   b.create<arith::ConstantOp>(
                                       loc, i64, b.getIntegerAttr(i64, 0)));
  return b.create<arith::TruncIOp>(loc, resTy, v);
}

/// Lower a store to a direct address (memory area / DB operand) to a call of
/// the runtime C ABI (ADR 002): `scl_rt_write_bit` for bit access, else
/// `scl_rt_write` with the value zero/sign-extended to i64 and the width in
/// bytes.
static LogicalResult lowerDirectStore(OpBuilder &b, Location loc,
                                      scl::DirectType dt, Value value,
                                      ModuleOp module, Operation *op = nullptr) {
  auto i8 = b.getI8Type();
  auto i32 = b.getI32Type();
  auto i64 = b.getI64Type();
  auto areaC = b.create<arith::ConstantOp>(
      loc, i8, b.getIntegerAttr(i8, (uint8_t)dt.getArea()));
  auto dbC = directDbValue(b, loc, dt, module, op);
  auto byteC = b.create<arith::ConstantOp>(loc, i32,
                                           b.getIntegerAttr(i32, dt.getByte()));
  if (dt.getWidth() == 1) {
    Value bitVal = value;
    if (!bitVal.getType().isInteger(1)) {
      if (!bitVal.getType().isIntOrIndex())
        return failure();
      bitVal = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, bitVal,
          b.create<arith::ConstantOp>(
              loc, bitVal.getType(),
              b.getIntegerAttr(bitVal.getType(), 0)));
    }
    auto bitC = b.create<arith::ConstantOp>(
        loc, i8, b.getIntegerAttr(i8, dt.getBit()));
    auto fn = getOrInsertRtFunc(b, module, "scl_rt_write_bit",
                                b.getFunctionType({i8, i32, i32, i8, b.getI1Type()},
                                                  {}));
    b.create<func::CallOp>(loc, fn, TypeRange{},
                           ValueRange{areaC, dbC, byteC, bitC, bitVal});
    return success();
  }
  Value v = value;
  if (v.getType().isa<FloatType>()) {
    auto intTy = b.getIntegerType(v.getType().getIntOrFloatBitWidth());
    v = b.create<arith::BitcastOp>(loc, intTy, v);
  }
  if (!v.getType().isIntOrIndex())
    return failure();
  if (v.getType() != i64)
    v = v.getType().getIntOrFloatBitWidth() < 64
            ? (Value)b.create<arith::ExtUIOp>(loc, i64, v)
            : (Value)b.create<arith::TruncIOp>(loc, i64, v);
  auto widthC = b.create<arith::ConstantOp>(
      loc, i8, b.getIntegerAttr(i8, dt.getWidth() / 8));
  auto fn = getOrInsertRtFunc(b, module, "scl_rt_write",
                              b.getFunctionType({i8, i32, i32, i8, i64}, {}));
  b.create<func::CallOp>(loc, fn, TypeRange{},
                         ValueRange{areaC, dbC, byteC, widthC, v});
  return success();
}

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
    auto module = op->template getParentOfType<ModuleOp>();
    if (auto callee = llvm::dyn_cast_or_null<func::FuncOp>(
            module.lookupSymbol(op.getCallee()))) {
      auto fnTy = callee.getFunctionType();
      for (auto [i, arg] : llvm::enumerate(args)) {
        if (i >= fnTy.getNumInputs())
          break;
        auto target = fnTy.getInput(i);
        auto argMem = arg.getType().dyn_cast<MemRefType>();
        auto tgtMem = target.dyn_cast<MemRefType>();
        if (!argMem || !tgtMem || arg.getType() == target)
          continue;
        auto loc = op.getLoc();
        auto meta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, arg);
        SmallVector<OpFoldResult, 1> sizes, strides;
        for (int64_t d : tgtMem.getShape()) {
          sizes.push_back(rewriter.getIndexAttr(d));
          strides.push_back(rewriter.getIndexAttr(1));
        }
        args[i] = rewriter.create<memref::ReinterpretCastOp>(
            loc, tgtMem, meta.getBaseBuffer(), meta.getOffset(), sizes,
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
    // The result type must be the converted type; retyping the attribute
    // accordingly (a bare arith.constant with a !scl.*-typed attribute is not
    // legalizable by the LLVM conversion, e.g. !scl.char char literals).
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();
    auto attr = op.getValue().cast<TypedAttr>();
    if (attr.getType() != resTy) {
      if (auto ia = attr.dyn_cast<IntegerAttr>())
        attr = rewriter.getIntegerAttr(resTy, ia.getValue());
      else if (auto fa = attr.dyn_cast<FloatAttr>())
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
      auto module = op->template getParentOfType<ModuleOp>();
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

// MARK: LoadOpLowering

/// Load a value of `resTy` from `addr`; byte-assemble if the address is an
/// i8 view (FB/struct members in a byte buffer).
static Value loadFromByteBuffer(OpBuilder &b, Location loc, Value addr,
                                Type resTy) {
  auto memTy = addr.getType().cast<MemRefType>();
  // Whole-buffer load (array value): the address of an array IS the buffer
  // memref, so the load is the address itself.
  if (resTy.isa<MemRefType>()) {
    if (resTy == memTy)
      return addr;
    // Byte view (array/struct member of an FB/IDB): cast the view to the
    // requested buffer type using the result's static size.
    auto rm = resTy.cast<MemRefType>();
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
    if (memTy.getRank() == 1 && memTy.getElementType().isInteger(8) &&
        rm.getRank() == 1 && rm.getElementType().isInteger(8) &&
        rm.getShape()[0] != ShapedType::kDynamic) {
      auto n = b.create<arith::ConstantIndexOp>(loc, rm.getShape()[0]);
      auto one = b.create<arith::ConstantIndexOp>(loc, 1);
      return b.create<memref::ReinterpretCastOp>(loc, rm, addr, zero,
                                                 ValueRange{n},
                                                 ValueRange{one});
    }
    return nullptr;
  }
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
  // assemble the value byte by byte, big-endian (ADR 001): byte i of the
  // value lives at offset i from the start, most significant byte first.
  // Do not chain another reinterpret_cast: its fold would drop the member
  // offset.
  Value acc = b.create<arith::ConstantOp>(loc, accTy,
                                          b.getIntegerAttr(accTy, 0));
  for (int64_t i = 0; i < nbytes; ++i) {
    auto idx = b.create<arith::ConstantIndexOp>(loc, i);
    Value byte = b.create<memref::LoadOp>(loc, addr, ValueRange{idx});
    Value ext = b.create<arith::ExtUIOp>(loc, accTy, byte);
    if (i < nbytes - 1) {
      auto sh = b.create<arith::ConstantOp>(
          loc, accTy, b.getIntegerAttr(accTy, 8 * (nbytes - 1 - i)));
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
    // Direct address (memory area / DB): runtime C ABI call (ADR 002).
    if (auto dt = op.getAddress().getType().cast<scl::AddressType>()
                      .getElementType().dyn_cast<scl::DirectType>()) {
      Value loaded = lowerDirectLoad(rewriter, op.getLoc(), dt, resTy,
                                     op->template getParentOfType<ModuleOp>());
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
  if (value.getType().isa<MemRefType>() && addr.getType().isa<MemRefType>()) {
    Value src = value;
    auto am = addr.getType().cast<MemRefType>();
    if (src.getType() != addr.getType()) {
      auto sm = src.getType().cast<MemRefType>();
      if (sm.getRank() != 1 || sm.getElementType() != am.getElementType())
        return failure();
      if (am.getRank() == 1 && am.getElementType().isInteger(8) &&
          am.getShape()[0] != ShapedType::kDynamic) {
        // struct window address: cast the buffer to the window type first
        auto n = b.create<arith::ConstantIndexOp>(loc, am.getShape()[0]);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        src = b.create<memref::ReinterpretCastOp>(loc, am, value, zero,
                                                 ValueRange{n}, ValueRange{one});
      } else if (am.getRank() == 1 && am.getElementType().isInteger(8) &&
                 am.getShape()[0] == ShapedType::kDynamic &&
                 sm.getShape()[0] != ShapedType::kDynamic) {
        // dynamic byte view (array/struct member of an FB/IDB): cast the
        // source buffer to the view type using the source's static size
        auto n = b.create<arith::ConstantIndexOp>(loc, sm.getShape()[0]);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        src = b.create<memref::ReinterpretCastOp>(loc, am, src, zero,
                                                 ValueRange{n}, ValueRange{one});
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
      auto sh = b.create<arith::ConstantOp>(
          loc, accTy, b.getIntegerAttr(accTy, 8 * (nbytes - 1 - i)));
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
    // Direct address (memory area / DB): runtime C ABI call (ADR 002).
    if (auto dt = op.getLhs().getType().cast<scl::AddressType>()
                      .getElementType().dyn_cast<scl::DirectType>()) {
      if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getRhs(),
                                  op->template getParentOfType<ModuleOp>(), op)))
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
      if (memref.getShape()[0] != ShapedType::kDynamic) {
        // struct window: alloca the whole buffer and cast to the window type
        auto plain = MemRefType::get({memref.getShape()[0]},
                                     memref.getElementType());
        auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
        auto n = rewriter.create<arith::ConstantIndexOp>(loc, memref.getShape()[0]);
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
            op, memref, alloca, zero, ValueRange{n}, ValueRange{one});
        return success();
      }
      // i8 address: rank-1 byte view holding a single byte. An instance-DB
      // address allocates the FB's whole instance buffer instead.
      int64_t allocSize = 1;
      if (auto idbTy = op.getResult().getType()
                           .cast<scl::AddressType>()
                           .getElementType()
                           .dyn_cast<scl::InstanceDbType>()) {
        auto &layouts =
            static_cast<const SclTypeConverter *>(typeConverter)->fbLayouts;
        auto it = layouts.find(idbTy.getFbSymbol());
        if (it != layouts.end())
          allocSize = it->second.size;
      }
      auto plain = MemRefType::get({allocSize}, memref.getElementType());
      auto alloca = rewriter.create<memref::AllocaOp>(loc, plain);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto n = rewriter.create<arith::ConstantIndexOp>(loc, allocSize);
      rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
          op, memref, alloca, zero, ValueRange{n}, ValueRange{one});
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
    Value v = adaptor.getRhs();
    if (v.getType().isInteger(1)) {
      // logical NOT on a bool
      auto i1 = rewriter.getIntegerType(1);
      auto falseVal = rewriter.create<ConstantIntOp>(loc, 0, i1);
      auto trueVal = rewriter.create<ConstantIntOp>(loc, 1, i1);
      rewriter.replaceOpWithNewOp<SelectOp>(op, v, falseVal, trueVal);
      return success();
    }
    // bitwise NOT on a bit string (BYTE/WORD/DWORD): x XOR all-ones
    auto allOnes = rewriter.create<ConstantIntOp>(
        loc, -1, v.getType().cast<IntegerType>().getWidth());
    rewriter.replaceOpWithNewOp<XOrIOp>(op, v, allOnes);
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
      // 0-d memref. NB: the global always keeps a plain identity-layout type
      // because the LLVM lowering of memref.global requires identity maps;
      // memref.get_global results that need a strided window type are cast in
      // GetGlobalOpLowering.
      if (elemTy.isa<MemRefType>())
        memrefTy = elemTy;
      else
        memrefTy = MemRefType::get({}, elemTy);
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
        // Bake cold-start initializers into the byte buffer, recursing into
        // FB-typed members (their builtin/user layouts may carry their own
        // initializers, e.g. F_TRIG's edge memory starts TRUE).
        std::function<void(const FbLayout &, int64_t)> bake =
            [&](const FbLayout &lay, int64_t baseOff) {
          for (auto &m : lay.members) {
            if (auto idbM = m.type.dyn_cast<scl::InstanceDbType>()) {
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
              if (auto arrTy = m.type.dyn_cast<scl::ArrayType>())
                if (arrTy.getElementType().isa<scl::IntegerType>() &&
                    arrTy.getElementType().cast<scl::IntegerType>().getWidth() == 8) {
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
          if (auto initDict = initOpt->dyn_cast<mlir::DictionaryAttr>()) {
            for (auto na : initDict.getValue()) {
              std::string mname = na.getName().str();
              auto ia = na.getValue().dyn_cast<IntegerAttr>();
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
    if (auto memrefRes = resTy.dyn_cast<MemRefType>();
        memrefRes && memrefRes.getLayout()) {
      // An instance-DB address converts to a dynamic-size byte view; the
      // global itself must use the FB layout's static size.
      SmallVector<int64_t, 1> shape(memrefRes.getShape());
      if (auto addrTy = op.getResult().getType()
                            .dyn_cast<scl::AddressType>()) {
        if (auto idbTy =
                addrTy.getElementType().dyn_cast<scl::InstanceDbType>()) {
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
                          int64_t size, MLIRContext *ctx,
                          bool staticSize = false) {
  auto i8 = IntegerType::get(ctx, 8);
  auto resTy = MemRefType::get(
      {staticSize ? size : ShapedType::kDynamic}, i8,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}));
  auto sizeV = b.create<arith::ConstantIndexOp>(loc, size);
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  return b.create<memref::ReinterpretCastOp>(loc, resTy, root, off,
                                             ValueRange{sizeV},
                                             ValueRange{one});
}

/// Create a rank-1 i8 byte view at byte offset `memberOff` within `src`,
/// folding in the runtime offset of `src` itself (e.g. an FB instance-DB
/// parameter that is a view with a nonzero offset). reinterpret_cast offsets
/// are absolute w.r.t. the base buffer, so resolve the root of any cast chain,
/// take the root's metadata (base buffer + its own offset), and emit a single
/// cast from the base buffer with the total offset.
static Value makeByteViewFromSource(OpBuilder &b, Location loc, Value src,
                                    int64_t memberOff, int64_t size,
                                    MLIRContext *ctx, bool staticSize = false) {
  Value dynOff;
  int64_t constOff;
  Value root = resolveByteViewRoot(src, dynOff, constOff, b, loc);
  if (!root || !root.getType().isa<MemRefType>())
    return nullptr;
  auto meta = b.create<memref::ExtractStridedMetadataOp>(loc, root);
  Value off =
      combineOffsets(b, loc, constOff + memberOff, dynOff, meta.getOffset());
  return makeByteView(b, loc, meta.getBaseBuffer(), off, size, ctx,
                      staticSize);
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
      Value byteFlat = flat;
      if (elemBytes != 1) {
        auto scale = rewriter.create<arith::ConstantIndexOp>(loc, elemBytes);
        byteFlat = rewriter.create<arith::MulIOp>(loc, flat, scale);
      }
      bool compositeElem =
          arrayTy.getElementType().isa<scl::StructType, scl::ArrayType>();
      // A static-size window is only correct when the result is an
      // address<composite> (plain array variable, no arrayType attr); the
      // byte-view path (array member of a FB/struct) yields address<i8>,
      // which converts to a dynamic-size view.
      bool staticSize = compositeElem && !op.getArrayType().has_value();
      // Fold in the byte index and the base view's own runtime offset.
      Value dynOff;
      int64_t constOff;
      Value root = resolveByteViewRoot(base, dynOff, constOff, rewriter, loc);
      if (!root || !root.getType().isa<MemRefType>())
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
    auto dt = op.getResult().getType().cast<scl::AddressType>()
                  .getElementType().cast<scl::DirectType>();
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
    auto addrTy = op.getAddr().getType().cast<scl::AddressType>();
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
      if (auto st = addrTy.getElementType().dyn_cast<scl::StructType>())
        structTy = st;
      else if (op.getBaseType().has_value())
        structTy = op.getBaseType()->cast<scl::StructType>();
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
    // the FbLayoutCache match the runtime's interpretation).
    std::string fbUp = llvm::StringRef(op.getFb()).upper();
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
      ModuleOp module = op->template getParentOfType<ModuleOp>();
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
      Value modeC = rewriter.create<arith::ConstantOp>(
          loc, i8, rewriter.getI8IntegerAttr((int8_t)mode));
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
    auto resTy = typeConverter->convertType(op.getType()).cast<IntegerType>();
    auto srcTy = adaptor.getValue().getType().cast<IntegerType>();
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


// MARK: AWL dialect lowering (ADR 003)
//
// The awl dialect (AWL/STL frontend output and SCL->AWL backend target)
// converges at std IR together with scl: !scl.direct operands lower to the
// runtime C ABI calls (ADR 002), accumulator/RLO ops to arith, jumps to cf
// blocks.

/// Convert an int/float value to `ty` (ext/trunc/sitofp/fptosi).
static Value convertScalarTo(OpBuilder &b, Location loc, Value v, Type ty,
                             bool zext) {
  Type src = v.getType();
  if (src == ty)
    return v;
  if (src.isa<FloatType>() && ty.isa<FloatType>()) {
    if (src.getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
      return b.create<arith::ExtFOp>(loc, ty, v);
    if (src.getIntOrFloatBitWidth() > ty.getIntOrFloatBitWidth())
      return b.create<arith::TruncFOp>(loc, ty, v);
    return v;
  }
  if (src.isa<FloatType>() && ty.isa<IntegerType>())
    return b.create<arith::FPToSIOp>(loc, ty, v);
  if (src.isa<IntegerType>() && ty.isa<FloatType>())
    return b.create<arith::SIToFPOp>(loc, ty, v);
  if (src.isa<IntegerType>() && ty.isa<IntegerType>()) {
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

struct AwlDirectOpLowering : public OpConversionPattern<awl::DirectOp> {
  using OpConversionPattern<awl::DirectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::DirectOp op, awl::DirectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    auto dt = op.getResult().getType().cast<scl::DirectType>();
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
        op.getValue().cast<TypedAttr>());
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
    auto tC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getTimer()));
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
    auto dt = op.getAddr().getType().template cast<scl::DirectType>();
    auto resTy = typeConverter->convertType(op.getType());
    if (!resTy)
      return failure();
    Value v = lowerDirectLoad(rewriter, op.getLoc(), dt, resTy,
                              op->template getParentOfType<ModuleOp>(), op);
    rewriter.replaceOp(op, v);
    return success();
  }
};

struct AwlStoreOpLowering : public OpConversionPattern<awl::StoreOp> {
  using OpConversionPattern<awl::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::StoreOp op, awl::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = op.getAddr().getType().template cast<scl::DirectType>();
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getValue(),
                                op->template getParentOfType<ModuleOp>(), op)))
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
    auto dt = op.getAddr().getType().template cast<scl::DirectType>();
    auto i1 = rewriter.getI1Type();
    Value bit = lowerDirectLoad(rewriter, op.getLoc(), dt, i1,
                                op->template getParentOfType<ModuleOp>());
    if (op.getNegate()) {
      auto trueC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i1, rewriter.getIntegerAttr(i1, 1));
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
    auto trueC = rewriter.create<arith::ConstantOp>(op.getLoc(), i1,
                                                    rewriter.getIntegerAttr(i1, 1));
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getRlo(), trueC);
    return success();
  }
};

struct AwlAssignOpLowering : public OpConversionPattern<awl::AssignOp> {
  using OpConversionPattern<awl::AssignOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(awl::AssignOp op, awl::AssignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dt = op.getAddr().getType().template cast<scl::DirectType>();
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, adaptor.getRlo(),
                                op->template getParentOfType<ModuleOp>(), op)))
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
    auto dt = op.getAddr().getType().template cast<scl::DirectType>();
    auto i1 = rewriter.getI1Type();
    auto v = rewriter.create<arith::ConstantOp>(
        op.getLoc(), i1, rewriter.getIntegerAttr(i1, set ? 1 : 0));
    if (failed(lowerDirectStore(rewriter, op.getLoc(), dt, v,
                                op->template getParentOfType<ModuleOp>(), op)))
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
    if (resTy.template isa<FloatType>()) {
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
        ModuleOp module = op->template getParentOfType<ModuleOp>();
        auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_arith_f",
                                    rewriter.getFunctionType({i8, f64, f64}, {f64}));
        Value opcC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i8, rewriter.getI8IntegerAttr(opc));
        Value res = rewriter.create<func::CallOp>(op.getLoc(), fn,
                                                  TypeRange{f64}, ValueRange{opcC, ld, rd})
                        .getResult(0);
        res = convertScalarTo(rewriter, op.getLoc(), res, resTy); // round to f32
        rewriter.replaceOp(op, res);
        return success();
      }
      rewriter.replaceOpWithNewOp<LoweredFloatOp>(op, resTy, l, r);
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
      return rewriter.create<arith::ConstantOp>(loc, i32,
                                                rewriter.getI32IntegerAttr(x));
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
      r = rewriter.create<arith::NegFOp>(loc, f32, v);
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
      auto mod = op->template getParentOfType<ModuleOp>();
      auto f64 = rewriter.getF64Type();
      auto i8 = rewriter.getI8Type();
      auto fn = getOrInsertRtFunc(rewriter, mod, "scl_rt_realmath_f",
                                  rewriter.getFunctionType({i8, f64}, {f64}));
      Value opcC = rewriter.create<arith::ConstantOp>(
          loc, i8, rewriter.getI8IntegerAttr(opc));
      Value d = rewriter.create<arith::ExtFOp>(loc, f64, v);
      Value rd = rewriter.create<func::CallOp>(loc, fn, TypeRange{f64},
                                               ValueRange{opcC, d}).getResult(0);
      r = rewriter.create<arith::TruncFOp>(loc, f32, rd);
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
    if (resTy.template isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, resTy, v);
    } else {
      auto zero = rewriter.create<arith::ConstantOp>(
          op.getLoc(), resTy, rewriter.getIntegerAttr(resTy, 0));
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
      Type ty = l.getType().template isa<FloatType>() || r.getType().template isa<FloatType>()
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
    if (l.getType().template isa<FloatType>())
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, predF, l, r);
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
      auto bitC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i8, rewriter.getI8IntegerAttr(4));
      auto zero = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i8, rewriter.getI8IntegerAttr(0));
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
        args.push_back(rewriter.create<arith::ConstantOp>(
            op.getLoc(), i32, rewriter.getI32IntegerAttr(dbAttr.value())));
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
      if (op.getCallee().startswith("SFB")) {
        sfbRt = true;
        sfbN = sfcAttr.value();
      }
    }
    // SFC n: run in the runtime (params staged in the marker area by the
    // frontend at kLocalBase)
    bool sfcRt = false;
    int32_t sfcN = 0;
    if (auto sfcAttr = op.getSfc()) {
      if (op.getCallee().startswith("SFC")) {
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
      auto arC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i8, rewriter.getI8IntegerAttr(2));
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
        auto arC2 = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i8, rewriter.getI8IntegerAttr(2));
        auto offC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i32, rewriter.getI32IntegerAttr(ar2off * 8));
        ptr = rewriter
                  .create<func::CallOp>(op.getLoc(), addFn, TypeRange{i64},
                                        ValueRange{arC2, offC})
                  .getResult(0);
      } else {
        // awlsim sets AR2 = AREA_DB_S | base offset (P#DBX 0.0 style, db
        // field 0 = current DI register at use time), not the DI area code
        auto areaC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i32, rewriter.getI32IntegerAttr(0x84));
        auto zero32 = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i32, rewriter.getI32IntegerAttr(0));
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
      auto nC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i32, rewriter.getI32IntegerAttr(sfbN));
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{nC});
      if (savedAr2) {
        auto i8 = rewriter.getI8Type();
        auto i64 = rewriter.getI64Type();
        auto setFn = getOrInsertRtFunc(
            rewriter, module, "scl_rt_ar_set",
            rewriter.getFunctionType({i8, i64}, {}));
        auto arC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i8, rewriter.getI8IntegerAttr(2));
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
      auto nC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i32, rewriter.getI32IntegerAttr(sfcN));
      rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange{},
                                    ValueRange{nC});
      rewriter.eraseOp(op);
      return success();
    }
    if (iecRt) {
      auto i32 = rewriter.getI32Type();
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_iec",
                                  rewriter.getFunctionType({i32}, {}));
      auto nC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i32, rewriter.getI32IntegerAttr(iecN));
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
        auto bitC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i8, rewriter.getI8IntegerAttr(bv.first));
        auto valC = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i8, rewriter.getI8IntegerAttr(bv.second));
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
      auto arC = rewriter.create<arith::ConstantOp>(
          op.getLoc(), i8, rewriter.getI8IntegerAttr(2));
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
    auto idC = rewriter.create<arith::ConstantOp>(
        op.getLoc(), i32, rewriter.getI32IntegerAttr(op.getId()));
    auto opC = rewriter.create<arith::ConstantOp>(
        op.getLoc(), i8, rewriter.getI8IntegerAttr(op.getOp()));
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
    auto idC = rewriter.create<arith::ConstantOp>(
        op.getLoc(), i32, rewriter.getI32IntegerAttr(op.getId()));
    auto opC = rewriter.create<arith::ConstantOp>(
        op.getLoc(), i8, rewriter.getI8IntegerAttr(op.getOp()));
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
    Value idx = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getIndex()));
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
    Value idx = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getIndex()));
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
    Value bitC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getBit()));
    Value shifted = rewriter.create<arith::ShRUIOp>(loc, word, bitC);
    Value one = rewriter.create<arith::ConstantOp>(loc, i32,
                                                   rewriter.getI32IntegerAttr(1));
    Value bit = rewriter.create<arith::AndIOp>(loc, shifted, one);
    // arith.cmpi returns i1 for any integer operands
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(0));
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
    Value bitC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getBit()));
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
      Value c = rewriter.create<arith::ConstantOp>(loc, i32,
                                                   rewriter.getI32IntegerAttr(bit));
      Value sh = rewriter.create<arith::ShRUIOp>(loc, word, c);
      Value one = rewriter.create<arith::ConstantOp>(
          loc, i32, rewriter.getI32IntegerAttr(1));
      return rewriter.create<arith::AndIOp>(loc, sh, one);
    };
    Value a0 = bitOf(6); // CC0: result < 0
    Value a1 = bitOf(7); // CC1: result > 0
    Value zero = rewriter.create<arith::ConstantOp>(loc, i32,
                                                    rewriter.getI32IntegerAttr(0));
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
    Value predC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getPred()));
    if (adaptor.getLhs().getType().isa<FloatType>()) {
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
    if (adaptor.getValue().getType().isa<FloatType>()) {
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
      Value opC = rewriter.create<arith::ConstantOp>(
          loc, i8, rewriter.getI8IntegerAttr(op.getOp()));
      Value rangeC = rewriter.create<arith::ConstantOp>(
          loc, i32, rewriter.getI32IntegerAttr(op.getRange()));
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
    if (adaptor.getResult().getType().isa<FloatType>()) {
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
      Value rng = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getRange()));
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
    Value modeC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getMode()));
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
      auto modeC = rewriter.create<arith::ConstantOp>(
          loc, i8, rewriter.getI8IntegerAttr(op.getMode()));
      Value durC = convertScalarTo(rewriter, loc, adaptor.getDur(), i64);
      Value inC = rewriter.create<arith::ConstantOp>(loc, i8,
                                                     rewriter.getI8IntegerAttr(1));
      if (adaptor.getIn())
        inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_start_ptr",
                                  rewriter.getFunctionType({i64, i8, i64, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getTimer(), modeC, durC, inC});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = op.getTimer().getType().dyn_cast<scl::DirectType>();
    if (!dt)
      return rewriter.notifyMatchFailure(op, "timer_start needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value timerC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(dt.getByte()));
    Value modeC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getMode()));
    Value dur = convertScalarTo(rewriter, loc, adaptor.getDur(), i64);
    Value inC = rewriter.create<arith::ConstantOp>(loc, i8,
                                                   rewriter.getI8IntegerAttr(1));
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
    auto dt = op.getTimer().getType().dyn_cast<scl::DirectType>();
    if (!dt)
      return rewriter.notifyMatchFailure(op, "timer_reset needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32 = rewriter.getI32Type();
    Value timerC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(dt.getByte()));
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
      Value inC = rewriter.create<arith::ConstantOp>(loc, i8,
                                                     rewriter.getI8IntegerAttr(1));
      if (adaptor.getIn())
        inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_fr_ptr",
                                  rewriter.getFunctionType({rewriter.getI64Type(), i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{},
                                    ValueRange{adaptor.getOperand(), inC});
      rewriter.eraseOp(op);
      return success();
    }
    auto dt = op.getOperand().getType().dyn_cast<scl::DirectType>();
    if (!dt)
      return rewriter.notifyMatchFailure(op, "fr needs a direct operand");
    auto loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    Value idxC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(dt.getByte()));
    Value inC = rewriter.create<arith::ConstantOp>(loc, i8,
                                                   rewriter.getI8IntegerAttr(1));
    if (adaptor.getIn())
      inC = rewriter.create<arith::ExtUIOp>(loc, i8, adaptor.getIn());
    if (dt.getArea() == scl::DirectArea::Timer) {
      auto fn = getOrInsertRtFunc(rewriter, module, "scl_rt_timer_fr",
                                  rewriter.getFunctionType({i32, i8}, {}));
      rewriter.create<func::CallOp>(loc, fn, TypeRange{}, ValueRange{idxC, inC});
    } else if (dt.getArea() == scl::DirectArea::Counter) {
      Value opC = rewriter.create<arith::ConstantOp>(
          loc, i8, rewriter.getI8IntegerAttr(4 /* SCL_RT_COUNTER_FR */));
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, i64, rewriter.getI64IntegerAttr(0));
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
    Value msC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getMs()));
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
    auto areaC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getArea()));
    auto byteC = convertScalarTo(rewriter, loc, adaptor.getByte(), i32);
    auto bitC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getBit()));
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
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto i8 = rewriter.getI8Type();
    auto i32 = rewriter.getI32Type();
    auto i64 = rewriter.getI64Type();
    auto opC0 = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getOp()));
    Value val0 = convertScalarTo(rewriter, loc, adaptor.getValue(), i64);
    Value inC0 = rewriter.create<arith::ConstantOp>(loc, i8,
                                                    rewriter.getI8IntegerAttr(0));
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
    auto dt = op.getCounter().getType().dyn_cast<scl::DirectType>();
    if (!dt)
      return rewriter.notifyMatchFailure(op, "counter needs a direct operand");
    Value cntC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(dt.getByte()));
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
    Value dbC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getDb()));
    Value szC = rewriter.create<arith::ConstantOp>(
        loc, i64, rewriter.getI64IntegerAttr(op.getBytes()));
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
    Value dbC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getDb()));
    Value szC = rewriter.create<arith::ConstantOp>(
        loc, i64, rewriter.getI64IntegerAttr(op.getBytes()));
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
    Value tC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getTarget()));
    Value vC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getValue()));
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
    Value kindC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getKind()));
    Value dbC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getDb()));
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
    Value dbC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getDb()));
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
    Value dbC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getDb()));
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
      return rewriter.create<arith::ConstantOp>(
          loc, i32, rewriter.getI32IntegerAttr(v));
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
    Value arC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getAr()));
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
    Value arC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getAr()));
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
    Value arC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getAr()));
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
    Value arC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getAr()));
    Value offC = rewriter.create<arith::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(op.getOffBits()));
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
    Value arC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getAr()));
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
    Value widthC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getWidthBits() / 8));
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
    Value widthC = rewriter.create<arith::ConstantOp>(
        loc, i8, rewriter.getI8IntegerAttr(op.getWidthBits() / 8));
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
    bool isFb = op.getSymName().startswith("FB") || op.getFb().has_value();
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
          auto one = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI1Type(), rewriter.getIntegerAttr(
                                             rewriter.getI1Type(), 1));
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
      auto frameC = rewriter.create<arith::ConstantOp>(
          loc, i32, rewriter.getI32IntegerAttr(frameSize));
      auto pushCall =
          rewriter.create<func::CallOp>(loc, pushFn, TypeRange{}, ValueRange{frameC});
      // awlsim adds the AR2 base to DI-area accesses only inside FBs
      {
        auto i8 = rewriter.getI8Type();
        auto fbFn = getOrInsertRtFunc(rewriter, module, "scl_rt_frame_set_fb",
                                      rewriter.getFunctionType({i8}, {}));
        auto fbC = rewriter.create<arith::ConstantOp>(
            loc, i8, rewriter.getI8IntegerAttr(op.getFb() ? 1 : 0));
        rewriter.create<func::CallOp>(loc, fbFn, TypeRange{}, ValueRange{fbC});
      }
      // OB entry: write the S7 OB-temp preset header into the new frame
      if (op.getSymName().startswith("OB")) {
        int32_t obNum = atoi(op.getSymName().substr(2).str().c_str());
        auto obFn = getOrInsertRtFunc(rewriter, module, "scl_rt_ob_start",
                                      rewriter.getFunctionType({i32}, {}));
        rewriter.setInsertionPointAfter(pushCall);
        auto obC = rewriter.create<arith::ConstantOp>(
            loc, i32, rewriter.getI32IntegerAttr(obNum));
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
        auto pC = rewriter.create<arith::ConstantOp>(
            loc, i32, rewriter.getI32IntegerAttr(paramBytes));
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
              auto pC = rewriter.create<arith::ConstantOp>(
                  loc, i32, rewriter.getI32IntegerAttr(paramBytes));
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
  target.addLegalOp<scl::DebugPrintOp>();

  // We want to lower all of SCL except for control flow:
  target.addIllegalDialect<scl::SclDialect>();
  // ... and all of AWL (parallel source dialect, ADR 003):
  target.addIllegalDialect<awl::AwlDialect>();

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
  // AWL dialect (ADR 003)
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
