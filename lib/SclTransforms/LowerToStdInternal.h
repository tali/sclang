//====- LowerToStdInternal.h - Shared internals of the LowerToStd pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header shared by the LowerToStd translation units
// (LowerToStd.cpp / LowerToStdUtil.cpp / LowerToStdScl.cpp /
// LowerToStdAwl.cpp). Not installed; do not include from outside
// lib/SclTransforms. The split is pure code motion: one pass, same
// patterns, same behavior — see AGENTS.md.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_SCLTRANSFORMS_LOWERTSTD_INTERNAL_H
#define SCLANG_SCLTRANSFORMS_LOWERTSTD_INTERNAL_H

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
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Internal header: keep the same using-declarations as the .cpp files so
// the moved code compiles unchanged.
using namespace mlir;
using llvm::all_of;
using namespace mlir::arith;
using namespace mlir::func;

namespace mlir::sclang::lowering {

// MARK: shared types (needed by the declarations below)


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

/// Maps each continue/exit op to the (continue, exit) blocks of the
/// innermost enclosing loop. Filled by the loop lowerings, which are applied
/// before the nested ops (dialect conversion is pre-order), and consumed by
/// ContinueOpLowering/ExitOpLowering when the nested op is converted.
struct LoopTerminatorTargets {
  DenseMap<Operation *, std::pair<Block *, Block *>> map;
};
using LoopTargetsPtr = std::shared_ptr<LoopTerminatorTargets>;

// MARK: shared helper declarations (defined in LowerToStdUtil.cpp)
/// Convert an int/float value to `ty` (ext/trunc/sitofp/fptosi).
Value convertScalarTo(OpBuilder &b, Location loc, Value v, Type ty,
                      bool zext = false);


/// Size of a type in bytes.
int64_t byteSize(Type type, const FbLayoutCache &layouts);


/// Fold a constant expression (literals + integer arithmetic) to a bit
/// pattern, used for FB member initializers (cold-start values baked into
/// the instance DB global).
std::optional<llvm::APInt> foldConstBits(Value v, unsigned width);


FbLayoutCache computeFbLayouts(ModuleOp module);


/// Fixed layouts of the builtin standard FBs (lowered to runtime calls on
/// the instance DB buffer, not func.call), forced even if user-declared (or
/// stub-declared by the frontend), so the runtime's interpretation matches
/// (see Runtime.h scl_rt_*_fb). Bool members occupy a full byte; ints are
/// big-endian; timers use IN @0, PT @4 (i32 ms), Q @8, ET @12 (i32 ms).
void forceBuiltinFbLayouts(ModuleOp module, FbLayoutCache &layouts);


/// Lower a load from a direct address (memory area / DB operand) to a call
/// of the runtime C ABI (ADR 002): `scl_rt_read_bit` for bit access, else
/// `scl_rt_read` with the width in bytes.
Value lowerDirectLoad(OpBuilder &b, Location loc, scl::DirectType dt,
                             Type resTy, ModuleOp module, Operation *op = nullptr);


/// Lower a store to a direct address (memory area / DB operand) to a call of
/// the runtime C ABI (ADR 002): `scl_rt_write_bit` for bit access, else
/// `scl_rt_write` with the value zero/sign-extended to i64 and the width in
/// bytes.
LogicalResult lowerDirectStore(OpBuilder &b, Location loc,
                                      scl::DirectType dt, Value value,
                                      ModuleOp module, Operation *op = nullptr);


// MARK: LoadOpLowering

/// Load a value of `resTy` from `addr`; byte-assemble if the address is an
/// i8 view (FB/struct members in a byte buffer).
Value loadFromByteBuffer(OpBuilder &b, Location loc, Value addr,
                                Type resTy);


// MARK: StoreOpLowering

/// Store `value` at `addr`, splitting into bytes if the address is an i8 view.
LogicalResult storeToByteBuffer(OpBuilder &b, Location loc, Value value,
                                       Value addr);


/// Walk a chain of memref.reinterpret_cast byte views down to the root
/// buffer, accumulating the offset (constant part into `constOff`, dynamic
/// part into `dynOff`). Returns the root value, or null if the chain bottoms
/// out at something that is not a memref. Used to create a *single* cast from
/// the root: chained reinterpret_casts would lose the inner offset when
/// ReinterpretCastOp::fold fires.
Value resolveByteViewRoot(Value v, Value &dynOff, int64_t &constOff,
                                 OpBuilder &b, Location loc);


/// Combine a constant offset, a dynamic offset value and an extra dynamic
/// offset into a single index value.
Value combineOffsets(OpBuilder &b, Location loc, int64_t constOff,
                            Value dynOff, Value extra);


/// Create a rank-1 i8 byte view of `size` bytes at `off` from a single cast
/// of the root buffer.
Value makeByteView(OpBuilder &b, Location loc, Value root, Value off,
                          int64_t size, MLIRContext *ctx,
                          bool staticSize = false);


/// Create a rank-1 i8 byte view at byte offset `memberOff` within `src`,
/// folding in the runtime offset of `src` itself (e.g. an FB instance-DB
/// parameter that is a view with a nonzero offset). reinterpret_cast offsets
/// are absolute w.r.t. the base buffer, so resolve the root of any cast chain,
/// take the root's metadata (base buffer + its own offset), and emit a single
/// cast from the base buffer with the total offset.
Value makeByteViewFromSource(OpBuilder &b, Location loc, Value src,
                                    int64_t memberOff, int64_t size,
                                    MLIRContext *ctx, bool staticSize = false);


// MARK: region moving helpers

/// Move all blocks of `src` into `dst` before `before`; returns the moved
/// entry block. After the move, `src` is empty.
Block *moveRegion(ConversionPatternRewriter &rewriter, Region &src,
                         Region &dst, Block *before);


/// Replace `scl.end` terminators in `blocks` with a branch to contB. Blocks
/// left without a terminator get a branch to contB. continue/exit are NOT
/// handled here: they are lowered by their own patterns to the innermost
/// enclosing loop's targets (recorded by the loop lowerings).
void lowerTerminatorsInBlocks(ConversionPatternRewriter &rewriter,
                                     Location loc, ArrayRef<Block *> blocks,
                                     Block *contB, Block *exitB);


/// Record continue/exit ops in `blocks` and in regions nested inside non-loop
/// ops (if/case bodies), so their lowerings can branch to the loop targets.
/// Nested loop ops are skipped: they record their own bodies.
void recordLoopTerminators(Operation *op, Block *contB, Block *exitB,
                                  LoopTerminatorTargets &targets);


void recordLoopTerminators(ArrayRef<Block *> blocks, Block *contB,
                                  Block *exitB, LoopTerminatorTargets &targets);


/// Collect the blocks strictly between `from` (exclusive) and `to`
/// (exclusive).
SmallVector<Block *> blocksBetween(Block *from, Block *to);


/// Collect the blocks from `from` (inclusive) up to `to` (exclusive).
SmallVector<Block *> blocksFrom(Block *from, Block *to);



// MARK: AWL dialect lowering (ADR 003)
//
// The awl dialect (AWL/STL frontend output and SCL->AWL backend target)
// converges at std IR together with scl: !scl.direct operands lower to the
// runtime C ABI calls (ADR 002), accumulator/RLO ops to arith, jumps to cf
// blocks.

/// Convert an int/float value to `ty` (ext/trunc/sitofp/fptosi).
Value convertScalarTo(OpBuilder &b, Location loc, Value v, Type ty,
                             bool zext);

// MARK: inline helpers and the type converter


/// Return `true` if all elements are of the given type.
template <typename U> bool all_of_type(ValueRange range) {
  return all_of(range, [](Value elem) { return llvm::isa<U>(elem.getType()); });
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
      if (llvm::isa<scl::DirectType>(elem))
        return (Type)IntegerType::get(type.getContext(), 64);
      // An instance DB address is a rank-1 i8 view of the instance buffer
      // with a dynamic offset (like any other byte view). Keeping the view
      // type here means FB calls on local instances (FB-typed members, which
      // are byte views into the parent IDB) pass the view directly — no cast,
      // and no reinterpret_cast chaining (whose fold would drop the inner
      // offset). Global instance DBs are get_global'ed as plain memrefs and
      // memref.cast to this view type in GetGlobalOpLowering.
      if (auto idbTy = llvm::dyn_cast<scl::InstanceDbType>(elem)) {
        auto it = fbLayouts.find(idbTy.getFbSymbol());
        if (it == fbLayouts.end())
          return Type();
        return (Type)mlir::MemRefType::get(
            {ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8),
            StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
                                   {ShapedType::kDynamic}));
      }
      auto celem = convertType(elem);
      if (!celem)
        return Type();
      // An address of a struct is a rank-1 i8 window of the struct buffer
      // (static size, dynamic offset), so that getelement/getindex on
      // struct array elements chain through resolveByteViewRoot with a
      // dynamic offset.
      if (auto st = llvm::dyn_cast<scl::StructType>(elem))
        return (Type)mlir::MemRefType::get(
            {byteSize(st, fbLayouts)}, IntegerType::get(type.getContext(), 8),
            StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
                                   {1}));
      // An address of a composite (array/IDB) IS the buffer memref.
      if (llvm::isa<MemRefType>(celem))
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
                                   {ShapedType::kDynamic}));
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
      return Float32Type::get(type.getContext());
    });
    addConversion([&](scl::LrealType type) {
      return Float64Type::get(type.getContext());
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
      if (llvm::isa<MemRefType>(elem))
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
                                 Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return builder.create<scl::DialectCastOp>(loc, resultType, inputs[0])
          .getResult();
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return builder.create<scl::DialectCastOp>(loc, resultType, inputs[0])
          .getResult();
    });
  }
};

/// The element type of an scl address value (`!scl.address<T>` -> T).
inline Type addrElemTy(Value addr) {
  return llvm::cast<scl::AddressType>(addr.getType()).getElementType();
}

/// Shorthand for an integer constant of type `ty` with value `v`.
inline Value constInt(OpBuilder &b, Location loc, Type ty, int64_t v) {
  return b.create<arith::ConstantOp>(loc, ty, b.getIntegerAttr(ty, v));
}

/// Shorthand for an index-typed constant with value `v`.
inline Value constIndex(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

/// The enclosing module of an operation.
inline ModuleOp moduleOf(Operation *op) {
  return op->getParentOfType<ModuleOp>();
}

// MARK: runtime C ABI (ADR 002)

/// Get or insert a private declaration of a runtime ABI function (ADR 002)
/// into the module, returning its symbol reference.
inline FlatSymbolRefAttr getOrInsertRtFunc(OpBuilder &b, ModuleOp module,
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
inline Value directDbValue(OpBuilder &b, Location loc, scl::DirectType dt,
                           ModuleOp module, Operation *op) {
  auto i32 = b.getI32Type();
  if (dt.getArea() == scl::DirectArea::DataBlock && dt.getDb() == -1) {
    // DI-area operand: pass the DI marker 0xFFFF through, so the runtime
    // resolves it to the DI-register DB *and* adds the AR2 multi-instance
    // base offset (awlsim cpu.py getArea_DI/setArea_DI)
    return constInt(b, loc, i32, (int32_t)0xFFFF);
  }
  if (op && dt.getArea() == scl::DirectArea::DataBlock && dt.getDb() == 0) {
    if (auto func = op->template getParentOfType<func::FuncOp>()) {
      if (func.getSymName().starts_with("FB") && func.getNumArguments() == 1 &&
          func.getBody().front().getNumArguments() == 1)
        return func.getBody().front().getArgument(0);
    }
  }
  return constInt(b, loc, i32, dt.getDb());
}

/// Offset OpFoldResult for a reinterpret_cast whose result type is `resTy`:
/// when the result layout offset is static, the op's static_offsets attribute
/// must equal it (MLIR verifier), so use a static attribute; otherwise a
/// dynamic offset operand is fine.
inline OpFoldResult rcOffsetOfr(OpBuilder &b, MemRefType resTy, Value dynOff) {
  int64_t off = 0;
  SmallVector<int64_t, 4> strides;
  (void)resTy.getStridesAndOffset(strides, off);
  if (ShapedType::isDynamic(off))
    return OpFoldResult(dynOff);
  return OpFoldResult(b.getIndexAttr(off));
}

// MARK: pattern-set population (defined in LowerToStdScl/LowerToStdAwl)

/// Add all scl-dialect lowering patterns (scalar/memory ops, control
/// flow, blocks, calls).
void populateSclToStdPatterns(RewritePatternSet &patterns,
                              SclTypeConverter &converter, MLIRContext *context,
                              LoopTargetsPtr loopTargets);
/// Add all awl-dialect lowering patterns (ADR 003).
void populateAwlToStdPatterns(RewritePatternSet &patterns,
                              SclTypeConverter &converter, MLIRContext *context);

} // namespace mlir::sclang::lowering

#endif // SCLANG_SCLTRANSFORMS_LOWERTSTD_INTERNAL_H
