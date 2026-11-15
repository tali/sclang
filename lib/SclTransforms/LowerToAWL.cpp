//===- LowerToAWL.cpp - Lower SCL to the AWL dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SCL -> AWL backend (docs/design/scl-to-awl.md, ADR 003): converts each
// top-level `scl.function`/`scl.function_block` whose name is OB/FB/FC n
// into an `awl.block` with the same name. FC/OB-local variables are
// allocated in the M area (bit-packed for bools), `%`-direct addresses stay
// direct, expressions are emitted in accumulator order (operands
// materialized recursively right before use), and structured control flow
// (if/while/repeat/for/case, goto blocks) becomes awl labels + jumps.
// FB instance data (getvariable), arrays/structs (getelement/getindex) and
// FB calls are rejected with a clear error (known gaps).
//
// The resulting module is printed as S7 AWL text by awl::printAwlText
// (`sclang-gen --emit=awl`).
//
//===----------------------------------------------------------------------===//

#include "sclang/SclTransforms/Passes.h"

#include "sclang/AwlDialect/AwlOps.h"
#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclDialect/SclTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

using namespace mlir;

namespace mlir {
namespace sclang {
namespace {

// FB member layout for the AWL backend: member name -> byte offset, width.
struct AwlFbMember {
  int64_t offset;
  int width;
  Type type; // declared scl type (for struct field / array element offsets)
};
using AwlFbLayout = llvm::StringMap<AwlFbMember>;

// Maps an scl type to the value type used in the awl dialect.
static Type awlValueType(MLIRContext *ctx, Type t) {
  if (isa<scl::LogicalType>(t))
    return IntegerType::get(ctx, 1);
  if (auto it = dyn_cast<scl::IntegerType>(t))
    return IntegerType::get(ctx, it.getWidth());
  if (isa<scl::RealType>(t))
    return FloatType::getF32(ctx);
  return nullptr;
}

// The awl value type of a direct operand's load/store.
static Type awlDirectValueType(MLIRContext *ctx, scl::DirectType dt) {
  return awlValueType(ctx, dt.getValueType());
}

// Storage width in bits of an scl type (for M-area allocation).
static int widthOf(Type t) {
  if (auto lt = dyn_cast<scl::LogicalType>(t))
    return lt.getWidth(); // bool = 1, byte = 8
  if (auto it = dyn_cast<scl::IntegerType>(t))
    return it.getWidth();
  if (isa<scl::RealType>(t))
    return 32;
  // composites occupy their flat byte size
  if (auto at = dyn_cast<scl::ArrayType>(t)) {
    int64_t count = 1;
    for (auto &dim : at.getDimensions())
      count *= dim.second - dim.first + 1;
    int ew = widthOf(at.getElementType());
    return ew > 0 ? (int)(count * ew) : -1;
  }
  if (auto st = dyn_cast<scl::StructType>(t)) {
    int total = 0;
    for (auto f : st.getElementTypes()) {
      int w = widthOf(f);
      if (w <= 0)
        return -1;
      total += w;
    }
    return total;
  }
  return -1;
}

struct FuncConverter {
  MLIRContext *ctx;
  ModuleOp module;
  Location loc;
  OpBuilder b;
  awl::BlockOp blockOp;
  Block *flat;

  DenseMap<Value, scl::DirectType> varDirect; // scl address -> direct type
  // FC parameters are passed through a reserved window at the top of the M
  // area (the AWL analog of the S7 L stack): the caller stores each argument
  // to its slot before `UC FC n`, the FC reads its inputs from there.
  static constexpr int32_t kParamBaseByte = 65280; // 8 bytes per param
  static int32_t paramSlot(int i) { return kParamBaseByte + i * 8; }
  // the FC return value is passed through this slot (just below the window)
  static constexpr int32_t kResultByte = kParamBaseByte - 8;
  DenseMap<Value, scl::DirectType> paramDirects; // FC input tempvar -> slot
  DenseMap<llvm::StringRef, int32_t> dbNumbers; // IDB symbol name -> DB number
  const llvm::StringMap<AwlFbLayout> *fbLayouts = nullptr;
  Value currentFbArg; // block arg holding the instance DB of the current FB
  DenseMap<Value, Value> materialized;        // scl value -> awl value
  // FC name (uppercased) -> input parameter names in declaration order,
  // collected before any function is erased
  llvm::StringMap<SmallVector<std::string, 4>> fcSignatures;
  int32_t nextBit = 0;                         // M-area allocation cursor
  int labelCounter = 0;
  struct LoopCtx {
    std::string cont; // continue target
    std::string end;  // exit target
  };
  SmallVector<LoopCtx, 4> loops;

  FuncConverter(MLIRContext *ctx, ModuleOp module, Location loc)
      : ctx(ctx), module(module), loc(loc), b(ctx) {}

  std::string newLabel(const char *prefix) {
    return std::string(prefix) + std::to_string(labelCounter++);
  }

  StringAttr strAttr(llvm::StringRef s) {
    return StringAttr::get(ctx, s);
  }

  // Allocate an M-area slot for a variable of the given element type.
  scl::DirectType allocDirect(Type elemType) {
    int w = widthOf(elemType);
    if (w <= 0) {
      emitError(loc) << "type not representable in AWL output: " << elemType;
      return nullptr;
    }
    int32_t byte, bit;
    if (w == 1) {
      byte = nextBit / 8;
      bit = nextBit % 8;
      ++nextBit;
    } else {
      nextBit = (nextBit + w - 1) / w * w; // align
      byte = nextBit / 8;
      bit = 0;
      nextBit += w;
    }
    return scl::DirectType::get(ctx, scl::DirectArea::Marker, 0, byte, bit, w);
  }

  // The declared scl type behind an address value (FB member, local
  // tempvar, array element), used for struct field offsets.
  Type declaredTypeOf(Value addr) {
    if (auto gv = addr.getDefiningOp<scl::GetVariableOp>()) {
      auto baseTy = gv.getAddr().getType().dyn_cast<scl::AddressType>();
      if (baseTy && baseTy.getElementType().isa<scl::InstanceDbType>() &&
          fbLayouts) {
        auto idbTy = baseTy.getElementType().cast<scl::InstanceDbType>();
        std::string fbSym = llvm::StringRef(idbTy.getFbSymbol()).upper();
        auto lit = fbLayouts->find(fbSym);
        if (lit != fbLayouts->end()) {
          auto mit = lit->second.find(gv.getSymName().str());
          if (mit != lit->second.end())
            return mit->second.type;
        }
      }
    }
    if (auto gi = addr.getDefiningOp<scl::GetIndexOp>()) {
      if (auto at = gi.getArrayType())
        if (auto atTy = at->dyn_cast<scl::ArrayType>())
          return atTy.getElementType();
    }
    if (auto ge = addr.getDefiningOp<scl::GetElementOp>()) {
      if (auto bt = ge.getBaseType())
        return *bt;
    }
    return addr.getType().cast<scl::AddressType>().getElementType();
  }

  // Resolve an scl address value to a direct type (M-area var or % direct).
  scl::DirectType directOf(Value addr) {
    auto it = varDirect.find(addr);
    if (it != varDirect.end())
      return it->second;
    if (auto gd = addr.getDefiningOp<scl::GetDirectOp>()) {
      return addr.getType().cast<scl::AddressType>()
          .getElementType()
          .cast<scl::DirectType>();
    }
    // getvariable from an IDB: FB member at its byte offset in the instance
    // DB (db 0 = instance DB of the enclosing FB, else the called DB number)
    if (auto gv = addr.getDefiningOp<scl::GetVariableOp>()) {
      Value base = gv.getAddr();
      auto baseTy = base.getType().dyn_cast<scl::AddressType>();
      if (baseTy &&
          baseTy.getElementType().isa<scl::InstanceDbType>() && fbLayouts) {
        auto idbTy = baseTy.getElementType().cast<scl::InstanceDbType>();
        std::string fbSymStr = llvm::StringRef(idbTy.getFbSymbol()).upper();
        auto lit = fbLayouts->find(fbSymStr);
        if (lit != fbLayouts->end()) {
          int32_t db = 0;
          if (base == currentFbArg) {
            db = 0; // instance DB of the enclosing FB
          } else if (auto gg = base.getDefiningOp<scl::GetGlobalOp>()) {
            auto dit = dbNumbers.find(gg.getName());
            if (dit != dbNumbers.end())
              db = dit->second;
          }
          std::string member = gv.getSymName().str();
          auto mit = lit->second.find(member);
          if (mit != lit->second.end()) {
            int w = mit->second.width;
            if (w == 1 || w == 8 || w == 16 || w == 32)
              return scl::DirectType::get(ctx, scl::DirectArea::DataBlock, db,
                                           (int32_t)mit->second.offset, 0, w);
            // composite member (array/struct/string): expose its base as a
            // byte direct; element access goes through GetIndexOp/GetElementOp
            return scl::DirectType::get(ctx, scl::DirectArea::DataBlock, db,
                                         (int32_t)mit->second.offset, 0, 8);
          }
        }
      }
      emitError(addr.getLoc()) << "unknown FB member '"
                                << gv.getSymName().str() << "'";
      return nullptr;
    }
    // getindex with a constant index: fold index * element size into the
    // base direct's byte offset (AWL has no indirect addressing, so dynamic
    // indices are not representable).
    if (auto gi = addr.getDefiningOp<scl::GetIndexOp>()) {
      if (gi.getIndices().size() != 1) {
        emitError(addr.getLoc()) << "multi-dimensional arrays are not "
                                   "representable in AWL output";
        return nullptr;
      }
      scl::DirectType base = directOf(gi.getAddr());
      if (!base)
        return nullptr;
      int64_t idx = 0;
      if (auto c = gi.getIndices()[0].getDefiningOp<scl::ConstantOp>()) {
        if (auto ia = c.getValue().dyn_cast<IntegerAttr>())
          idx = ia.getValue().getSExtValue();
      } else {
        emitError(addr.getLoc()) << "dynamic array index is not representable "
                                   "in AWL output";
        return nullptr;
      }
      // Element size/width: from the declared array type if tracked (byte-view
      // path: result element is i8, so use the array's element type), else
      // from the result element type (plain array variable). Composite
      // elements (structs) stay byte views (width 8) but advance by the full
      // element size.
      int elemBits = widthOf(addr.getType().cast<scl::AddressType>()
                                 .getElementType());
      int64_t elemBytes = elemBits > 0 ? (elemBits + 7) / 8 : 1;
      if (auto at = gi.getArrayType()) {
        if (auto atTy = at->dyn_cast<scl::ArrayType>()) {
          int ew = widthOf(atTy.getElementType());
          if (ew > 0) {
            elemBytes = (ew + 7) / 8;
            elemBits = ew > 32 ? 8 : ew;
          }
        }
      }
      int w = elemBits > 0 ? elemBits : 8;
      return scl::DirectType::get(ctx, base.getArea(), base.getDb(),
                                   base.getByte() + (int32_t)(idx * elemBytes),
                                   0, w);
    }
    // getelement: struct field at its byte offset within the base
    if (auto ge = addr.getDefiningOp<scl::GetElementOp>()) {
      scl::DirectType base = directOf(ge.getAddr());
      if (!base)
        return nullptr;
      Type baseTy = declaredTypeOf(ge.getAddr());
      auto st = baseTy.dyn_cast<scl::StructType>();
      if (!st) {
        emitError(addr.getLoc()) << "struct element of non-struct base";
        return nullptr;
      }
      int64_t fieldOff = 0;
      int fw = 8;
      bool found = false;
      for (auto [fname, fty] : llvm::zip(st.getElementNames(), st.getElementTypes())) {
        if (fname.str() == ge.getName().str()) {
          fw = widthOf(fty);
          found = true;
          break;
        }
        int w = widthOf(fty);
        if (w <= 0)
          break;
        fieldOff += (w + 7) / 8;
      }
      if (!found || fw <= 0) {
        emitError(addr.getLoc()) << "unknown struct field '" << ge.getName()
                                  << "'";
        return nullptr;
      }
      return scl::DirectType::get(ctx, base.getArea(), base.getDb(),
                                   base.getByte() + (int32_t)fieldOff, 0, fw);
    }
    emitError(addr.getLoc())
        << "address not representable as an AWL operand (only M-area vars, % "
           "directs, FB/struct members and constant-indexed elements are "
           "supported by the AWL backend)";
    return nullptr;
  }

  Value directValue(scl::DirectType dt) {
    return b.create<awl::DirectOp>(loc, dt.getArea(), dt.getDb(), dt.getByte(),
                                   dt.getBit(), dt.getWidth());
  }

  Value castTo(Value v, Type t) {
    if (!v || v.getType() == t)
      return v;
    return b.create<awl::CastOp>(loc, t, v);
  }

  Value constI(llvm::APInt v, Type t) {
    return b.create<awl::ConstOp>(loc, t, IntegerAttr::get(t, v));
  }

  // Materialize an scl value as an awl value, emitting producer ops at the
  // current insertion point (operands recursively first).
  /// Emit `UC FC n`: store the arguments into the callee's parameter window
  /// slots, then the call. If `wantResult`, load the FC's return value from
  /// the result slot afterwards and return it.
  Value emitCallFc(scl::CallFcOp call, bool wantResult) {
    std::string calleeStr = llvm::StringRef(call.getCallee()).upper();
    llvm::StringRef callee = calleeStr;
    if (!callee.startswith("FC")) {
      emitError(call.getLoc()) << "AWL backend only supports CALL FC n, got @"
                               << callee;
      return nullptr;
    }
    std::string calleeUp = llvm::StringRef(call.getCallee()).upper();
    auto sigIt = fcSignatures.find(calleeUp);
    if (sigIt != fcSignatures.end() && !call.getArguments().empty()) {
      llvm::StringMap<unsigned> paramIndex;
      for (auto [i, pname] : llvm::enumerate(sigIt->second))
        paramIndex[pname] = (unsigned)i;
      auto argNames = call.getArgNames();
      for (auto [argVal, nameAttr] : llvm::zip(call.getArguments(), argNames)) {
        unsigned idx = 0;
        if (auto sa = dyn_cast<StringAttr>(nameAttr)) {
          auto it = paramIndex.find(sa.getValue());
          if (it != paramIndex.end())
            idx = it->second;
        }
        // String/array argument: copy the buffer byte-wise into the
        // parameter window slot (AWL has no whole-buffer transfer).
        if (auto at = argVal.getType().dyn_cast<scl::ArrayType>()) {
          auto ld = argVal.getDefiningOp<scl::LoadOp>();
          if (!ld || widthOf(at.getElementType()) != 8) {
            emitError(call.getLoc()) << "array argument type not representable "
                                        "in AWL (only strings/byte arrays)";
            return nullptr;
          }
          scl::DirectType src = directOf(ld.getAddress());
          if (!src)
            return nullptr;
          int64_t count = 1;
          for (auto &dim : at.getDimensions())
            count *= dim.second - dim.first + 1;
          auto i8 = IntegerType::get(ctx, 8);
          for (int64_t i = 0; i < count; ++i) {
            auto srcDt = scl::DirectType::get(ctx, src.getArea(), src.getDb(),
                                               src.getByte() + (int32_t)i, 0, 8);
            auto dstDt = scl::DirectType::get(ctx, scl::DirectArea::Marker, 0,
                                               paramSlot((int)idx) + (int32_t)i,
                                               0, 8);
            auto v = b.create<awl::LoadOp>(loc, i8, directValue(srcDt));
            b.create<awl::StoreOp>(loc, v, directValue(dstDt));
          }
          continue;
        }
        Type vt = awlValueType(ctx, argVal.getType());
        if (!vt) {
          emitError(call.getLoc()) << "argument type not representable in AWL";
          return nullptr;
        }
        Value v = castTo(materialize(argVal), vt);
        if (!v)
          return nullptr;
        scl::DirectType dt =
            scl::DirectType::get(ctx, scl::DirectArea::Marker, 0,
                                 paramSlot((int)idx), 0,
                                 vt.getIntOrFloatBitWidth());
        b.create<awl::StoreOp>(loc, v, directValue(dt));
      }
    }
    SmallVector<NamedAttribute, 1> attrs;
    attrs.push_back(
        NamedAttribute(strAttr("callee"), FlatSymbolRefAttr::get(ctx, callee)));
    b.create<awl::CallOp>(loc, TypeRange{}, ValueRange{}, attrs);
    if (wantResult) {
      Type rt = awlValueType(ctx, call.getResult().getType());
      scl::DirectType dt = scl::DirectType::get(
          ctx, scl::DirectArea::Marker, 0, kResultByte, 0,
          rt.getIntOrFloatBitWidth());
      return b.create<awl::LoadOp>(loc, rt, directValue(dt));
    }
    return nullptr;
  }

  Value materialize(Value v) {
    auto it = materialized.find(v);
    if (it != materialized.end())
      return it->second;
    Operation *def = v.getDefiningOp();
    if (!def) {
      emitError(v.getLoc()) << "block arguments are not supported by the AWL "
                               "backend";
      return nullptr;
    }
    Value r = nullptr;
    Type rt = awlValueType(ctx, v.getType());
    // S7 AWL has no byte arithmetic/comparison mnemonic: promote char (i8)
    // values to int (i16), like the S7 compiler does.
    if (rt && rt.isInteger(8))
      rt = IntegerType::get(ctx, 16);
    if (!rt) {
      emitError(def->getLoc()) << "type not representable in AWL output: "
                               << v.getType();
      return nullptr;
    }
    if (auto call = dyn_cast<scl::CallFcOp>(def)) {
      r = emitCallFc(call, true);
    } else if (auto c = dyn_cast<scl::ConstantOp>(def)) {
      Attribute attr = c.getValue();
      if (isa<StringAttr>(attr)) {
        emitError(def->getLoc()) << "string constants are not supported by "
                                    "the AWL backend";
        return nullptr;
      }
      r = b.create<awl::ConstOp>(loc, rt, attr);
    } else if (auto l = dyn_cast<scl::LoadOp>(def)) {
      scl::DirectType dt = directOf(l.getAddress());
      if (!dt)
        return nullptr;
      Value loaded = b.create<awl::LoadOp>(loc, awlDirectValueType(ctx, dt),
                                           directValue(dt));
      r = castTo(loaded, rt);
    } else if (isa<scl::AddOp, scl::SubOp, scl::MulOp, scl::DivOp,
                   scl::AndOp, scl::OrOp, scl::XOrOp>(def)) {
      Value a = materialize(def->getOperand(0));
      Value b2 = materialize(def->getOperand(1));
      if (!a || !b2)
        return nullptr;
      if (isa<scl::LogicalType>(v.getType())) {
        // boolean logic: rlo chain over bit directs
        r = boolLogic(def);
      } else {
        a = castTo(a, rt);
        b2 = castTo(b2, rt);
        if (isa<scl::AddOp>(def))
          r = b.create<awl::AddOp>(loc, rt, a, b2);
        else if (isa<scl::SubOp>(def))
          r = b.create<awl::SubOp>(loc, rt, a, b2);
        else if (isa<scl::MulOp>(def))
          r = b.create<awl::MulOp>(loc, rt, a, b2);
        else if (isa<scl::DivOp>(def))
          r = b.create<awl::DivOp>(loc, rt, a, b2);
        else if (isa<scl::AndOp>(def))
          r = b.create<awl::AndOp>(loc, rt, a, b2);
        else if (isa<scl::OrOp>(def))
          r = b.create<awl::OrOp>(loc, rt, a, b2);
        else
          r = b.create<awl::XOrOp>(loc, rt, a, b2);
      }
    } else if (auto m = dyn_cast<scl::ModOp>(def)) {
      // a mod b = a - (a / b) * b
      Value a = castTo(materialize(m.getLhs()), rt);
      Value b2 = castTo(materialize(m.getRhs()), rt);
      if (!a || !b2)
        return nullptr;
      Value q = b.create<awl::DivOp>(loc, rt, a, b2);
      Value t = b.create<awl::MulOp>(loc, rt, q, b2);
      r = b.create<awl::SubOp>(loc, rt, a, t);
    } else if (isa<scl::EqualOp, scl::NotEqualOp, scl::LessThanOp,
                   scl::LessEqualOp, scl::GreaterThanOp, scl::GreaterEqualOp>(
                   def)) {
      Value a = materialize(def->getOperand(0));
      Value b2 = materialize(def->getOperand(1));
      if (!a || !b2)
        return nullptr;
      if (isa<scl::LogicalType>(def->getOperand(0).getType())) {
        if (!isa<scl::EqualOp, scl::NotEqualOp>(def)) {
          emitError(def->getLoc()) << "ordering comparison on bools is not "
                                      "supported by the AWL backend";
          return nullptr;
        }
        r = boolCompare(def);
      } else {
        Type ot = awlValueType(ctx, def->getOperand(0).getType());
        if (ot && ot.isInteger(8))
          ot = IntegerType::get(ctx, 16);
        a = castTo(a, ot);
        b2 = castTo(b2, ot);
        Type i1 = IntegerType::get(ctx, 1);
        if (isa<scl::EqualOp>(def))
          r = b.create<awl::CmpEqOp>(loc, i1, a, b2);
        else if (isa<scl::NotEqualOp>(def))
          r = b.create<awl::CmpNeOp>(loc, i1, a, b2);
        else if (isa<scl::LessThanOp>(def))
          r = b.create<awl::CmpLtOp>(loc, i1, a, b2);
        else if (isa<scl::LessEqualOp>(def))
          r = b.create<awl::CmpLeOp>(loc, i1, a, b2);
        else if (isa<scl::GreaterThanOp>(def))
          r = b.create<awl::CmpGtOp>(loc, i1, a, b2);
        else
          r = b.create<awl::CmpGeOp>(loc, i1, a, b2);
      }
    } else if (auto n = dyn_cast<scl::UnaryNotOp>(def)) {
      r = boolNot(n.getRhs());
    } else if (auto n = dyn_cast<scl::UnaryMinusOp>(def)) {
      Value a = castTo(materialize(n.getRhs()), rt);
      if (!a)
        return nullptr;
      r = b.create<awl::SubOp>(loc, rt, constI(APInt(rt.getIntOrFloatBitWidth(), 0), rt), a);
    } else if (isa<scl::IntegerCastOp, scl::BitCastOp, scl::DialectCastOp>(
                   def)) {
      Value a = materialize(def->getOperand(0));
      if (!a)
        return nullptr;
      r = castTo(a, rt);
    } else {
      emitError(def->getLoc())
          << "operation not supported by the AWL backend: " << def->getName();
      return nullptr;
    }
    materialized[v] = r;
    return r;
  }

  scl::DirectType tempBit() {
    return scl::DirectType::get(ctx, scl::DirectArea::Marker, 0, nextBit / 8,
                                 nextBit % 8, 1);
  }

  Value constTrue() {
    return constI(APInt(1, 1), IntegerType::get(ctx, 1));
  }
  Value constFalse() {
    return constI(APInt(1, 0), IntegerType::get(ctx, 1));
  }

  // Get a bit direct for a boolean value, spilling computed values to a temp
  // bit. Returns null for constants (callers fold those).
  scl::DirectType asBitDirect(Value v) {
    if (v.getDefiningOp<scl::ConstantOp>())
      return nullptr;
    if (auto l = v.getDefiningOp<scl::LoadOp>())
      return directOf(l.getAddress());
    return spillBit(v);
  }

  // Spill an i1 value to a fresh temp bit and return its direct.
  scl::DirectType spillBit(Value v) {
    scl::DirectType dt = tempBit();
    ++nextBit;
    Value iv = materialize(v);
    if (!iv)
      return nullptr;
    b.create<awl::AssignOp>(loc, iv, directValue(dt));
    return dt;
  }

  Value rloAnd(Value rlo, scl::DirectType dt, bool negate = false) {
    SmallVector<NamedAttribute, 1> attrs;
    if (negate)
      attrs.push_back(NamedAttribute(strAttr("negate"), UnitAttr::get(ctx)));
    return b.create<awl::RloAndOp>(loc, TypeRange{IntegerType::get(ctx, 1)},
                                   ValueRange{rlo, directValue(dt)}, attrs);
  }
  Value rloOr(Value rlo, scl::DirectType dt, bool negate = false) {
    SmallVector<NamedAttribute, 1> attrs;
    if (negate)
      attrs.push_back(NamedAttribute(strAttr("negate"), UnitAttr::get(ctx)));
    return b.create<awl::RloOrOp>(loc, TypeRange{IntegerType::get(ctx, 1)},
                                  ValueRange{rlo, directValue(dt)}, attrs);
  }

  static bool isConstVal(Value v, bool one) {
    auto c = v.getDefiningOp<scl::ConstantOp>();
    if (!c)
      return false;
    auto ia = dyn_cast<IntegerAttr>(c.getValue());
    return ia && ia.getValue() == (one ? 1 : 0);
  }

  // Boolean and/or/xor as an rlo chain over bit directs.
  Value boolLogic(Operation *def) {
    Value x = def->getOperand(0), y = def->getOperand(1);
    bool isAnd = isa<scl::AndOp>(def);
    bool isOr = isa<scl::OrOp>(def);
    Type i1 = IntegerType::get(ctx, 1);
    // constant folding: and(x, true) = x, and(x, false) = false, etc.
    if (isConstVal(x, true))
      return isAnd ? materialize(y) : isOr ? constTrue() : materialize(y);
    if (isConstVal(x, false))
      return isAnd ? constFalse() : isOr ? materialize(y) : materialize(y);
    if (isConstVal(y, true))
      return isAnd ? materialize(x) : isOr ? constTrue() : materialize(x);
    if (isConstVal(y, false))
      return isAnd ? constFalse() : isOr ? materialize(x) : materialize(x);
    scl::DirectType da = asBitDirect(x);
    scl::DirectType db = asBitDirect(y);
    if (!da || !db)
      return nullptr;
    if (isAnd)
      return rloAnd(rloAnd(constTrue(), da), db);
    if (isOr)
      return rloOr(rloOr(constTrue(), da), db);
    // xor(a,b) = (a & ~b) | (~a & b)
    Value t1 = rloAnd(rloAnd(constTrue(), da), db, /*negate=*/true);
    Value t2 = rloAnd(rloAnd(constTrue(), da, /*negate=*/true), db);
    scl::DirectType dt2 = spillBit(t2);
    if (!dt2)
      return nullptr;
    return rloOr(t1, dt2);
  }

  Value boolNot(Value a) {
    if (isConstVal(a, true))
      return constFalse();
    if (isConstVal(a, false))
      return constTrue();
    scl::DirectType da = asBitDirect(a);
    if (!da)
      return nullptr;
    return rloAnd(constTrue(), da, /*negate=*/true);
  }

  // Boolean equality/inequality over bit directs (ordering comparisons on
  // bools are rejected by the caller).
  Value boolCompare(Operation *def) {
    Value x = def->getOperand(0), y = def->getOperand(1);
    bool isEq = isa<scl::EqualOp>(def);
    Value other = isConstVal(x, true) || isConstVal(x, false) ? y : x;
    if (isConstVal(x, true))
      return isEq ? materialize(y) : boolNot(y);
    if (isConstVal(x, false))
      return isEq ? boolNot(y) : materialize(y);
    if (isConstVal(y, true))
      return isEq ? materialize(x) : boolNot(x);
    if (isConstVal(y, false))
      return isEq ? boolNot(x) : materialize(x);
    scl::DirectType da = asBitDirect(x);
    scl::DirectType db = asBitDirect(y);
    if (!da || !db)
      return nullptr;
    if (isEq) {
      // eq(a,b) = (a & b) | (~a & ~b)
      Value u1 = rloAnd(rloAnd(constTrue(), da), db);
      Value u2 = rloAnd(rloAnd(constTrue(), da, true), db, true);
      scl::DirectType du2 = spillBit(u2);
      if (!du2)
        return nullptr;
      return rloOr(u1, du2);
    }
    // ne(a,b) = (a & ~b) | (~a & b)
    Value t1 = rloAnd(rloAnd(constTrue(), da), db, true);
    Value t2 = rloAnd(rloAnd(constTrue(), da, true), db);
    scl::DirectType dt2 = spillBit(t2);
    if (!dt2)
      return nullptr;
    return rloOr(t1, dt2);
  }

  void emitStore(Value addr, Value val) {
    scl::DirectType dt = directOf(addr);
    if (!dt)
      return;
    if (dt.getWidth() == 1) {
      Value r = materialize(val);
      if (!r)
        return;
      if (isConstVal(val, true))
        b.create<awl::SetOp>(loc, IntegerType::get(ctx, 1), directValue(dt));
      else if (isConstVal(val, false))
        b.create<awl::ResetOp>(loc, IntegerType::get(ctx, 1), directValue(dt));
      else
        b.create<awl::AssignOp>(loc, r, directValue(dt));
    } else {
      Type vt = awlDirectValueType(ctx, dt);
      Value raw = materialize(val);
      if (!raw)
        return;
      // cross-width assignment: insert an explicit cast (awl.cast)
      raw = castTo(raw, vt);
      b.create<awl::StoreOp>(loc, raw, directValue(dt));
    }
  }

  Value materializeCondition(Value cond) { return materialize(cond); }

  // Like materialize() but always emits a fresh producer (used where the
  // AWL text needs a re-load, e.g. the selector before each CASE compare:
  // in AWL each `L` shifts ACC1 into ACC2, so the selector must be reloaded).
  Value materializeFresh(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return materialize(v);
    if (auto l = dyn_cast<scl::LoadOp>(def)) {
      scl::DirectType dt = directOf(l.getAddress());
      if (!dt)
        return nullptr;
      return b.create<awl::LoadOp>(loc, awlDirectValueType(ctx, dt),
                                   directValue(dt));
    }
    return materialize(v);
  }

  void emitLabel(llvm::StringRef name) {
    b.create<awl::LabelOp>(loc, strAttr(name));
  }
  void emitJump(llvm::StringRef name) {
    b.create<awl::JumpOp>(loc, strAttr(name));
  }
  void emitJumpIf(Value rlo, llvm::StringRef name) {
    b.create<awl::JumpIfOp>(loc, rlo, strAttr(name));
  }
  void emitJumpIfNot(Value rlo, llvm::StringRef name) {
    b.create<awl::JumpIfNotOp>(loc, rlo, strAttr(name));
  }

  LogicalResult emitRegion(Region &region,
                           DenseMap<Block *, std::string> &blockLabels) {
    for (Block &blk : region) {
      if (&blk != &region.front())
        emitLabel(blockLabels[&blk]);
      for (Operation &op : blk) {
        if (failed(emitOp(&op, blockLabels)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult emitOp(Operation *op,
                       DenseMap<Block *, std::string> &blockLabels) {
    if (auto tv = dyn_cast<scl::TempVariableOp>(op)) {
      Type elem = tv.getResult().getType().cast<scl::AddressType>()
                      .getElementType();
      auto pit = paramDirects.find(tv.getResult());
      scl::DirectType dt =
          pit != paramDirects.end() ? pit->second : allocDirect(elem);
      if (!dt)
        return failure();
      varDirect[tv.getResult()] = dt;
      return success();
    }
    if (isa<scl::VariableOp>(op))
      return success(); // declaration only; the layout is precomputed
    if (isa<scl::StoreOp>(op)) {
      auto st = cast<scl::StoreOp>(op);
      // the entry-block copy of an FC input argument: the caller already
      // stored the value into the parameter window slot
      if (!st.getRhs().getDefiningOp())
        return success();
      emitStore(st.getLhs(), st.getRhs());
      return success();
    }
    if (auto iff = dyn_cast<scl::IfThenElseOp>(op)) {
      Value c = materializeCondition(iff.getCond());
      if (!c)
        return failure();
      std::string elseL = newLabel("ELSE"), endL = newLabel("END");
      emitJumpIfNot(c, elseL);
      if (failed(emitRegion(iff.getThenBody(), blockLabels)))
        return failure();
      emitJump(endL);
      emitLabel(elseL);
      if (!iff.getElseBody().empty())
        if (failed(emitRegion(iff.getElseBody(), blockLabels)))
          return failure();
      emitLabel(endL);
      return success();
    }
    if (auto w = dyn_cast<scl::WhileDoOp>(op)) {
      std::string condL = newLabel("WH"), bodyL = newLabel("BODY"),
                  endL = newLabel("END");
      emitLabel(condL);
      loops.push_back({condL, endL});
      if (failed(emitLoopHeader(w.getWhileBody(), bodyL, endL)))
        return failure();
      emitLabel(bodyL);
      if (failed(emitRegion(w.getDoBody(), blockLabels)))
        return failure();
      emitJump(condL);
      emitLabel(endL);
      loops.pop_back();
      return success();
    }
    if (auto r = dyn_cast<scl::RepeatOp>(op)) {
      std::string headL = newLabel("REP"), condL = newLabel("UNTIL"),
                  endL = newLabel("END");
      emitLabel(headL);
      loops.push_back({condL, endL});
      if (failed(emitLoopFooter(r.getRepeatBody(), condL, headL)))
        return failure();
      emitLabel(endL);
      loops.pop_back();
      return success();
    }
    if (auto f = dyn_cast<scl::ForDoOp>(op)) {
      return emitFor(f, blockLabels);
    }
    if (auto c = dyn_cast<scl::CaseOp>(op)) {
      return emitCase(c, blockLabels);
    }
    if (auto call = dyn_cast<scl::CallFcOp>(op)) {
      // a call whose result is used is emitted by materialize() at the use
      // site; only emit a statement-level call when the result is unused
      if (call->getResult(0).use_empty())
        emitCallFc(call, false);
      return success();
    }
    if (auto call = dyn_cast<scl::CallFbOp>(op)) {
      // `CALL FB n, DB m`: the instance DB number comes from the IDB symbol
      // (a data block named DB m).
      std::string fbStr = llvm::StringRef(call.getFb()).upper();
      llvm::StringRef fb = fbStr;
      int32_t db = 0;
      if (auto gg = call.getIdb().getDefiningOp<scl::GetGlobalOp>()) {
        auto it = dbNumbers.find(gg.getName());
        if (it != dbNumbers.end())
          db = it->second;
      }
      SmallVector<NamedAttribute, 2> attrs;
      attrs.push_back(NamedAttribute(strAttr("callee"),
                                     FlatSymbolRefAttr::get(ctx, fb)));
      attrs.push_back(
          NamedAttribute(strAttr("db"), b.getI32IntegerAttr(db)));
      b.create<awl::CallOp>(loc, TypeRange{}, ValueRange{}, attrs);
      return success();
    }
    if (isa<scl::DebugPrintOp>(op))
      return success(); // no AWL equivalent; dropped (docs/design)
    if (isa<scl::ReturnOp, scl::ReturnValueOp>(op)) {
      b.create<awl::BeOp>(loc);
      return success();
    }
    if (isa<scl::ContinueOp>(op)) {
      if (loops.empty()) {
        emitError(op->getLoc()) << "continue outside a loop";
        return failure();
      }
      emitJump(loops.back().cont);
      return success();
    }
    if (isa<scl::ExitOp>(op)) {
      if (loops.empty()) {
        emitError(op->getLoc()) << "exit outside a loop";
        return failure();
      }
      emitJump(loops.back().end);
      return success();
    }
    if (isa<scl::EndOp>(op))
      return success();
    if (auto br = dyn_cast<cf::BranchOp>(op)) {
      auto it = blockLabels.find(br.getDest());
      if (it == blockLabels.end()) {
        emitError(op->getLoc()) << "goto target outside the current region is "
                                   "not supported by the AWL backend";
        return failure();
      }
      emitJump(it->second);
      return success();
    }
    if (auto cbr = dyn_cast<cf::CondBranchOp>(op)) {
      Value c = materialize(cbr.getCondition());
      if (!c)
        return failure();
      if (!blockLabels.count(cbr.getTrueDest()) ||
          !blockLabels.count(cbr.getFalseDest())) {
        emitError(op->getLoc()) << "goto target outside the current region is "
                                   "not supported by the AWL backend";
        return failure();
      }
      emitJumpIf(c, blockLabels[cbr.getTrueDest()]);
      emitJump(blockLabels[cbr.getFalseDest()]);
      return success();
    }
    if (isa<scl::ConditionOp, scl::UntilOp>(op)) {
      emitError(op->getLoc()) << "loop condition outside a loop header";
      return failure();
    }
    // pure value ops are emitted on demand by materialize()
    if (op->getNumResults() > 0 &&
        isa<scl::LoadOp, scl::ConstantOp, scl::AddOp, scl::SubOp, scl::MulOp,
            scl::DivOp, scl::ModOp, scl::ExpOp, scl::AndOp, scl::OrOp,
            scl::XOrOp, scl::EqualOp, scl::NotEqualOp, scl::LessThanOp,
            scl::LessEqualOp, scl::GreaterThanOp, scl::GreaterEqualOp,
            scl::UnaryNotOp, scl::UnaryMinusOp, scl::IntegerCastOp,
            scl::BitCastOp, scl::DialectCastOp, scl::GetDirectOp,
            scl::GetVariableOp, scl::GetGlobalOp, scl::GetIndexOp,
            scl::GetElementOp>(op))
      return success();
    emitError(op->getLoc()) << "operation not supported by the AWL backend: "
                            << op->getName();
    return failure();
  }

  // while header: blocks ending in scl.condition; true -> body, false -> end
  LogicalResult emitLoopHeader(Region &region, llvm::StringRef bodyL,
                               llvm::StringRef endL) {
    DenseMap<Block *, std::string> blockLabels;
    int n = 0;
    for (Block &blk : region)
      if (&blk != &region.front())
        blockLabels[&blk] = "BB" + std::to_string(n++);
    for (Block &blk : region) {
      if (&blk != &region.front()) {
        // skip blocks that are just an empty scl.end (no label needed)
        bool onlyEnd = llvm::all_of(blk.getOperations(),
                                    [](Operation &o) { return isa<scl::EndOp>(&o); });
        if (!onlyEnd)
          emitLabel(blockLabels[&blk]);
      }
      for (Operation &op : blk) {
        if (auto cond = dyn_cast<scl::ConditionOp>(&op)) {
          Value c = materializeCondition(cond.getCond());
          if (!c)
            return failure();
          emitJumpIf(c, bodyL);
          emitJump(endL);
          continue;
        }
        if (failed(emitOp(&op, blockLabels)))
          return failure();
      }
    }
    return success();
  }

  // repeat footer: blocks ending in scl.until; true -> exit, false -> head
  LogicalResult emitLoopFooter(Region &region, llvm::StringRef condL,
                               llvm::StringRef headL) {
    DenseMap<Block *, std::string> blockLabels;
    int n = 0;
    for (Block &blk : region)
      if (&blk != &region.front())
        blockLabels[&blk] = "BB" + std::to_string(n++);
    for (Block &blk : region) {
      if (&blk != &region.front()) {
        bool onlyEnd = llvm::all_of(blk.getOperations(),
                                    [](Operation &o) { return isa<scl::EndOp>(&o); });
        if (!onlyEnd)
          emitLabel(blockLabels[&blk]);
      }
      for (Operation &op : blk) {
        if (auto until = dyn_cast<scl::UntilOp>(&op)) {
          emitLabel(condL);
          Value c = materializeCondition(until.getCond());
          if (!c)
            return failure();
          emitJumpIfNot(c, headL);
          continue;
        }
        if (failed(emitOp(&op, blockLabels)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult emitFor(scl::ForDoOp f,
                        DenseMap<Block *, std::string> &blockLabels) {
    scl::DirectType dt = directOf(f.getVar());
    if (!dt)
      return failure();
    emitStore(f.getVar(), f.getInit());
    std::string condL = newLabel("FOR"), bodyL = newLabel("BODY"),
                stepL = newLabel("STEP"), endL = newLabel("END");
    emitLabel(condL);
    // cond: var <= to  (assumes positive step)
    Type vt = awlDirectValueType(ctx, dt);
    Value v = b.create<awl::LoadOp>(loc, vt, directValue(dt));
    Value t = castTo(materialize(f.getTo()), vt);
    Value c = b.create<awl::CmpLeOp>(loc, IntegerType::get(ctx, 1),
                                     castTo(v, vt), t);
    emitJumpIfNot(c, endL);
    emitLabel(bodyL);
    loops.push_back({stepL, endL});
    if (failed(emitRegion(f.getDoBody(), blockLabels)))
      return failure();
    loops.pop_back();
    emitLabel(stepL);
    // var := var + by
    Value by = f.getBy() ? materialize(f.getBy()) : nullptr;
    if (!by)
      by = constI(APInt(vt.getIntOrFloatBitWidth(), 1), vt);
    Value v2 = b.create<awl::LoadOp>(loc, vt, directValue(dt));
    Value nv = b.create<awl::AddOp>(loc, vt, castTo(v2, vt), castTo(by, vt));
    b.create<awl::StoreOp>(loc, nv, directValue(dt));
    emitJump(condL);
    emitLabel(endL);
    return success();
  }

  LogicalResult emitCase(scl::CaseOp c,
                         DenseMap<Block *, std::string> &blockLabels) {
    Value sel = materialize(c.getSelector());
    if (!sel)
      return failure();
    Type st = awlValueType(ctx, c.getSelector().getType());
    sel = castTo(sel, st);
    ArrayAttr caseValues = c.getCaseValues();
    // one region (single block) per case; the last is the else block
    SmallVector<Region *, 8> regions;
    for (Region &r : c.getBody())
      regions.push_back(&r);
    SmallVector<Block *, 8> blocks;
    for (Region *r : regions)
      blocks.push_back(&r->front());
    int numRegions = blocks.size();
    std::string endL = newLabel("CASEEND");
    SmallVector<std::string, 8> caseLabels;
    for (int i = 0; i < numRegions; ++i)
      caseLabels.push_back(newLabel("CASE"));
    // dispatch: compare chain over the case values; the selector is
    // re-loaded before each compare (AWL L shifts ACC1 into ACC2)
    for (int i = 0; i + 1 < numRegions; ++i) {
      auto vals = cast<DenseI32ArrayAttr>(caseValues[i]);
      for (int32_t val : vals.asArrayRef()) {
        Value selV = castTo(materializeFresh(c.getSelector()), st);
        Value cv = constI(APInt(st.getIntOrFloatBitWidth(), val, true), st);
        Value cc = b.create<awl::CmpEqOp>(loc, IntegerType::get(ctx, 1), selV,
                                          castTo(cv, st));
        emitJumpIf(cc, caseLabels[i]);
      }
    }
    emitJump(caseLabels.back()); // else
    for (int i = 0; i < numRegions; ++i) {
      emitLabel(caseLabels[i]);
      // walk just this case block; goto targets inside a case body are not
      // supported (emitOp errors on an unknown successor label)
      for (Operation &op : *blocks[i])
        if (failed(emitOp(&op, blockLabels)))
          return failure();
      emitJump(endL);
    }
    emitLabel(endL);
    return success();
  }

  LogicalResult run(Operation *funcOp, llvm::StringRef name) {
    // Start M-area local allocation after the bytes used by direct %M
    // addresses in this function, so locals and directs don't collide.
    {
      int32_t maxBit = 0;
      funcOp->walk([&](scl::GetDirectOp gd) {
        if (gd.getArea() == "M") {
          int32_t end = gd.getByte() * 8 + gd.getBit() + gd.getWidth();
          maxBit = std::max(maxBit, end);
        }
      });
      nextBit = maxBit;
    }
    b.setInsertionPointToEnd(module.getBody());
    blockOp = b.create<awl::BlockOp>(loc, strAttr(name), mlir::IntegerAttr(),
                                    mlir::IntegerAttr(), mlir::UnitAttr());
    flat = b.createBlock(&blockOp.getBody());
    b.setInsertionPointToStart(flat);
    Region &body = funcOp->getRegion(0);
    DenseMap<Block *, std::string> blockLabels;
    int n = 0;
    for (Block &blk : body)
      if (&blk != &body.front())
        blockLabels[&blk] = "BB" + std::to_string(n++);
    if (auto fb = dyn_cast<scl::FunctionBlockOp>(funcOp)) {
      // the FB's instance DB address arrives as the single block argument
      if (body.front().getArguments().size() != 1) {
        emitError(funcOp->getLoc()) << "FB needs exactly one IDB argument";
        return failure();
      }
      currentFbArg = body.front().getArgument(0);
    } else if (!body.front().getArguments().empty()) {
      // FC inputs: the entry block stores each block argument into its input
      // tempvar; map the i-th such tempvar to parameter window slot i.
      unsigned i = 0;
      for (Operation &o : body.front()) {
        auto st = dyn_cast<scl::StoreOp>(&o);
        if (!st || st.getRhs().getDefiningOp() != nullptr)
          continue;
        auto tv = st.getLhs().getDefiningOp<scl::TempVariableOp>();
        if (!tv)
          continue;
        Type elem = tv.getResult().getType().cast<scl::AddressType>()
                        .getElementType();
        int w = widthOf(elem);
        if (w <= 0) {
          emitError(funcOp->getLoc())
              << "FC input type not representable in AWL output: " << elem;
          return failure();
        }
        paramDirects[tv.getResult()] = scl::DirectType::get(
            ctx, scl::DirectArea::Marker, 0, paramSlot((int)i), 0, w);
        ++i;
      }
      if (i != body.front().getArguments().size()) {
        emitError(funcOp->getLoc())
            << "FC input count mismatch with block arguments";
        return failure();
      }
    }
    if (failed(emitRegion(body, blockLabels)))
      return failure();
    // An FC with a return value stores it into the result slot before BE.
    if (auto fn = dyn_cast<scl::FunctionOp>(funcOp)) {
      auto results = fn.getFunctionType().getResults();
      std::string fnName = llvm::StringRef(fn.getSymName()).upper();
      bool isFc = llvm::StringRef(fnName).startswith("FC");
      if (isFc && !results.empty() && !flat->empty() &&
          isa<awl::BeOp>(flat->back())) {
        Type rt = awlValueType(ctx, results[0]);
        for (Operation &o : body.front()) {
          auto tv = dyn_cast<scl::TempVariableOp>(&o);
          if (tv && tv.getName() == fn.getSymName() && rt) {
            auto dit = varDirect.find(tv.getResult());
            if (dit != varDirect.end()) {
              b.setInsertionPoint(&flat->back()); // before the BE
              Value v =
                  b.create<awl::LoadOp>(loc, rt, directValue(dit->second));
              scl::DirectType rdt = scl::DirectType::get(
                  ctx, scl::DirectArea::Marker, 0, kResultByte, 0,
                  rt.getIntOrFloatBitWidth());
              b.create<awl::StoreOp>(loc, v, directValue(rdt));
            }
          }
        }
      }
    }
    if (flat->empty() || !isa<awl::BeOp>(flat->back()))
      b.create<awl::BeOp>(loc);
    funcOp->erase();
    return success();
  }
};

struct LowerToAWLPass
    : public PassWrapper<LowerToAWLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToAWLPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<awl::AwlDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 8> funcs;
    module.walk([&](Operation *op) {
      if (isa<scl::FunctionOp, scl::FunctionBlockOp>(op))
        funcs.push_back(op);
    });
    // Map data-block (IDB) symbol names to AWL DB numbers: names like DB m
    // keep their number, others get sequential numbers.
    DenseMap<llvm::StringRef, int32_t> dbNumbers;
    {
      int32_t next = 1;
      SmallVector<llvm::StringRef, 8> names;
      module.walk([&](scl::DataBlockOp op) { names.push_back(op.getSymName()); });
      for (auto n : names) {
        std::string upStr = llvm::StringRef(n).upper();
        llvm::StringRef up = upStr;
        if (up.size() > 2 && up.startswith("DB") &&
            llvm::all_of(up.drop_front(2), [](char c) { return isdigit(c); }))
          dbNumbers[n] = atoi(up.drop_front(2).str().c_str());
        else
          dbNumbers[n] = next++;
      }
    }
    // FC signatures (input parameter names in declaration order), collected
    // before any function is erased
    llvm::StringMap<SmallVector<std::string, 4>> fcSignatures;
    for (auto fn : module.getOps<scl::FunctionOp>()) {
      SmallVector<std::string, 4> params;
      for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
        if (auto na = fn.getArgAttr(i, "scl.name"))
          params.push_back(cast<StringAttr>(na).getValue().str());
      }
      if (!params.empty())
        fcSignatures[std::string(llvm::StringRef(fn.getSymName()).upper())] =
            std::move(params);
    }
    // FB member layouts (sequential byte offsets, as in computeFbLayouts)
    llvm::StringMap<AwlFbLayout> fbLayouts;
    for (auto fb : module.getOps<scl::FunctionBlockOp>()) {
      AwlFbLayout layout;
      int64_t off = 0;
      for (auto var : fb.getBody().getOps<scl::VariableOp>()) {
        int w = widthOf(var.getType());
        if (w <= 0)
          continue;
        layout[var.getSymName().str()] = {off, w, var.getType()};
        off += (w + 7) / 8;
      }
      fbLayouts[std::string(llvm::StringRef(fb.getName()).upper())] = layout;
    }
    for (Operation *op : funcs) {
      llvm::StringRef name = cast<SymbolOpInterface>(op).getName();
      if (name.startswith("_mlir_ciface_")) {
        op->erase(); // JIT wrapper; the AWL frontend re-synthesizes it
        continue;
      }
      std::string upperStr = llvm::StringRef(name).upper();
      llvm::StringRef upper = upperStr;
      bool ok = upper.size() > 2 &&
                (upper.startswith("OB") || upper.startswith("FB") ||
                 upper.startswith("FC")) &&
                llvm::all_of(upper.drop_front(2),
                             [](char c) { return isdigit(c); });
      if (!ok) {
        op->emitError("only OB/FB/FC blocks can be lowered to AWL, got @"
                      + name);
        signalPassFailure();
        return;
      }
      FuncConverter c(&getContext(), module, op->getLoc());
      c.dbNumbers = dbNumbers;
      c.fbLayouts = &fbLayouts;
      c.fcSignatures = fcSignatures;
      if (failed(c.run(op, upper))) {
        signalPassFailure();
        return;
      }
    }
    // Data blocks have no AWL declaration; initializers are dropped.
    SmallVector<Operation *, 4> dbs;
    module.walk([&](scl::DataBlockOp op) { dbs.push_back(op); });
    for (Operation *op : dbs) {
      if (auto db = dyn_cast<scl::DataBlockOp>(op))
        if (db.getInitAttr())
          db.emitWarning("data block initial values are not representable in "
                         "AWL output; dropped");
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerToAWLPass() {
  return std::make_unique<LowerToAWLPass>();
}

} // namespace sclang
} // namespace mlir
