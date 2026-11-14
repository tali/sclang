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
  if (isa<scl::LogicalType>(t))
    return 1;
  if (auto it = dyn_cast<scl::IntegerType>(t))
    return it.getWidth();
  if (isa<scl::RealType>(t))
    return 32;
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
  DenseMap<Value, Value> materialized;        // scl value -> awl value
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
    emitError(addr.getLoc())
        << "address not representable as an AWL operand (FB members, arrays, "
           "structs and DB symbols are not yet supported by the AWL backend)";
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
    if (!rt) {
      emitError(def->getLoc()) << "type not representable in AWL output: "
                               << v.getType();
      return nullptr;
    }
    if (auto c = dyn_cast<scl::ConstantOp>(def)) {
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
      if (raw.getType() != vt) {
        emitError(val.getLoc())
            << "cross-width assignment is not yet supported by the AWL backend";
        return;
      }
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
      scl::DirectType dt = allocDirect(elem);
      if (!dt)
        return failure();
      varDirect[tv.getResult()] = dt;
      return success();
    }
    if (isa<scl::StoreOp>(op)) {
      auto st = cast<scl::StoreOp>(op);
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
      llvm::StringRef callee = call.getCallee();
      if (!callee.startswith("FC")) {
        emitError(op->getLoc()) << "AWL backend only supports CALL FC n, got @"
                                << callee;
        return failure();
      }
      SmallVector<NamedAttribute, 1> attrs;
      attrs.push_back(NamedAttribute(
          strAttr("callee"), FlatSymbolRefAttr::get(ctx, callee)));
      b.create<awl::CallOp>(loc, TypeRange{}, ValueRange{}, attrs);
      return success();
    }
    if (isa<scl::CallFbOp>(op)) {
      emitError(op->getLoc())
          << "CALL FB n, DB m is not yet supported by the AWL backend";
      return failure();
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
            scl::BitCastOp, scl::DialectCastOp, scl::GetDirectOp>(op))
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
    b.setInsertionPointToEnd(module.getBody());
    blockOp = b.create<awl::BlockOp>(loc, strAttr(name));
    flat = b.createBlock(&blockOp.getBody());
    b.setInsertionPointToStart(flat);
    Region &body = funcOp->getRegion(0);
    DenseMap<Block *, std::string> blockLabels;
    int n = 0;
    for (Block &blk : body)
      if (&blk != &body.front())
        blockLabels[&blk] = "BB" + std::to_string(n++);
    if (!body.front().getArguments().empty()) {
      emitError(funcOp->getLoc())
          << "block parameters (FC inputs) are not yet supported by the AWL "
             "backend";
      return failure();
    }
    if (failed(emitRegion(body, blockLabels)))
      return failure();
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
    for (Operation *op : funcs) {
      llvm::StringRef name = cast<SymbolOpInterface>(op).getName();
      if (name.startswith("_mlir_ciface_")) {
        op->erase(); // JIT wrapper; the AWL frontend re-synthesizes it
        continue;
      }
      bool ok = name.size() > 2 &&
                (name.startswith("OB") || name.startswith("FB") ||
                 name.startswith("FC")) &&
                llvm::all_of(name.drop_front(2),
                             [](char c) { return isdigit(c); });
      if (!ok) {
        op->emitError("only OB/FB/FC blocks can be lowered to AWL, got @"
                      + name);
        signalPassFailure();
        return;
      }
      FuncConverter c(&getContext(), module, op->getLoc());
      if (failed(c.run(op, name))) {
        signalPassFailure();
        return;
      }
    }
    // Data blocks have no AWL declaration; initializers are dropped.
    SmallVector<Operation *, 4> dbs;
    module.walk([&](scl::DataBlockOp op) { dbs.push_back(op); });
    for (Operation *op : dbs) {
      op->emitWarning("data block initializers are not representable in AWL "
                      "output; dropped");
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
