//===- Printer.cpp - AWL (STL) text printer ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prints the flat ops of an `awl.block` as S7 AWL text by simulating the
// two-accumulator model: word values live on a virtual ACC stack (`L`/`T`,
// arith pops two pushes one), bit values live in the virtual RLO (`U`/`O`/
// `NOT`/`=`/`S`/`R`, comparisons set RLO). `awl.cast` ops are implicit in the
// mnemonic letter and are not printed. Bracket expressions (`A(`/`O(`) are
// reconstructed from the rlo-chain + closing And/Or(i1) pattern the frontend
// produces. docs/design/scl-to-awl.md.
//
//===----------------------------------------------------------------------===//

#include "sclang/AwlDialect/Printer.h"

#include "sclang/AwlDialect/AwlOps.h"
#include "sclang/SclDialect/SclTypes.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>

using namespace mlir;

namespace mlir {
namespace awl {
namespace {

struct TextPrinter {
  raw_ostream &os;
  Block *flat = nullptr;

  // Virtual ACC stack (values) and RLO (value or null).
  std::vector<Value> acc;
  Value curRlo;

  // Pre-passes results.
  DenseSet<Value> skipConsts;   // i1 const-true rlo-init / bracket-start
  DenseMap<Value, char> bracketOpen; // const-true -> 'A' or 'O'
  DenseSet<Operation *> bracketClose; // And/Or(i1) closing a bracket
  DenseMap<Value, std::string> directText;
  DenseMap<Value, std::string> s5Text; // S5TIME consts -> "S5T#.." literal

  explicit TextPrinter(raw_ostream &os) : os(os) {}

  // MARK: helpers

  static char typeLetter(Type t) {
    if (t.isInteger(16))
      return 'I';
    if (t.isInteger(32))
      return 'D';
    if (t.isF32())
      return 'R';
    return '?';
  }

  static std::string operandText(scl::DirectType dt) {
    int32_t w = dt.getWidth();
    char sizeLetter = w == 1 ? 'X' : w == 8 ? 'B' : w == 16 ? 'W' : 'D';
    std::string s;
    if (dt.getArea() == scl::DirectArea::DataBlock) {
      s = "DB" + std::to_string(dt.getDb()) + ".DB" + sizeLetter;
      s += std::to_string(dt.getByte());
      if (w == 1)
        s += "." + std::to_string(dt.getBit());
    } else {
      char a = 'M';
      switch (dt.getArea()) {
      case scl::DirectArea::Input: a = 'E'; break;
      case scl::DirectArea::Output: a = 'A'; break;
      case scl::DirectArea::Marker: a = 'M'; break;
      case scl::DirectArea::Timer: a = 'T'; break;
      case scl::DirectArea::Counter: a = 'Z'; break;
      default: break;
      }
      s = std::string(1, a);
      // T/Z operands print without a size letter (T0 / Z0), bit access as T0.0
      bool isTimerCounter = dt.getArea() == scl::DirectArea::Timer ||
                            dt.getArea() == scl::DirectArea::Counter;
      if (w != 1 && !isTimerCounter)
        s += sizeLetter;
      s += std::to_string(dt.getByte());
      if (w == 1 && !isTimerCounter)
        s += "." + std::to_string(dt.getBit());
    }
    return s;
  }

  static bool isConstTrue(Value v) {
    auto c = v.getDefiningOp<ConstOp>();
    if (!c)
      return false;
    auto ia = dyn_cast<IntegerAttr>(c.getValue());
    return ia && ia.getType().isInteger(1) && ia.getValue().isOne();
  }

  std::string directTextOf(Value v) {
    auto it = directText.find(v);
    if (it != directText.end())
      return it->second;
    auto dt = v.getType().dyn_cast<scl::DirectType>();
    std::string s = dt ? operandText(dt) : std::string("?");
    directText[v] = s;
    return s;
  }

  std::string constText(Attribute attr, Type t) {
    if (auto ia = dyn_cast<IntegerAttr>(attr)) {
      if (ia.getType().isInteger(1))
        return ia.getValue().isOne() ? "1" : "0";
      return std::to_string(ia.getValue().getSExtValue());
    }
    if (auto fa = dyn_cast<FloatAttr>(attr))
      return fa.getValue().convertToDouble() == 0.0
                 ? "0.0"
                 : std::to_string(fa.getValue().convertToDouble());
    if (t.isInteger(1))
      return "1";
    return "?";
  }

  Value popAcc() {
    if (acc.empty())
      return nullptr; // tolerate unbalanced input; callers just pop
    Value v = acc.back();
    acc.pop_back();
    return v;
  }

  // MARK: pre-pass

  // Detect bracket groups: And/Or(i1) whose rhs is an rlo chain rooted at a
  // const-true; mark the const as the `A(`/`O(` opener and the And/Or as the
  // `)` closer.
  void prepass() {
    for (Operation &o : flat->getOperations()) {
      Operation *combining = nullptr;
      if (auto a = dyn_cast<AndOp>(&o))
        combining = a;
      else if (auto orr = dyn_cast<OrOp>(&o))
        combining = orr;
      if (!combining || !combining->getResult(0).getType().isInteger(1))
        continue;
      Value v = combining->getOperand(1);
      while (Operation *def = v.getDefiningOp()) {
        if (isa<RloAndOp, RloOrOp, NotOp>(def)) {
          v = def->getOperand(0);
          continue;
        }
        break;
      }
      if (isConstTrue(v)) {
        bracketOpen[v] = isa<OrOp>(&o) ? 'O' : 'A';
        bracketClose.insert(&o);
        skipConsts.insert(v);
      }
    }
    // rlo-init consts: const true whose uses are only rlo ops / assigns as
    // the rlo operand (the neutral RLO at the start of a logic chain)
    for (Operation &o : flat->getOperations()) {
      auto c = dyn_cast<ConstOp>(&o);
      if (!c)
        continue;
      Value res = c->getResult(0);
      if (isConstTrue(res) && !bracketOpen.count(res)) {
        bool onlyRloInit =
            res.use_empty() ||
            llvm::all_of(res.getUsers(), [&](Operation *u) {
              return (isa<RloAndOp, RloOrOp>(u) && u->getOperand(0) == res) ||
                     (isa<AssignOp>(u) && u->getOperand(0) == res);
            });
        if (onlyRloInit)
          skipConsts.insert(res);
      }
      // the placeholder value operand of ZV/ZR/R counter ops is not printed
      if (!res.use_empty() &&
          llvm::all_of(res.getUsers(), [&](Operation *u) {
            return isa<CounterOp>(u) && u->getOperand(0) == res &&
                   cast<CounterOp>(u).getOp() != 2;
          }))
        skipConsts.insert(res);
      // S5TIME consts feeding a timer start print as S5T# literals
      if (auto ia = o.getAttrOfType<IntegerAttr>("value")) {
        for (Operation *u : res.getUsers()) {
          if (auto ts = dyn_cast<TimerStartOp>(u)) {
            if (ts.getDur() == res) {
              uint64_t v = ia.getValue().getZExtValue();
              uint64_t code = (v >> 12) & 3, bcd = v & 0xFFF;
              uint64_t val =
                  ((bcd >> 8) & 0xF) * 100 + ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
              // print with the short units the lexer understands:
              // MS = 10ms (code 0), S = 1s (code 2); codes 1/3 fold x10
              uint64_t eff = (code == 1 || code == 3) ? val * 10 : val;
              s5Text[res] = "S5T#" + std::to_string(eff) +
                            (code <= 1 ? "MS" : "S");
            }
          }
        }
      }
    }
  }

  // MARK: main walk

  void printBlock(BlockOp block) {
    flat = &block.getBody().front();
    prepass();
    // header: OB n / FB n / FC n
    llvm::StringRef name = block.getSymName();
    if (name.size() > 2 && (name.startswith("OB") || name.startswith("FB") ||
                            name.startswith("FC"))) {
      os << name.take_front(2) << ' ' << name.drop_front(2) << '\n';
    } else {
      os << "// block " << name << '\n';
    }
    for (Operation &o : flat->getOperations()) {
      if (auto d = dyn_cast<DirectOp>(&o)) {
        directText[d.getResult()] = operandText(
            d.getResult().getType().cast<scl::DirectType>());
        continue;
      }
      if (auto c = dyn_cast<ConstOp>(&o)) {
        if (skipConsts.count(c.getResult())) {
          auto it = bracketOpen.find(c.getResult());
          if (it != bracketOpen.end())
            os << "  " << it->second << "(\n";
          continue;
        }
        auto s5 = s5Text.find(c.getResult());
        os << "  L "
           << (s5 != s5Text.end() ? s5->second
                                  : constText(c.getValue(),
                                              c.getResult().getType()))
           << '\n';
        acc.push_back(c.getResult());
        continue;
      }
      if (auto l = dyn_cast<LoadOp>(&o)) {
        std::string op = directTextOf(l.getAddr());
        if (l.getResult().getType().isInteger(1)) {
          os << "  U " << op << '\n'; // bit load goes to RLO
          curRlo = l.getResult();
        } else {
          os << "  L " << op << '\n';
          acc.push_back(l.getResult());
        }
        continue;
      }
      if (auto st = dyn_cast<StoreOp>(&o)) {
        std::string op = directTextOf(st.getAddr());
        if (st.getValue().getType().isInteger(1)) {
          if (isConstTrue(st.getValue()))
            os << "  S " << op << '\n';
          else if (auto c = st.getValue().getDefiningOp<ConstOp>()) {
            auto ia = dyn_cast<IntegerAttr>(c.getValue());
            os << "  " << (ia && ia.getValue().isZero() ? "R " : "= ") << op
               << '\n';
          } else {
            os << "  = " << op << '\n';
          }
        } else {
          os << "  T " << op << '\n';
          popAcc();
        }
        continue;
      }
      if (auto r = dyn_cast<RloAndOp>(&o)) {
        os << "  U" << (r.getNegate() ? "N " : " ") << directTextOf(r.getAddr())
           << '\n';
        curRlo = r.getResult();
        continue;
      }
      if (auto r = dyn_cast<RloOrOp>(&o)) {
        os << "  O" << (r.getNegate() ? "N " : " ") << directTextOf(r.getAddr())
           << '\n';
        curRlo = r.getResult();
        continue;
      }
      if (isa<NotOp>(&o)) {
        os << "  NOT\n";
        curRlo = o.getResult(0);
        continue;
      }
      if (isa<AssignOp>(&o)) {
        auto a = cast<AssignOp>(&o);
        os << "  = " << directTextOf(a.getAddr()) << '\n';
        continue;
      }
      if (auto s = dyn_cast<SetOp>(&o)) {
        os << "  S " << directTextOf(s.getAddr()) << '\n';
        curRlo = s.getResult();
        continue;
      }
      if (auto r = dyn_cast<ResetOp>(&o)) {
        os << "  R " << directTextOf(r.getAddr()) << '\n';
        curRlo = r.getResult();
        continue;
      }
      if (isa<AddOp>(&o) || isa<SubOp>(&o) || isa<MulOp>(&o) ||
          isa<DivOp>(&o)) {
        const char *mn = isa<AddOp>(&o) ? "+" : isa<SubOp>(&o) ? "-"
                          : isa<MulOp>(&o)   ? "*"
                                             : "/";
        popAcc();
        popAcc();
        os << "  " << mn << typeLetter(o.getResult(0).getType()) << '\n';
        acc.push_back(o.getResult(0));
        continue;
      }
      if (isa<AndOp>(&o) || isa<OrOp>(&o) || isa<XOrOp>(&o)) {
        if (o.getResult(0).getType().isInteger(1)) {
          // bracket closer
          if (bracketClose.count(&o))
            os << "  )\n";
          else
            os << "  // unsupported i1 logic op\n";
          curRlo = o.getResult(0);
          continue;
        }
        // AW/OW/XOW for words, AD/OD/XOD for dwords
        std::string m = isa<AndOp>(&o) ? "A" : isa<OrOp>(&o) ? "O" : "XO";
        m += o.getResult(0).getType().isInteger(32) ? "D" : "W";
        popAcc();
        popAcc();
        os << "  " << m << '\n';
        acc.push_back(o.getResult(0));
        continue;
      }
      if (isa<NegOp>(&o)) {
        // no S7 mnemonic; LowerToAWL lowers negate as `0 - x` instead
        os << "  // unsupported awl.neg\n";
        continue;
      }
      if (isa<CastOp>(&o)) {
        // implicit in the mnemonic letter / operand size; not printed
        continue;
      }
      if (isa<CmpEqOp>(&o) || isa<CmpNeOp>(&o) || isa<CmpLtOp>(&o) ||
          isa<CmpLeOp>(&o) || isa<CmpGtOp>(&o) || isa<CmpGeOp>(&o)) {
        const char *mn = isa<CmpEqOp>(&o) ? "==" : isa<CmpNeOp>(&o) ? "<>"
                          : isa<CmpLtOp>(&o)   ? "<"
                          : isa<CmpLeOp>(&o)   ? "<="
                          : isa<CmpGtOp>(&o)   ? ">"
                                               : ">=";
        popAcc();
        popAcc();
        os << "  " << mn << typeLetter(o.getOperand(0).getType()) << '\n';
        curRlo = o.getResult(0);
        continue;
      }
      if (auto j = dyn_cast<JumpOp>(&o)) {
        os << "  SPA " << j.getLabel() << '\n';
        continue;
      }
      if (auto j = dyn_cast<JumpIfOp>(&o)) {
        os << "  SPB " << j.getLabel() << '\n';
        continue;
      }
      if (auto j = dyn_cast<JumpIfNotOp>(&o)) {
        os << "  SPN " << j.getLabel() << '\n';
        continue;
      }
      if (auto l = dyn_cast<LabelOp>(&o)) {
        os << l.getName() << ":\n";
        continue;
      }
      if (auto c = dyn_cast<CallOp>(&o)) {
        llvm::StringRef callee = c.getCallee();
        if (callee.startswith("FC"))
          os << "  UC FC " << callee.drop_front(2) << '\n';
        else
          os << "  // unsupported call @" << callee << '\n';
        continue;
      }
      if (auto ts = dyn_cast<TimerStartOp>(&o)) {
        static const char *mn[] = {"SP", "SE", "SS", "SV", "SI"};
        int mode = ts.getMode();
        popAcc(); // the S5TIME duration
        os << "  " << (mode >= 0 && mode < 5 ? mn[mode] : "SP") << ' '
           << directTextOf(ts.getTimer()) << '\n';
        continue;
      }
      if (auto co = dyn_cast<CounterOp>(&o)) {
        int op = co.getOp();
        const char *mn = op == 0 ? "ZV" : op == 1 ? "ZR" : op == 2 ? "S" : "R";
        if (op == 2)
          popAcc(); // S C n takes the count from ACC1
        os << "  " << mn << ' ' << directTextOf(co.getCounter()) << '\n';
        continue;
      }
      if (isa<NopOp>(&o)) {
        os << "  NOP 0\n";
        continue;
      }
      if (isa<BeOp>(&o)) {
        os << "  BE\n";
        continue;
      }
      os << "  // unsupported op " << o.getName() << '\n';
    }
  }
};

} // namespace

void printAwlText(mlir::ModuleOp module, llvm::raw_ostream &os) {
  for (Operation &op : module.getBody()->getOperations()) {
    auto block = dyn_cast<BlockOp>(&op);
    if (!block)
      continue;
    if (block.getSymName() == "_mlir_ciface_OB1")
      continue; // JIT wrapper, re-synthesized by the frontend
    TextPrinter(os).printBlock(block);
  }
}

} // namespace awl
} // namespace mlir
