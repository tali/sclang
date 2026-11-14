//===- AwlGen.cpp - AWL (STL) lexer, parser and MLIR codegen --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classic S7-300/400 AWL frontend (ADR 003, docs/design/awl-frontend.md):
// one instruction per line, `:` labels, `//` comments, blocks OB/FB/FC n
// ended by BE. The two-accumulator model is explicit SSA values in codegen:
// `awl.load` pushes, arithmetic pops two and pushes one, RLO is an i1 SSA
// value (U/O/NOT/A(/O(), =, S, R), jumps are awl.jump* with string labels.
//
//===----------------------------------------------------------------------===//

#include "sclang/AwlGen/AwlGen.h"

#include "sclang/AwlDialect/AwlOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstring>

using namespace mlir;
using namespace mlir::awl;

namespace sclang {
namespace {

// MARK: Lexer

enum class Tok {
  eof,
  ident,   // [A-Za-z_][A-Za-z0-9_]*
  integer, // decimal, or base-prefixed 2#.. 8#.. 16#..
  real,    // digits.digits
  colon,       // :
  lparen,      // (
  rparen,      // )
  dot,         // .
  comma,       // ,
  minus,       // -
  plus,        // +
  times,       // *
  divide,      // / (single; // is a comment)
  assign,      // =
  eq,          // ==
  ne,          // <>
  lt, le, gt, ge,
};

class Lexer {
public:
  Lexer(llvm::StringRef src, llvm::StringRef filename)
      : cur(src), file(filename) {}

  Tok getCurToken() const { return cur_.tok; }
  Tok peekToken() const { return ahead_.tok; }
  llvm::StringRef getIdent() const { return cur_.ident; }
  int64_t getInteger() const { return cur_.intVal; }
  double getReal() const { return cur_.realVal; }
  int line() const { return lineNo; }
  int col() const { return colNo; }

  Tok next() {
    cur_ = ahead_;
    if (cur_.tok != Tok::eof)
      ahead_ = scan();
    return cur_.tok;
  }
  // Prime the two-token lookahead.
  void start() {
    cur_ = scan();
    ahead_ = scan();
  }

private:
  struct Token {
    Tok tok = Tok::eof;
    llvm::StringRef ident;
    int64_t intVal = 0;
    double realVal = 0;
  };

  Token scan() {
    Token t;
    // skip whitespace and // comments
    while (true) {
      while (!cur.empty() && isspace(cur.front())) {
        if (cur.front() == '\n') {
          ++lineNo;
          colNo = 0;
        }
        ++colNo;
        cur = cur.drop_front();
      }
      if (cur.starts_with("//")) {
        while (!cur.empty() && cur.front() != '\n')
          cur = cur.drop_front();
        continue;
      }
      break;
    }
    if (cur.empty())
      return t; // eof
    char c = cur.front();
    // S5T#<value><base> literal, e.g. S5T#2S or S5T#500MS -> S5TIME-encoded
    // integer (bits 12-15 time base, bits 0-11 BCD value)
    if (cur.size() >= 4 && tolower((unsigned char)cur[0]) == 's' &&
        tolower((unsigned char)cur[1]) == '5' &&
        tolower((unsigned char)cur[2]) == 't' && cur[3] == '#') {
      cur = cur.drop_front(4);
      std::string s;
      while (!cur.empty() && isalnum((unsigned char)cur.front())) {
        s += (char)toupper((unsigned char)cur.front());
        cur = cur.drop_front();
      }
      // base suffix: MS -> 10ms (code 0), S -> 1s (code 2); the leading
      // digits are the (decimal, <= 999) value, BCD-encoded into bits 0-11
      static const char *suffixes[] = {"MS", "S"};
      static const int codes[] = {0, 2};
      int code = -1;
      size_t valLen = 0;
      for (int i = 0; i < 2; ++i) {
        size_t n = strlen(suffixes[i]);
        if (s.size() > n && s.compare(s.size() - n, n, suffixes[i]) == 0) {
          code = codes[i];
          valLen = s.size() - n;
          break;
        }
      }
      int64_t val = -1;
      if (code >= 0 && valLen > 0) {
        val = 0;
        for (size_t i = 0; i < valLen; ++i) {
          if (!isdigit((unsigned char)s[i]))
            val = -1;
          else
            val = val >= 0 ? val * 10 + (s[i] - '0') : -1;
        }
      }
      if (val < 0 || val > 999) {
        llvm::errs() << file << ":" << lineNo << ":" << colNo
                     << ": bad S5TIME literal 'S5T#" << s << "'\n";
        t.tok = Tok::eof;
        return t;
      }
      int64_t bcd = ((val / 100) << 8) | (((val / 10) % 10) << 4) | (val % 10);
      t.intVal = ((int64_t)code << 12) | bcd;
      t.tok = Tok::integer;
      return t;
    }
    if (isalpha(c) || c == '_') {
      auto begin = cur.begin();
      while (!cur.empty() && (isalnum(cur.front()) || cur.front() == '_'))
        cur = cur.drop_front();
      t.ident = llvm::StringRef(begin, cur.begin() - begin);
      t.tok = Tok::ident;
      return t;
    }
    if (isdigit(c)) {
      int64_t base = 0;
      auto readDigits = [&](int b) -> int64_t {
        int64_t v = 0;
        while (!cur.empty() && isdigit(cur.front())) {
          v = v * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        if (b == 16) {
          while (!cur.empty() && isxdigit(cur.front())) {
            char h = cur.front();
            int d = isdigit(h) ? h - '0' : tolower(h) - 'a' + 10;
            v = v * 16 + d;
            cur = cur.drop_front();
          }
        } else if (b == 8 || b == 2) {
          while (!cur.empty() && cur.front() >= '0' &&
                 cur.front() < (char)('0' + b)) {
            v = v * b + (cur.front() - '0');
            cur = cur.drop_front();
          }
        }
        return v;
      };
      // read the leading decimal number (also the base for base# literals)
      int64_t lead = 0;
      while (!cur.empty() && isdigit(cur.front())) {
        lead = lead * 10 + (cur.front() - '0');
        cur = cur.drop_front();
      }
      if (!cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        base = lead;
        t.intVal = readDigits((int)base);
        t.tok = Tok::integer;
        return t;
      }
      if (!cur.empty() && cur.front() == '.' && cur.size() > 1 &&
          isdigit(cur[1])) {
        cur = cur.drop_front();
        std::string s = std::to_string(lead) + ".";
        while (!cur.empty() && isdigit(cur.front())) {
          s += cur.front();
          cur = cur.drop_front();
        }
        t.realVal = std::stod(s);
        t.tok = Tok::real;
        return t;
      }
      t.intVal = lead;
      t.tok = Tok::integer;
      return t;
    }
    cur = cur.drop_front();
    switch (c) {
    case ':': t.tok = Tok::colon; return t;
    case '(': t.tok = Tok::lparen; return t;
    case ')': t.tok = Tok::rparen; return t;
    case '.': t.tok = Tok::dot; return t;
    case ',': t.tok = Tok::comma; return t;
    case '-': t.tok = Tok::minus; return t;
    case '+': t.tok = Tok::plus; return t;
    case '*': t.tok = Tok::times; return t;
    case '/': t.tok = Tok::divide; return t;
    case '=':
      if (!cur.empty() && cur.front() == '=') {
        cur = cur.drop_front();
        t.tok = Tok::eq;
      } else {
        t.tok = Tok::assign;
      }
      return t;
    case '<':
      if (!cur.empty() && cur.front() == '>') {
        cur = cur.drop_front();
        t.tok = Tok::ne;
      } else if (!cur.empty() && cur.front() == '=') {
        cur = cur.drop_front();
        t.tok = Tok::le;
      } else {
        t.tok = Tok::lt;
      }
      return t;
    case '>':
      if (!cur.empty() && cur.front() == '=') {
        cur = cur.drop_front();
        t.tok = Tok::ge;
      } else {
        t.tok = Tok::gt;
      }
      return t;
    default:
      llvm::errs() << file << ":" << lineNo << ":" << colNo
                   << ": unexpected character '" << c << "'\n";
      t.tok = Tok::eof;
      return t;
    }
  }

  llvm::StringRef cur;
  llvm::StringRef file;
  Token cur_;
  Token ahead_;
  int lineNo = 1;
  int colNo = 0;
};

// MARK: Parser + codegen

class AwlGenImpl {
public:
  AwlGenImpl(mlir::MLIRContext &context, llvm::StringRef source,
             llvm::StringRef filename)
      : builder(&context), lexer(source, filename), file(filename) {}

  mlir::OwningOpRef<mlir::ModuleOp> run() {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    lexer.start();
    if (!parseModule())
      return nullptr;
    // The JIT driver invokes "_mlir_ciface_OB1"; synthesize the wrapper for
    // OB 1 (same convention as the SCL tests).
    if (sawOB1) {
      auto wrapLoc = loc();
      builder.setInsertionPointToEnd(module.getBody());
      auto wrap = builder.create<awl::BlockOp>(
          wrapLoc, builder.getStringAttr("_mlir_ciface_OB1"));
      auto *entry = builder.createBlock(&wrap.getBody());
      builder.setInsertionPointToStart(entry);
      SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), "OB1")));
      builder.create<awl::CallOp>(wrapLoc, mlir::TypeRange{}, mlir::ValueRange{},
                                  attrs);
      builder.create<awl::BeOp>(wrapLoc);
    }
    if (failed(mlir::verify(module))) {
      module.emitError("module verification error");
      return nullptr;
    }
    return module;
  }

private:
  mlir::OpBuilder builder;
  Lexer lexer;
  llvm::StringRef file;
  mlir::ModuleOp module;

  // Accumulator model (per block): ACC stack and RLO.
  llvm::SmallVector<mlir::Value, 4> acc;
  mlir::Value rlo;
  bool rloActive = false;
  struct Bracket {
    mlir::Value savedRlo;
    bool isOr;
  };
  llvm::SmallVector<Bracket, 4> brackets;
  bool sawOB1 = false;

  mlir::Location loc() {
    return mlir::FileLineColLoc::get(builder.getStringAttr(file), lexer.line(),
                                     lexer.col());
  }

  bool error(const llvm::Twine &msg) {
    llvm::errs() << file << ":" << lexer.line() << ":" << lexer.col() << ": "
                 << msg << "\n";
    return false;
  }

  std::string upperIdent() {
    std::string s(lexer.getIdent());
    for (auto &c : s)
      c = toupper(c);
    return s;
  }

  bool parseModule() {
    while (lexer.getCurToken() != Tok::eof) {
      if (!parseBlock())
        return false;
    }
    return true;
  }

  // block := ("OB"|"FB"|"FC") number instruction* "BE"
  bool parseBlock() {
    if (lexer.getCurToken() != Tok::ident)
      return error("expected OB/FB/FC block header");
    std::string kind = upperIdent();
    if (kind != "OB" && kind != "FB" && kind != "FC")
      return error("expected OB/FB/FC block header, got '" + kind + "'");
    lexer.next();
    if (lexer.getCurToken() != Tok::integer)
      return error("expected block number after " + kind);
    int64_t num = lexer.getInteger();
    lexer.next();
    std::string name = kind + std::to_string(num);
    if (kind == "OB" && num == 1)
      sawOB1 = true;

    auto blockLoc = loc();
    builder.setInsertionPointToEnd(module.getBody());
    auto blockOp =
        builder.create<awl::BlockOp>(blockLoc, builder.getStringAttr(name));
    auto *entry = builder.createBlock(&blockOp.getBody());
    builder.setInsertionPointToStart(entry);
    acc.clear();
    rlo = nullptr;
    rloActive = false;
    brackets.clear();

    while (true) {
      // optional label: ident ':'
      if (lexer.getCurToken() == Tok::ident &&
          lexer.peekToken() == Tok::colon) {
        auto label = upperIdent();
        lexer.next(); // ident
        lexer.next(); // colon
        builder.create<awl::LabelOp>(loc(), builder.getStringAttr(label));
        // A label starts a new statement (ADR 003): the accumulator stack and
        // RLO do not cross the label boundary, so the next U/O/NOT starts
        // from the neutral RLO instead of an SSA value from a predecessor
        // block (which would not dominate the label's block after the
        // lowering splits at labels).
        acc.clear();
        rlo = nullptr;
        rloActive = false;
        brackets.clear();
        continue;
      }
      if (lexer.getCurToken() == Tok::eof)
        return error("missing BE in block " + name);
      if (!parseInstruction(name))
        return false;
      if (blockEnded)
        break;
    }
    blockEnded = false;
    return true;
  }

  bool blockEnded = false;

  bool parseInstruction(const std::string &blockName) {
    auto t = lexer.getCurToken();
    if (t == Tok::ident) {
      std::string m = upperIdent();
      lexer.next();
      return parseIdentInstruction(m, blockName);
    }
    if (t == Tok::plus || t == Tok::minus || t == Tok::times ||
        t == Tok::divide) {
      // +I -I *I /I +D ... : operator followed by I/D/R letter
      ArithKind k = t == Tok::plus ? ArithKind::Add
                    : t == Tok::minus ? ArithKind::Sub
                    : t == Tok::times ? ArithKind::Mul
                                      : ArithKind::Div;
      lexer.next();
      if (lexer.getCurToken() != Tok::ident)
        return error("expected I/D/R after arithmetic operator");
      std::string w = upperIdent();
      lexer.next();
      if (w != "I" && w != "D" && w != "R")
        return error("expected I/D/R after arithmetic operator, got '" + w + "'");
      return parseArith(k, w[0]);
    }
    if (t == Tok::eq || t == Tok::ne || t == Tok::lt || t == Tok::le ||
        t == Tok::gt || t == Tok::ge) {
      Tok cmp = t;
      lexer.next();
      if (lexer.getCurToken() != Tok::ident)
        return error("expected I/D/R after comparison operator");
      std::string w = upperIdent();
      lexer.next();
      if (w != "I" && w != "D" && w != "R")
        return error("expected I/D/R after comparison, got '" + w + "'");
      return parseCompare(cmp, w[0]);
    }
    if (t == Tok::assign) {
      lexer.next();
      return parseAssign();
    }
    if (t == Tok::rparen) {
      lexer.next();
      return parseCloseBracket();
    }
    return error("expected instruction");
  }

  bool parseIdentInstruction(const std::string &m, const std::string &blockName) {
    if (m == "BE") {
      builder.create<awl::BeOp>(loc());
      blockEnded = true;
      return true;
    }
    if (m == "NOP") {
      if (lexer.getCurToken() == Tok::integer)
        lexer.next(); // NOP 0 / NOP 1
      builder.create<awl::NopOp>(loc());
      rloActive = false;
      return true;
    }
    if (m == "L")
      return parseLoad();
    if (m == "T")
      return parseStore();
    if (m == "U" || m == "UN" || m == "O" || m == "ON")
      return parseRloLogic(m);
    if (m == "NOT") {
      if (!ensureRlo())
        return false;
      rlo = builder.create<awl::NotOp>(loc(), builder.getI1Type(), rlo);
      return true;
    }
    if (m == "A" || m == "O") {
      if (lexer.getCurToken() != Tok::lparen)
        return error(m + " must be followed by (");
      lexer.next();
      brackets.push_back({rlo, m == "O"});
      rlo = builder.create<awl::ConstOp>(loc(), builder.getI1Type(),
                                          builder.getIntegerAttr(builder.getI1Type(), 1));
      rloActive = true;
      return true;
    }
    if (m == "S")
      return parseSetReset(true);
    if (m == "R")
      return parseSetReset(false);
    if (m == "AW" || m == "OW" || m == "XOW" || m == "AD" || m == "OD" ||
        m == "XOD")
      return parseWordLogic(m);
    if (m == "SP" || m == "SE" || m == "SS" || m == "SV" || m == "SI")
      return parseTimerStart(m);
    if (m == "ZV" || m == "ZR")
      return parseCounterZW(m == "ZV");
    if (m == "SPA" || m == "SPB" || m == "SPN")
      return parseJump(m);
    if (m == "CALL" || m == "UC" || m == "CC")
      return parseCall(m == "CC");
    return error("unknown instruction '" + m + "' in block " + blockName);
  }

  // MARK: accumulator helpers

  mlir::Type widthType(char w) {
    switch (w) {
    case 'I': return builder.getI16Type();
    case 'D': return builder.getI32Type();
    case 'R': return builder.getF32Type();
    }
    return nullptr;
  }

  mlir::Value castTo(mlir::Value v, mlir::Type t) {
    if (v.getType() == t)
      return v;
    return builder.create<awl::CastOp>(loc(), t, v);
  }

  bool ensureRlo() {
    if (!rloActive) {
      // a new logic chain starts with RLO = 1
      rlo = builder.create<awl::ConstOp>(loc(), builder.getI1Type(),
                                          builder.getIntegerAttr(builder.getI1Type(), 1));
      rloActive = true;
    }
    return true;
  }

  mlir::Value popAcc() {
    if (acc.empty()) {
      error("accumulator stack underflow");
      return nullptr;
    }
    mlir::Value v = acc.back();
    acc.pop_back();
    return v;
  }

  // MARK: operands

  static bool isAreaLetter(char c) {
    return c == 'E' || c == 'A' || c == 'M' || c == 'T' || c == 'Z';
  }
  static mlir::scl::DirectArea areaOf(char c) {
    switch (c) {
    case 'E': return mlir::scl::DirectArea::Input;
    case 'A': return mlir::scl::DirectArea::Output;
    case 'M': return mlir::scl::DirectArea::Marker;
    case 'T': return mlir::scl::DirectArea::Timer;
    case 'Z': return mlir::scl::DirectArea::Counter;
    }
    return mlir::scl::DirectArea::Marker;
  }

  // Parse an operand: E/A/M/T/Z [B|W|D] byte [.bit] or DBn.DB[XBWD] byte [.bit]
  // `bitContext`: for T/Z operands without a size letter, `U T0`-style bit
  // access yields the status bit (width 1, bit 0); otherwise the 16-bit value.
  mlir::Value parseOperand(bool bitContext = false) {
    if (lexer.getCurToken() != Tok::ident)
      return (error("expected operand"), nullptr);
    std::string w = upperIdent();
    lexer.next();

    // DBn . DB[XBWD] byte [.bit]   (DB without a number = the instance DB
    // of the enclosing FB, i.e. db = 0)
    if (w.rfind("DB", 0) == 0 &&
        (w.size() == 2 ||
         llvm::all_of(llvm::drop_begin(w, 2),
                      [](char c) { return isdigit(c); }))) {
      int32_t db = w.size() > 2 ? atoi(w.c_str() + 2) : 0;
      if (lexer.getCurToken() != Tok::dot)
        return (error("expected '.' after DB number"), nullptr);
      lexer.next();
      if (lexer.getCurToken() != Tok::ident)
        return (error("expected DBX/DBB/DBW/DBD"), nullptr);
      std::string dw = upperIdent();
      lexer.next();
      if (dw.size() < 3 || dw.rfind("DB", 0) != 0)
        return (error("expected DBX/DBB/DBW/DBD, got '" + dw + "'"), nullptr);
      int32_t width;
      switch (dw[2]) {
      case 'X': width = 1; break;
      case 'B': width = 8; break;
      case 'W': width = 16; break;
      case 'D': width = 32; break;
      default: return (error("expected DBX/DBB/DBW/DBD, got '" + dw + "'"), nullptr);
      }
      int32_t byte = -1;
      if (dw.size() > 3)
        byte = atoi(dw.c_str() + 3);
      double realBit = -1;
      if (byte < 0) {
        if (lexer.getCurToken() == Tok::integer) {
          byte = (int32_t)lexer.getInteger();
          lexer.next();
        } else if (lexer.getCurToken() == Tok::real) {
          // e.g. DB1.DBX 0.0 lexes as a real: split into byte and bit
          double r = lexer.getReal();
          byte = (int32_t)r;
          realBit = r - byte;
          lexer.next();
        } else {
          return (error("expected byte address"), nullptr);
        }
      }
      int32_t bit = 0;
      bool hasBit = false;
      if (realBit >= 0) {
        bit = (int32_t)(realBit * 10 + 0.5);
        hasBit = true;
      } else if (lexer.getCurToken() == Tok::dot) {
        lexer.next();
        if (lexer.getCurToken() != Tok::integer)
          return (error("expected bit number"), nullptr);
        bit = (int32_t)lexer.getInteger();
        lexer.next();
        hasBit = true;
      }
      if (width == 1 && !hasBit)
        return (error("DBX bit access requires a bit number"), nullptr);
      if (width != 1 && hasBit)
        return (error("only DBX takes a bit number"), nullptr);
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::DataBlock,
                                            db, byte, bit, width);
    }

    // E/A/M/T/Z [B|W|D] [byte] [.bit]
    if (w.size() < 1 || !isAreaLetter(w[0]))
      return (error("expected operand, got '" + w + "'"), nullptr);
    auto area = areaOf(w[0]);
    size_t p = 1;
    int32_t width = 1;
    if (p < w.size() && (w[p] == 'B' || w[p] == 'W' || w[p] == 'D')) {
      width = w[p] == 'B' ? 8 : w[p] == 'W' ? 16 : 32;
      ++p;
    }
    int32_t byte = -1;
    if (p < w.size()) {
      for (size_t i = p; i < w.size(); ++i)
        if (!isdigit(w[i]))
          return (error("bad operand '" + w + "'"), nullptr);
      byte = atoi(w.c_str() + p);
    }
    double realBit = -1;
    if (byte < 0) {
      if (lexer.getCurToken() == Tok::integer) {
        byte = (int32_t)lexer.getInteger();
        lexer.next();
      } else if (lexer.getCurToken() == Tok::real) {
        // e.g. M 10.3 lexes as a real: split into byte and bit
        double r = lexer.getReal();
        byte = (int32_t)r;
        realBit = r - byte;
        lexer.next();
      } else {
        return (error("expected byte address in operand '" + w + "'"), nullptr);
      }
    }
    int32_t bit = 0;
    bool hasBit = false;
    if (realBit >= 0) {
      bit = (int32_t)(realBit * 10 + 0.5);
      hasBit = true;
    } else if (lexer.getCurToken() == Tok::dot) {
      lexer.next();
      if (lexer.getCurToken() != Tok::integer)
        return (error("expected bit number"), nullptr);
      bit = (int32_t)lexer.getInteger();
      lexer.next();
      hasBit = true;
    }
    // T/Z (timer/counter) operands: `T 0` / `Z 0` without a size letter is the
    // 16-bit value; in a bit context (`U T0`) it is the status bit 0.
    if ((area == mlir::scl::DirectArea::Timer ||
         area == mlir::scl::DirectArea::Counter) &&
        width == 1 && !hasBit) {
      if (bitContext) {
        bit = 0;
        hasBit = true;
      } else {
        width = 16;
      }
    }
    if (width == 1 && !hasBit)
      return (error("bit access '" + w + "' requires a bit number (.bit)"), nullptr);
    if (width != 1 && hasBit)
      return (error("only bit access takes a bit number"), nullptr);
    return builder.create<awl::DirectOp>(loc(), area, 0, byte, bit, width);
  }

  static bool isBit(mlir::Value v) {
    return v.getType().cast<mlir::scl::DirectType>().getWidth() == 1;
  }

  mlir::Value parseBitOperand() {
    mlir::Value v = parseOperand(true);
    if (!v)
      return nullptr;
    if (!isBit(v))
      return (error("expected a bit operand (e.g. M 0.0)"), nullptr);
    return v;
  }

  mlir::Value parseValueOperand() {
    mlir::Value v = parseOperand(false);
    if (!v)
      return nullptr;
    if (isBit(v))
      return (error("cannot use a bit operand here (need byte/word/dword)"),
              nullptr);
    return v;
  }

  // MARK: instructions

  mlir::Value parseConst() {
    bool neg = false;
    if (lexer.getCurToken() == Tok::minus) {
      lexer.next();
      neg = true;
    }
    if (lexer.getCurToken() == Tok::integer) {
      int64_t v = lexer.getInteger();
      lexer.next();
      if (neg)
        v = -v;
      mlir::Type ty;
      int64_t val;
      if (v >= -32768 && v <= 32767) {
        ty = builder.getI16Type();
        val = v;
      } else if (v >= 0 && v <= 65535) {
        ty = builder.getI16Type();
        val = v - 65536; // 16-bit bit pattern
      } else {
        ty = builder.getI32Type();
        val = v;
      }
      return builder.create<awl::ConstOp>(
          loc(), ty, builder.getIntegerAttr(ty, val));
    }
    if (lexer.getCurToken() == Tok::real) {
      double v = lexer.getReal();
      lexer.next();
      if (neg)
        v = -v;
      auto ty = builder.getF32Type();
      return builder.create<awl::ConstOp>(loc(), ty,
                                          builder.getF32FloatAttr((float)v));
    }
    error("expected constant");
    return nullptr;
  }

  bool parseLoad() {
    // L <constant> | L <operand>
    mlir::Value v;
    if (lexer.getCurToken() == Tok::integer ||
        lexer.getCurToken() == Tok::real || lexer.getCurToken() == Tok::minus) {
      v = parseConst();
    } else {
      mlir::Value addr = parseValueOperand();
      if (!addr)
        return false;
      v = builder.create<awl::LoadOp>(loc(), addr);
    }
    if (!v)
      return false;
    acc.push_back(v);
    rloActive = false;
    return true;
  }

  bool parseStore() {
    mlir::Value addr = parseValueOperand();
    if (!addr)
      return false;
    mlir::Value v = popAcc();
    if (!v)
      return false;
    builder.create<awl::StoreOp>(loc(), v, addr);
    rloActive = false;
    return true;
  }

  bool parseRloLogic(const std::string &m) {
    mlir::Value addr = parseBitOperand();
    if (!addr)
      return false;
    if (!ensureRlo())
      return false;
    bool isOr = (m == "O" || m == "ON");
    bool neg = (m == "UN" || m == "ON");
    auto i1 = builder.getI1Type();
    llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
    if (neg)
      attrs.push_back(
          builder.getNamedAttr("negate", builder.getUnitAttr()));
    if (isOr)
      rlo = builder.create<awl::RloOrOp>(loc(), mlir::TypeRange{i1},
                                         mlir::ValueRange{rlo, addr}, attrs);
    else
      rlo = builder.create<awl::RloAndOp>(loc(), mlir::TypeRange{i1},
                                          mlir::ValueRange{rlo, addr}, attrs);
    return true;
  }

  bool parseCloseBracket() {
    if (brackets.empty())
      return error("unbalanced )");
    Bracket b = brackets.pop_back_val();
    if (!rloActive)
      return error("empty bracket expression");
    mlir::Value group = rlo;
    auto i1 = builder.getI1Type();
    rlo = b.isOr ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, b.savedRlo, group)
                 : (mlir::Value)builder.create<awl::AndOp>(loc(), i1, b.savedRlo, group);
    rloActive = true;
    return true;
  }

  bool parseAssign() {
    mlir::Value addr = parseBitOperand();
    if (!addr)
      return false;
    if (!ensureRlo())
      return false;
    builder.create<awl::AssignOp>(loc(), rlo, addr);
    rloActive = false;
    return true;
  }

  bool parseSetReset(bool set) {
    mlir::Value v = parseOperand(false);
    if (!v)
      return false;
    auto dt = v.getType().cast<mlir::scl::DirectType>();
    // `S C n` sets the counter to ACC1, `R C n` resets it to 0
    if (dt.getArea() == mlir::scl::DirectArea::Counter && dt.getWidth() != 1) {
      mlir::Value val;
      int op;
      if (set) {
        val = popAcc();
        if (!val)
          return false;
        op = 2; // SCL_RT_COUNTER_SET
      } else {
        auto i16 = builder.getI16Type();
        val = builder.create<awl::ConstOp>(loc(), i16,
                                            builder.getIntegerAttr(i16, 0));
        op = 3; // SCL_RT_COUNTER_RESET
      }
      llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr("op", builder.getI32IntegerAttr(op)));
      builder.create<awl::CounterOp>(loc(), mlir::TypeRange{},
                                     mlir::ValueRange{val, v}, attrs);
      auto i1 = builder.getI1Type();
      rlo = builder.create<awl::ConstOp>(
          loc(), i1, builder.getIntegerAttr(i1, set ? 1 : 0));
      rloActive = true;
      return true;
    }
    if (!isBit(v))
      return error("expected a bit operand (e.g. M 0.0)");
    auto i1 = builder.getI1Type();
    if (set)
      rlo = builder.create<awl::SetOp>(loc(), i1, v);
    else
      rlo = builder.create<awl::ResetOp>(loc(), i1, v);
    rloActive = true;
    return true;
  }

  // SP/SE/SS/SV/SI T n: start timer n with ACC1 = S5TIME duration
  bool parseTimerStart(const std::string &m) {
    mlir::Value dur = popAcc();
    if (!dur)
      return false;
    mlir::Value t = parseOperand(false);
    if (!t)
      return false;
    auto dt = t.getType().cast<mlir::scl::DirectType>();
    if (dt.getArea() != mlir::scl::DirectArea::Timer)
      return error("timer instruction " + m + " needs a T operand");
    int mode = m == "SP" ? 0 : m == "SE" ? 1 : m == "SS" ? 2 : m == "SV" ? 3 : 4;
    llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
    attrs.push_back(
        builder.getNamedAttr("mode", builder.getI32IntegerAttr(mode)));
    builder.create<awl::TimerStartOp>(loc(), mlir::TypeRange{},
                                      mlir::ValueRange{dur, t}, attrs);
    rloActive = false;
    return true;
  }

  // ZV/ZR Z n: increment/decrement counter n
  bool parseCounterZW(bool up) {
    mlir::Value c = parseOperand(false);
    if (!c)
      return false;
    auto dt = c.getType().cast<mlir::scl::DirectType>();
    if (dt.getArea() != mlir::scl::DirectArea::Counter)
      return error("ZV/ZR need a Z (counter) operand");
    auto i16 = builder.getI16Type();
    mlir::Value zero = builder.create<awl::ConstOp>(
        loc(), i16, builder.getIntegerAttr(i16, 0));
    llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
    attrs.push_back(builder.getNamedAttr(
        "op", builder.getI32IntegerAttr(up ? 0 : 1)));
    builder.create<awl::CounterOp>(loc(), mlir::TypeRange{},
                                   mlir::ValueRange{zero, c}, attrs);
    rloActive = false;
    return true;
  }

  enum class ArithKind { Add, Sub, Mul, Div };

  bool parseArith(ArithKind k, char w) {
    mlir::Type ty = widthType(w);
    mlir::Value b = popAcc();
    mlir::Value a = popAcc();
    if (!a || !b)
      return false;
    a = castTo(a, ty);
    b = castTo(b, ty);
    mlir::Value r;
    switch (k) {
    case ArithKind::Add: r = builder.create<awl::AddOp>(loc(), ty, a, b); break;
    case ArithKind::Sub: r = builder.create<awl::SubOp>(loc(), ty, a, b); break;
    case ArithKind::Mul: r = builder.create<awl::MulOp>(loc(), ty, a, b); break;
    case ArithKind::Div: r = builder.create<awl::DivOp>(loc(), ty, a, b); break;
    }
    acc.push_back(r);
    rloActive = false;
    return true;
  }

  bool parseWordLogic(const std::string &m) {
    // AW/OW/XOW operate on words (i16), AD/OD/XOD on double words (i32)
    auto ty = (m.size() == 3 && m[2] == 'D') ? builder.getI32Type()
                                             : builder.getI16Type();
    mlir::Value b = popAcc();
    mlir::Value a = popAcc();
    if (!a || !b)
      return false;
    a = castTo(a, ty);
    b = castTo(b, ty);
    mlir::Value r;
    if (m[0] == 'A')
      r = builder.create<awl::AndOp>(loc(), ty, a, b);
    else if (m[0] == 'O')
      r = builder.create<awl::OrOp>(loc(), ty, a, b);
    else
      r = builder.create<awl::XOrOp>(loc(), ty, a, b);
    acc.push_back(r);
    rloActive = false;
    return true;
  }

  bool parseCompare(Tok cmp, char w) {
    mlir::Type ty = widthType(w);
    if (acc.size() < 2)
      return error("comparison needs two accumulator values");
    mlir::Value a = castTo(acc[acc.size() - 2], ty); // ACC2
    mlir::Value b = castTo(acc[acc.size() - 1], ty); // ACC1
    auto i1 = builder.getI1Type();
    mlir::Value r;
    switch (cmp) {
    case Tok::eq: r = builder.create<awl::CmpEqOp>(loc(), i1, a, b); break;
    case Tok::ne: r = builder.create<awl::CmpNeOp>(loc(), i1, a, b); break;
    case Tok::lt: r = builder.create<awl::CmpLtOp>(loc(), i1, a, b); break;
    case Tok::le: r = builder.create<awl::CmpLeOp>(loc(), i1, a, b); break;
    case Tok::gt: r = builder.create<awl::CmpGtOp>(loc(), i1, a, b); break;
    case Tok::ge: r = builder.create<awl::CmpGeOp>(loc(), i1, a, b); break;
    default: return error("bad comparison");
    }
    rlo = r;
    rloActive = true;
    return true;
  }

  bool parseJump(const std::string &m) {
    if (lexer.getCurToken() != Tok::ident)
      return error("expected label after " + m);
    std::string label = upperIdent();
    lexer.next();
    if (m == "SPA") {
      builder.create<awl::JumpOp>(loc(), builder.getStringAttr(label));
    } else {
      if (!ensureRlo())
        return false;
      if (m == "SPB")
        builder.create<awl::JumpIfOp>(loc(), rlo, builder.getStringAttr(label));
      else
        builder.create<awl::JumpIfNotOp>(loc(), rlo, builder.getStringAttr(label));
    }
    rloActive = false;
    return true;
  }

  bool parseCall(bool cond) {
    // CALL/UC/CC FC n | CALL/UC/CC FB n, DB m ; CC calls only if RLO = 1
    mlir::Value rloVal;
    if (cond) {
      if (!ensureRlo())
        return false;
      rloVal = rlo;
    }
    if (lexer.getCurToken() != Tok::ident)
      return error("expected FC/FB after CALL");
    std::string kind = upperIdent();
    lexer.next();
    if (kind != "FC" && kind != "FB")
      return error("expected FC/FB after CALL, got '" + kind + "'");
    if (lexer.getCurToken() != Tok::integer)
      return error("expected block number after " + kind);
    int64_t num = lexer.getInteger();
    lexer.next();
    std::string callee = kind + std::to_string(num);
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), callee)));
    if (kind == "FB") {
      // CALL FB n, DB m: the instance data block number
      if (lexer.getCurToken() != Tok::comma)
        return error("expected ',' after FB number (CALL FB n, DB m)");
      lexer.next();
      if (lexer.getCurToken() != Tok::ident || upperIdent() != "DB")
        return error("expected DB after ','");
      lexer.next();
      if (lexer.getCurToken() != Tok::integer)
        return error("expected DB number");
      int64_t db = lexer.getInteger();
      lexer.next();
      attrs.push_back(
          builder.getNamedAttr("db", builder.getI32IntegerAttr((int32_t)db)));
    }
    builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                cond ? mlir::ValueRange{rloVal}
                                     : mlir::ValueRange{},
                                attrs);
    if (!cond)
      rloActive = false;
    // CC leaves RLO unchanged (S7 semantics), so the chain stays active
    return true;
  }
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> awlGen(mlir::MLIRContext &context,
                                         llvm::StringRef source,
                                         llvm::StringRef filename) {
  return AwlGenImpl(context, source, filename).run();
}

} // namespace sclang
