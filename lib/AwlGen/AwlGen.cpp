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
  localvar, // #name (awlsim block-local variable)
  integer, // decimal, or base-prefixed 2#.. 8#.. 16#..
  real,    // digits.digits
  colon,       // :
  lparen,      // (
  rparen,      // )
  semi,        // ; statement separator
  lbracket,    // [
  rbracket,    // ]
  lbrace,      // {
  rbrace,      // }
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
  // line of the current token (lineNo itself is ahead due to lookahead)
  int curLine() const { return cur_.line; }
  bool curQuoted() const { return cur_.quoted; }
  bool curBcdLit() const { return cur_.bcdLit; }

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
  // Skip the rest of the current line (block header lines, directives) and
  // rescan both lookahead tokens (the primed ahead_ is stale afterwards).
  void nextLine() {
    while (!cur.empty() && cur.front() != '\n')
      cur = cur.drop_front();
    cur_ = scan();
    ahead_ = scan();
  }

private:
  struct Token {
    Tok tok = Tok::eof;
    llvm::StringRef ident;
    int64_t intVal = 0;
    double realVal = 0;
    bool quoted = false; // came from a "..." or '...' literal
    bool bcdLit = false; // C#... counter literal (raw BCD pattern)
    int line = 0;        // line the token starts on
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
    t.line = lineNo;
    char c = cur.front();
    // S5T#<value><base> literal, e.g. S5T#2S or S5T#500MS -> S5TIME-encoded
    // integer (bits 12-15 time base, bits 0-11 BCD value)
    if (cur.size() >= 4 && tolower((unsigned char)cur[0]) == 's' &&
        tolower((unsigned char)cur[1]) == '5' &&
        tolower((unsigned char)cur[2]) == 't' && cur[3] == '#') {
      cur = cur.drop_front(4);
      std::string s;
      while (!cur.empty() && (isalnum((unsigned char)cur.front()) ||
                              cur.front() == '_')) {
        if (cur.front() != '_') // awlsim: underscores in literals are ignored
          s += (char)toupper((unsigned char)cur.front());
        cur = cur.drop_front();
      }
      // components: <digits><unit> repeated, units MS (10ms), S (1s),
      // M (1min), H (1h); the total is re-encoded into the S5TIME base
      // (10ms/100ms/1s/10s) that fits the BCD value range <= 999
      int64_t total = 0; // in 10ms units
      bool ok = true;
      size_t i = 0;
      while (i < s.size() && ok) {
        int64_t num = 0;
        bool anyDigit = false;
        while (i < s.size() && isdigit((unsigned char)s[i])) {
          num = num * 10 + (s[i] - '0');
          anyDigit = true;
          ++i;
        }
        int64_t unit = -1;
        if (i + 1 < s.size() && s[i] == 'M' && s[i + 1] == 'S') {
          unit = 1; // 10ms
          i += 2;
        } else if (i < s.size() && s[i] == 'H') {
          unit = 360000;
          ++i;
        } else if (i < s.size() && s[i] == 'M') {
          unit = 6000;
          ++i;
        } else if (i < s.size() && s[i] == 'S') {
          unit = 100;
          ++i;
        }
        if (!anyDigit || unit < 0)
          ok = false;
        else
          total += num * unit;
      }
      if (!ok || total < 0 || total > 999LL * 1000) {
        llvm::errs() << file << ":" << lineNo << ":" << colNo
                     << ": bad S5TIME literal 'S5T#" << s << "'\n";
        t.tok = Tok::eof;
        return t;
      }
      static const int64_t steps[] = {1000, 100, 10, 1}; // 10s, 1s, 100ms, 10ms
      static const int codes[] = {3, 2, 1, 0};
      int code = 0;
      int64_t val = total;
      for (int k = 0; k < 4; ++k) {
        if (total % steps[k] == 0 && total / steps[k] <= 999) {
          code = codes[k];
          val = total / steps[k];
          break;
        }
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
      // awlsim named constants: __CNST_PI etc. (operands, not directives)
      if (t.ident.starts_with("__CNST")) {
        if (t.ident.contains("PI")) {
          t.realVal = 3.14159265358979;
          t.tok = Tok::real;
        } else if (t.ident.contains("NAN")) {
          t.realVal = __builtin_nan("");
          t.tok = Tok::real;
        } else {
          t.intVal = 0;
          t.tok = Tok::integer;
        }
        return t;
      }
      // awlsim test directives (__FEATURE, __STWRST, ...): skip the line.
      // __ASSERT... is kept as an ident — the parser lowers it to a runtime
      // oracle check (awl.assert), see parseAssert.
      if (t.ident.starts_with("__") && !t.ident.starts_with("__ASSERT") &&
          t.ident != "__ACCU" && t.ident != "__STW" && t.ident != "__AR" &&
          t.ident != "__DBR" && t.ident != "__SLEEP" &&
          t.ident != "__STWRST") {
        while (!cur.empty() && cur.front() != '\n')
          cur = cur.drop_front();
        return scan();
      }
      // DATE/TIME literals: D#1990-1-1 (days since 1990-01-01), T#2s (ms),
      // DT#... (not modelled, 0)
      if ((t.ident == "D" || t.ident == "T" || t.ident == "DT" ||
           t.ident == "TOD" || t.ident == "DATE" || t.ident == "TIME" ||
           t.ident == "TIME_OF_DAY") &&
          !cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        std::string s;
        while (!cur.empty() &&
               (isalnum((unsigned char)cur.front()) || cur.front() == ':' ||
                cur.front() == '.' || cur.front() == '-' ||
                cur.front() == '_')) {
          if (cur.front() != '_') // awlsim: underscores in literals are ignored
            s += (char)toupper((unsigned char)cur.front());
          cur = cur.drop_front();
        }
        int64_t v = 0;
        if (t.ident == "T" || t.ident == "TIME") {
          // components <digits><unit>, units MS/S/M/H -> total ms
          size_t i = 0;
          while (i < s.size()) {
            size_t before = i;
            int64_t num = 0;
            while (i < s.size() && isdigit((unsigned char)s[i]))
              num = num * 10 + (s[i++] - '0');
            int64_t unit = 0;
            if (i + 1 < s.size() && s[i] == 'M' && s[i + 1] == 'S') {
              unit = 1;
              i += 2;
            } else if (i < s.size() && s[i] == 'H') {
              unit = 3600000;
              ++i;
            } else if (i < s.size() && s[i] == 'M') {
              unit = 60000;
              ++i;
            } else if (i < s.size() && s[i] == 'S') {
              unit = 1000;
              ++i;
            }
            if (i == before)
              break; // no progress: avoid an infinite loop
            v += num * unit;
          }
        } else if (t.ident == "TOD" || t.ident == "TIME_OF_DAY") {
          // TOD#h:m:s.ms -> ms since midnight
          int h = 0, mi = 0, sec = 0, ms = 0;
          if (sscanf(s.c_str(), "%d:%d:%d.%d", &h, &mi, &sec, &ms) >= 3)
            v = ((int64_t)h * 3600 + mi * 60 + sec) * 1000 + ms;
        } else if (t.ident == "D" || t.ident == "DATE") {
          int y = 0, mo = 0, d = 0;
          if (sscanf(s.c_str(), "%d-%d-%d", &y, &mo, &d) == 3) {
            auto daysFromCivil = [](int yy, int m, int dd) -> int64_t {
              yy -= m <= 2;
              int64_t era = (yy >= 0 ? yy : yy - 399) / 400;
              unsigned yoe = (unsigned)(yy - era * 400);
              unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + dd - 1;
              unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
              return era * 146097 + (int64_t)doe - 719468;
            };
            v = daysFromCivil(y, mo, d) - daysFromCivil(1990, 1, 1);
          }
        }
        t.intVal = v;
        t.tok = Tok::integer;
        return t;
      }
      // pointer literal P#<byte>.<bit> (P#M 10.0 etc. in CALL params are
      // skipped with the parameter list): value = byte*8 + bit
      if (t.ident == "P" && !cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        if (!cur.empty() && cur.front() == '#') {
          // P##<ident>: pointer to a symbolic area — not modelled, 0
          cur = cur.drop_front();
          while (!cur.empty() && (isalnum(cur.front()) || cur.front() == '_'))
            cur = cur.drop_front();
          t.intVal = 0;
          t.tok = Tok::integer;
          return t;
        }
        // P#"DB".VAR / P#DBX 4.0: symbolic pointer target — skip the name
        auto skipQuoted = [&]() {
          if (!cur.empty() && cur.front() == '"') {
            cur = cur.drop_front();
            while (!cur.empty() && cur.front() != '"' && cur.front() != '\n')
              cur = cur.drop_front();
            if (!cur.empty() && cur.front() == '"')
              cur = cur.drop_front();
          }
        };
        skipQuoted();
        while (!cur.empty() && cur.front() == '.') {
          cur = cur.drop_front();
          skipQuoted();
          while (!cur.empty() &&
                 (isalnum((unsigned char)cur.front()) || cur.front() == '_'))
            cur = cur.drop_front();
        }
        // optional area letter and whitespace: P#E 2.1 / P#M 0.0
        while (!cur.empty() && (cur.front() == ' ' || cur.front() == '\t' ||
                                isalpha((unsigned char)cur.front())))
          cur = cur.drop_front();
        int64_t byte = 0;
        while (!cur.empty() && isdigit(cur.front())) {
          byte = byte * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        int64_t bit = 0;
        if (!cur.empty() && cur.front() == '.') {
          cur = cur.drop_front();
          if (!cur.empty() && isdigit(cur.front())) {
            bit = cur.front() - '0';
            cur = cur.drop_front();
          }
        }
        // trailing symbolic member: P#DB42.DTVAR1 (the '.' may already
        // have been consumed by the bit check above)
        while (!cur.empty() &&
               (isalnum((unsigned char)cur.front()) || cur.front() == '_' ||
                cur.front() == '.' || cur.front() == '[')) {
          if (cur.front() == '.') {
            cur = cur.drop_front();
            skipQuoted();
          } else if (cur.front() == '[') {
            int depth = 0;
            do {
              if (cur.front() == '[')
                ++depth;
              else if (cur.front() == ']')
                --depth;
              cur = cur.drop_front();
            } while (depth > 0 && !cur.empty());
          } else
            cur = cur.drop_front();
        }
        t.intVal = byte * 8 + bit;
        t.tok = Tok::integer;
        return t;
      }
      // local-data immediate L#<value> (awlsim), e.g. L L#-1
      if (t.ident == "L" && !cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        bool neg = false;
        if (!cur.empty() && cur.front() == '-') {
          cur = cur.drop_front();
          neg = true;
        }
        int64_t v = 0;
        while (!cur.empty() && isdigit(cur.front())) {
          v = v * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        t.intVal = neg ? -v : v;
        t.tok = Tok::integer;
        return t;
      }
      // C#123: BCD counter literal — the value is the BCD pattern as a
      // binary number (0x123 = 291), decoded by S Z n / compared raw
      if (t.ident == "C" && !cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        int64_t v = 0;
        while (!cur.empty() && isxdigit((unsigned char)cur.front())) {
          char h = cur.front();
          int d = isdigit(h) ? h - '0' : tolower(h) - 'a' + 10;
          v = v * 16 + d;
          cur = cur.drop_front();
        }
        t.intVal = v;
        t.tok = Tok::integer;
        t.bcdLit = true;
        return t;
      }
      // typed literal: ident '#' base '#' digits, e.g. B#16#FF
      if (!cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        // awlsim byte list: B#(66, 99) — take the first byte
        if (!cur.empty() && cur.front() == '(') {
          cur = cur.drop_front();
          while (!cur.empty() && !isdigit((unsigned char)cur.front()) &&
                 cur.front() != ')')
            cur = cur.drop_front();
          int64_t v = 0;
          while (!cur.empty() && isdigit((unsigned char)cur.front())) {
            v = v * 10 + (cur.front() - '0');
            cur = cur.drop_front();
          }
          while (!cur.empty() && cur.front() != ')')
            cur = cur.drop_front();
          if (!cur.empty())
            cur = cur.drop_front();
          t.intVal = v;
          t.tok = Tok::integer;
          return t;
        }
        int64_t base = 0;
        while (!cur.empty() && isdigit(cur.front())) {
          base = base * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        if (!cur.empty() && cur.front() == '#')
          cur = cur.drop_front();
        int64_t v = 0;
        if (base == 16) {
          // C-style hex prefix, e.g. DW#16#0xF7654660
          if (cur.size() >= 2 && cur.front() == '0' &&
              tolower((unsigned char)cur[1]) == 'x')
            cur = cur.drop_front(2);
          while (!cur.empty() && isxdigit(cur.front())) {
            char h = cur.front();
            int d = isdigit(h) ? h - '0' : tolower(h) - 'a' + 10;
            v = v * 16 + d;
            cur = cur.drop_front();
          }
        } else if (base == 8 || base == 2) {
          while (!cur.empty() && cur.front() >= '0' &&
                 cur.front() < (char)('0' + base)) {
            v = v * base + (cur.front() - '0');
            cur = cur.drop_front();
          }
        } else {
          while (!cur.empty() && isdigit(cur.front())) {
            v = v * 10 + (cur.front() - '0');
            cur = cur.drop_front();
          }
        }
        t.intVal = v;
        t.tok = Tok::integer;
        return t;
      }
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
        // scientific notation, e.g. 2.000000e+001
        if (!cur.empty() && (cur.front() == 'e' || cur.front() == 'E') &&
            cur.size() > 1 &&
            (isdigit((unsigned char)cur[1]) ||
             ((cur[1] == '+' || cur[1] == '-') && cur.size() > 2 &&
              isdigit((unsigned char)cur[2])))) {
          s += 'e';
          cur = cur.drop_front();
          if (!cur.empty() && (cur.front() == '+' || cur.front() == '-')) {
            s += cur.front();
            cur = cur.drop_front();
          }
          while (!cur.empty() && isdigit(cur.front())) {
            s += cur.front();
            cur = cur.drop_front();
          }
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
    case ';': t.tok = Tok::semi; return t; // statement separator
    case '\'': { // quoted literal ('abc', '16#2'): one quoted ident token
      auto begin = cur.begin();
      while (!cur.empty() && cur.front() != '\'' && cur.front() != '\n')
        cur = cur.drop_front();
      t.ident = llvm::StringRef(begin, cur.begin() - begin);
      if (!cur.empty() && cur.front() == '\'')
        cur = cur.drop_front();
      t.tok = Tok::ident;
      t.quoted = true;
      return t;
    }
    case '"': { // quoted name ("OB1"): scan as ident
      auto begin = cur.begin();
      while (!cur.empty() && cur.front() != '"' && cur.front() != '\n')
        cur = cur.drop_front();
      t.ident = llvm::StringRef(begin, cur.begin() - begin);
      if (!cur.empty() && cur.front() == '"')
        cur = cur.drop_front();
      t.tok = Tok::ident;
      t.quoted = true;
      return t;
    }
    case ':': t.tok = Tok::colon; return t;
    case '(': t.tok = Tok::lparen; return t;
    case ')': t.tok = Tok::rparen; return t;
    case '[': t.tok = Tok::lbracket; return t;
    case ']': t.tok = Tok::rbracket; return t;
    case '{': t.tok = Tok::lbrace; return t;
    case '}': t.tok = Tok::rbrace; return t;
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
    case '#':
      // #name: awlsim local variable of the block
      if (!cur.empty() &&
          (isalpha((unsigned char)cur.front()) || cur.front() == '_')) {
        auto begin = cur.begin();
        while (!cur.empty() &&
               (isalnum((unsigned char)cur.front()) || cur.front() == '_'))
          cur = cur.drop_front();
        t.ident = llvm::StringRef(begin, cur.begin() - begin);
        t.tok = Tok::localvar;
        return t;
      }
      cur = cur.drop_front(); // skip unknown char (header text etc.)
      return scan();
    default:
      cur = cur.drop_front(); // skip unknown char (header text etc.)
      return scan();
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
    // Same for the DATA_BLOCK init block, if one was emitted.
    if (dbInitBlock) {
      auto wrapLoc = loc();
      builder.setInsertionPointToEnd(module.getBody());
      auto wrap = builder.create<awl::BlockOp>(
          wrapLoc, builder.getStringAttr("_mlir_ciface___scl_db_init"));
      auto *entry = builder.createBlock(&wrap.getBody());
      builder.setInsertionPointToStart(entry);
      SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                 "__scl_db_init")));
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
    bool isXor = false;
  };
  llvm::SmallVector<Bracket, 4> brackets;
  bool sawOB1 = false;
  // awl.block "__scl_db_init" collecting DATA_BLOCK initial values, created
  // lazily; the JIT driver invokes it once before the first scan cycle.
  awl::BlockOp dbInitBlock = nullptr;
  // width (bits) of the most recent indirect [ARn, P#x.y] operand
  int32_t lastIndirectWidth = 0;
  // DB number of the DATA_BLOCK currently being parsed (-1 = none): bare
  // DBX/DBB/DBW/DBD operands in its init section address this DB
  int32_t dbOverride = -1;

  mlir::Location loc() {
    return mlir::FileLineColLoc::get(builder.getStringAttr(file), lexer.curLine(),
                                     lexer.col());
  }

  bool error(const llvm::Twine &msg) {
    llvm::errs() << file << ":" << lexer.curLine() << ":" << lexer.col() << ": "
                 << msg << "\n";
    return false;
  }

  std::string upperIdent() {
    std::string s(lexer.getIdent());
    for (auto &c : s)
      c = toupper(c);
    return s;
  }

  // block header keywords (TITLE = ..., AUTHOR : ...): line is skipped
  static bool isHeaderKw(const std::string &w) {
    return w == "TITLE" || w == "AUTHOR" || w == "FAMILY" || w == "NAME" ||
           w == "VERSION" || w == "NETWORK" || w == "KNOW_HOW_PROTECT";
  }

  // quoted block name ("CYCLE"): digits if present, else a stable hash
  static int64_t blockNumFromName(llvm::StringRef name) {
    int64_t n = 0;
    bool any = false;
    for (char c : name)
      if (isdigit((unsigned char)c)) {
        n = n * 10 + (c - '0');
        any = true;
      }
    if (any)
      return n;
    uint32_t h = 5381;
    for (char c : name)
      h = h * 33 + (unsigned char)c;
    return (int64_t)(h % 100000) + 1;
  }

  bool parseModule() {
    while (lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() != Tok::ident) {
        lexer.next(); // stray token between blocks (e.g. after a BEA): skip
        continue;
      }
      std::string w = upperIdent();
      if (w == "ORGANIZATION_BLOCK") {
        // long block form: ORGANIZATION_BLOCK OB 1 ... END_ORGANIZATION_BLOCK
        lexer.next();
        if (!parseBlockBody(/*longForm=*/true))
          return false;
        continue;
      }
      if (w == "DATA_BLOCK") {
        if (!parseDataBlock())
          return false;
        continue;
      }
      if (w == "FUNCTION" || w == "FUNCTION_BLOCK" || w == "TYPE") {
        // SCL-style block in a mixed awlsim project file: skip to its END_*
        lexer.next();
        std::string endKw = w == "TYPE" ? "END_TYPE"
                          : w == "FUNCTION_BLOCK" ? "END_FUNCTION_BLOCK"
                                                 : "END_FUNCTION";
        while (lexer.getCurToken() != Tok::eof) {
          if (lexer.getCurToken() == Tok::ident && upperIdent() == endKw) {
            lexer.next();
            break;
          }
          lexer.next();
        }
        continue;
      }
      if (w == "OB" || w == "FB" || w == "FC") {
        if (!parseBlockBody(/*longForm=*/false))
          return false;
        continue;
      }
      // stray END_* keyword between blocks (e.g. after a BE): skip
      if (w.rfind("END_", 0) == 0) {
        lexer.next();
        continue;
      }
      // header line before the first block
      if (isHeaderKw(w)) {
        lexer.nextLine();
        continue;
      }
      // { ... } attribute block: skip balanced
      if (lexer.getCurToken() == Tok::lbrace) {
        int d = 0;
        do {
          if (lexer.getCurToken() == Tok::lbrace)
            ++d;
          else if (lexer.getCurToken() == Tok::rbrace)
            --d;
          lexer.next();
        } while (d > 0 && lexer.getCurToken() != Tok::eof);
        continue;
      }
      // stray declaration section between blocks
      if (w.rfind("VAR", 0) == 0 || w == "TYPE" || w == "STRUCT") {
        lexer.next();
        std::string endKw = w == "TYPE" ? "END_TYPE"
                          : w == "STRUCT" ? "END_STRUCT"
                                         : "END_VAR";
        while (lexer.getCurToken() != Tok::eof) {
          if (lexer.getCurToken() == Tok::ident && upperIdent() == endKw) {
            lexer.next();
            break;
          }
          lexer.next();
        }
        continue;
      }
      // bare instruction list (no block header): wrap in an implicit OB 1
      // (only if no real OB 1 was parsed — e.g. dead code after a BEA)
      if (!sawOB1) {
        if (!parseBlockBody(/*longForm=*/false, /*implicitOB1=*/true))
          return false;
        continue;
      }
      lexer.next(); // stray token after the blocks: skip
    }
    return true;
  }

  /// Parse a DATA_BLOCK DB n [STRUCT ... END_STRUCT] [BEGIN ... END_DATA_BLOCK].
  /// The STRUCT declaration only sizes the DB (runtime memory, ADR 002); the
  /// BEGIN section holds initial values (`DBn.DBW 2 := 16#00FF`), which are
  /// emitted as stores into an awl.block "__scl_db_init" that the JIT driver
  /// runs once before the first scan.
  bool parseDataBlock() {
    lexer.next(); // DATA_BLOCK
    int32_t db = 0;
    if (lexer.getCurToken() == Tok::ident && upperIdent() == "DB") {
      lexer.next();
      if (lexer.getCurToken() == Tok::integer) {
        db = (int32_t)lexer.getInteger();
        lexer.next();
      }
    } else if (lexer.getCurToken() == Tok::integer) {
      db = (int32_t)lexer.getInteger();
      lexer.next();
    }
    while (lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() == Tok::ident &&
          upperIdent() == "END_DATA_BLOCK") {
        lexer.next();
        return true;
      }
      if (lexer.getCurToken() == Tok::ident && upperIdent() == "BEGIN") {
        lexer.next();
        // the init section consumes the closing END_DATA_BLOCK
        return parseDbInitSection(db);
      }
      if (lexer.getCurToken() == Tok::ident && isHeaderKw(upperIdent())) {
        lexer.nextLine();
        continue;
      }
      lexer.next();
    }
    return error("unterminated DATA_BLOCK");
  }

  /// Parse the BEGIN ... END_DATA_BLOCK initial-value section of a DATA_BLOCK
  /// into the __scl_db_init block (created lazily at module level).
  bool parseDbInitSection(int32_t db) {
    if (!dbInitBlock) {
      builder.setInsertionPointToEnd(module.getBody());
      dbInitBlock = builder.create<awl::BlockOp>(
          loc(), builder.getStringAttr("__scl_db_init"));
      builder.createBlock(&dbInitBlock.getBody());
    }
    OpBuilder::InsertionGuard guard(builder);
    Block *entry = &dbInitBlock.getBody().front();
    // a previous DATA_BLOCK already added the terminator: insert before it
    bool hadTerm = !entry->empty() && isa<awl::BeOp>(entry->back());
    if (hadTerm)
      builder.setInsertionPoint(&entry->back());
    else
      builder.setInsertionPointToEnd(entry);
    int32_t savedOverride = dbOverride;
    dbOverride = db;
    while (lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() == Tok::ident &&
          upperIdent() == "END_DATA_BLOCK") {
        lexer.next();
        break;
      }
      if (lexer.getCurToken() == Tok::semi) {
        lexer.next();
        continue;
      }
      if (!parseDbInitLine())
        return false;
    }
    dbOverride = savedOverride;
    if (!hadTerm) {
      builder.setInsertionPointToEnd(entry);
      builder.create<awl::BeOp>(loc());
    }
    return true;
  }

  /// One init assignment: <operand> := <value>   (e.g. DB1.DBX 0.0 := TRUE,
  /// DB1.DBW 2 := 16#00FF, DBW 4 := -1). Unrecognized lines are skipped.
  bool parseDbInitLine() {
    if (lexer.getCurToken() != Tok::ident || lexer.curQuoted()) {
      lexer.nextLine();
      return true;
    }
    // only real operands are handled here; symbolic member names
    // (TIMER_VAR, TEST1_A, ...) are approximated as DB word 0 via the
    // DBn branch of parseOperand — detect those by shape and skip the line
    std::string symName;
    {
      std::string w0 = upperIdent();
      bool looksOperand =
          w0.rfind("DB", 0) == 0 ||
          (isAreaLetter(w0[0]) &&
           (w0.size() == 1 || isdigit((unsigned char)w0[1]) ||
            w0[1] == 'B' || w0[1] == 'W' || w0[1] == 'D'));
      if (!looksOperand)
        symName = w0; // symbolic member name (e.g. TIMER_VAR := T 24)
    }
    mlir::Value addr;
    if (symName.empty()) {
      addr = parseOperand(false);
    } else {
      lexer.next(); // member name
      addr = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::DataBlock,
                                            effDb(dbOverride), 0, 0, 16);
    }
    if (!addr) {
      lexer.nextLine();
      return true;
    }
    // :=
    if (lexer.getCurToken() == Tok::colon)
      lexer.next();
    if (lexer.getCurToken() != Tok::assign) {
      lexer.nextLine();
      return true;
    }
    lexer.next(); // =
    mlir::Value val;
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "TRUE" || w == "FALSE") {
        auto i1 = builder.getI1Type();
        val = builder.create<awl::ConstOp>(
            loc(), i1, builder.getIntegerAttr(i1, w == "TRUE" ? 1 : 0));
        lexer.next();
      } else if ((w == "T" || w == "C" || w == "Z" || w == "TIME" || w == "S5TIME") &&
                 lexer.peekToken() == Tok::integer) {
        // TIMER_VAR := T 24 — raw S5TIME value
        lexer.next();
        int64_t n = lexer.getInteger();
        lexer.next();
        auto i16 = builder.getI16Type();
        val = builder.create<awl::ConstOp>(loc(), i16,
                                            builder.getIntegerAttr(i16, n));
      } else {
        lexer.nextLine();
        return true;
      }
    } else if (lexer.getCurToken() == Tok::integer ||
               lexer.getCurToken() == Tok::real ||
               lexer.getCurToken() == Tok::minus) {
      val = parseConst();
      if (!val)
        return false;
    } else {
      lexer.nextLine();
      return true;
    }
    if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(addr.getType())) {
      if (dt.getWidth() == 1)
        val = castTo(val, builder.getI1Type());
      builder.create<awl::StoreOp>(loc(), val, addr);
    } else if (addr.getType().isInteger(64)) {
      builder.create<awl::StoreIndirectOp>(
          loc(), castTo(val, builder.getI64Type()), addr,
          builder.getI32IntegerAttr(lastIndirectWidth));
    }
    // assignments are self-delimiting; the next token starts the next line
    return true;
  }

  // block := ("OB"|"FB"|"FC") number instruction* "BE"
  // longForm: ORGANIZATION_BLOCK OB n BEGIN ... END_ORGANIZATION_BLOCK
  bool parseBlockBody(bool longForm, bool implicitOB1 = false) {
    std::string kind;
    int64_t num;
    if (implicitOB1) {
      kind = "OB";
      num = 1;
    } else if (longForm) {
      if (lexer.getCurToken() == Tok::ident) {
        kind = upperIdent();
        if (kind == "OB" || kind == "FB" || kind == "FC")
          lexer.next();
        else
          kind.clear();
      }
      if (kind.empty())
        kind = "OB";
      if (lexer.getCurToken() == Tok::ident) {
        num = blockNumFromName(lexer.getIdent()); // quoted name: OB "CYCLE"
      } else if (lexer.getCurToken() == Tok::integer) {
        num = lexer.getInteger();
      } else {
        return error("expected block number after " + kind);
      }
      lexer.next();
    } else {
      kind = upperIdent();
      lexer.next();
      if (lexer.getCurToken() == Tok::ident) {
        num = blockNumFromName(lexer.getIdent());
      } else if (lexer.getCurToken() == Tok::integer) {
        num = lexer.getInteger();
      } else {
        return error("expected block number after " + kind);
      }
      lexer.next();
    }
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

    if (lexer.getCurToken() == Tok::ident && upperIdent() == "BEGIN")
      lexer.next();

    while (true) {
      // block end: END_ORGANIZATION_BLOCK / END_FUNCTION_BLOCK / ... or any
      // stray END_* keyword (END_VAR etc. are consumed by the VAR skip above)
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w.rfind("END_", 0) == 0) {
          lexer.next();
          if (!blockEnded)
            builder.create<awl::BeOp>(loc());
          blockEnded = true;
          break;
        }
      }
      if (lexer.getCurToken() == Tok::ident && upperIdent() == "BEGIN") {
        lexer.next();
        continue;
      }
      // declaration sections inside a block (awlsim project files):
      // skip VAR.../TYPE.../STRUCT... up to the matching END_*
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "VAR" || w == "VAR_INPUT" || w == "VAR_OUTPUT" ||
            w == "VAR_TEMP" || w == "VAR_IN_OUT" || w == "VAR_STAT" ||
            w == "TYPE" || w == "STRUCT") {
          lexer.next();
          std::string endKw = w == "TYPE" ? "END_TYPE"
                             : w == "STRUCT" ? "END_STRUCT"
                                             : "END_VAR";
          while (lexer.getCurToken() != Tok::eof) {
            if (lexer.getCurToken() == Tok::ident && upperIdent() == endKw) {
              lexer.next();
              break;
            }
            lexer.next();
          }
          continue;
        }
      }
      // { ... } attribute block: skip balanced
      if (lexer.getCurToken() == Tok::lbrace) {
        int d = 0;
        do {
          if (lexer.getCurToken() == Tok::lbrace)
            ++d;
          else if (lexer.getCurToken() == Tok::rbrace)
            --d;
          lexer.next();
        } while (d > 0 && lexer.getCurToken() != Tok::eof);
        continue;
      }
      // header lines inside a block (TITLE = ..., VERSION : ...): skip
      if (lexer.getCurToken() == Tok::ident && isHeaderKw(upperIdent())) {
        lexer.nextLine();
        continue;
      }
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
      if (lexer.getCurToken() == Tok::eof) {
        if (implicitOB1) {
          // bare instruction list: end the implicit OB 1 at EOF
          builder.create<awl::BeOp>(loc());
          blockEnded = true;
          break;
        }
        return error("missing BE in block " + name);
      }
      if (lexer.getCurToken() == Tok::semi) { // statement separator
        lexer.next();
        continue;
      }
      if (!parseInstruction(name))
        return false;
      if (blockEnded) {
        // dead code after BEA/BEB up to the END_* keyword: labels still
        // exist (jumps may target them); other instructions are skipped
        while (lexer.getCurToken() != Tok::eof) {
          if (lexer.getCurToken() == Tok::ident &&
              lexer.peekToken() == Tok::colon) {
            auto label = upperIdent();
            lexer.next();
            lexer.next();
            // keep the block ending in awl.be: insert dead-code labels
            // before the terminator
            Operation *term = &builder.getInsertionBlock()->back();
            auto lab = builder.create<awl::LabelOp>(loc(),
                                                    builder.getStringAttr(label));
            lab->moveBefore(term);
            continue;
          }
          if (lexer.getCurToken() == Tok::ident) {
            std::string w2 = upperIdent();
            // stop at the end keyword or at the next block's header
            if (w2.rfind("END_", 0) == 0 || w2 == "ORGANIZATION_BLOCK" ||
                w2 == "DATA_BLOCK" || w2 == "OB" || w2 == "FB" ||
                w2 == "FC" || w2 == "SFB" || w2 == "SFC" || w2 == "DB")
              break;
          }
          lexer.nextLine();
        }
        break;
      }
    }
    blockEnded = false;
    return true;
  }

  bool blockEnded = false;

  bool parseInstruction(const std::string &blockName) {
    auto t = lexer.getCurToken();
    if (t == Tok::semi) { // statement separator
      lexer.next();
      return true;
    }
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
      // `+ <immediate>` (awlsim convenience): add/sub/... constant to ACCU1
      if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real || lexer.getCurToken() == Tok::minus) {
        mlir::Type ty = lexer.getCurToken() == Tok::real
                            ? (mlir::Type)builder.getF32Type()
                            : (mlir::Type)builder.getI32Type();
        mlir::Value b = parseConst();
        if (!b)
          return false;
        if (acc.empty())
          acc.push_back(constInt(builder.getI32Type(), 0));
        mlir::Value a = popAcc();
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
      if (lexer.getCurToken() != Tok::ident)
        return error("expected I/D/R after arithmetic operator");
      std::string w = upperIdent();
      lexer.next();
      if (w.rfind("AR", 0) == 0) { // +AR1 P#...: address register arithmetic
        if (lexer.getCurToken() == Tok::integer)
          lexer.next();
        return true;
      }
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
    if (m == "BE" || m == "BEA" || m == "BEB") { // BEA/BEB: block end variants
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
    if ((m == "U" || m == "UN" || m == "A" || m == "AN" || m == "O" ||
         m == "ON") &&
        lexer.getCurToken() != Tok::lparen) {
      // bare U/O/... (no operand, e.g. at end of a chain): RLO unchanged.
      // An ident followed by an address token is an operand (U A 0.0,
      // UN L 0.0); only if no address follows and the ident is an
      // instruction keyword is the logic op bare (O / L STW).
      if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        Tok nxt = lexer.peekToken();
        bool addrFollows = nxt == Tok::integer || nxt == Tok::real ||
                           nxt == Tok::dot || nxt == Tok::lbracket;
        if (!addrFollows &&
            (w == "L" || w == "T" || w == "U" || w == "UN" || w == "O" ||
             w == "ON" || w == "A" || w == "AN" || w == "NOT" || w == "SET" ||
             w == "CLR" || w == "BE" || w == "NOP" || w == "CALL" ||
             w == "SPA" || w == "SPB" || w == "SPN"))
          return ensureRlo();
      }
      return parseRloLogic(m == "AN" ? "UN" : m);
    }
    if (m == "NOT") {
      if (!ensureRlo())
        return false;
      rlo = builder.create<awl::NotOp>(loc(), builder.getI1Type(), rlo);
      return true;
    }
    if (m == "SET") { // RLO := 1
      rlo = builder.create<awl::ConstOp>(loc(), builder.getI1Type(),
                                          builder.getIntegerAttr(builder.getI1Type(), 1));
      rloActive = true;
      return true;
    }
    if (m == "CLR") { // RLO := 0
      rlo = builder.create<awl::ConstOp>(loc(), builder.getI1Type(),
                                          builder.getIntegerAttr(builder.getI1Type(), 0));
      rloActive = true;
      return true;
    }
    if (m == "AUF") { // AUF DB n / AUF DI n: open data block -> set the
      // current DB register (DB 0 operands then address it, ABI v5)
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "DB" || w == "DI")
          lexer.next();
        else if ((w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0) &&
                 w.size() > 2) {
          // attached number: AUF DB1 / OPN DI12
          builder.create<awl::SetDbOp>(
              loc(), builder.getI32IntegerAttr(atoi(w.c_str() + 2)));
          lexer.next();
          return true;
        }
      }
      if (lexer.getCurToken() == Tok::integer) {
        int64_t db = lexer.getInteger();
        lexer.next();
        builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
      } else if (lexer.getCurToken() == Tok::lbracket) { // AUF DB [LW 0]
        lexer.next();
        while (lexer.getCurToken() != Tok::rbracket &&
               lexer.getCurToken() != Tok::eof)
          lexer.next();
        if (lexer.getCurToken() == Tok::rbracket)
          lexer.next();
      }
      return true;
    }
    if (m.rfind("LAR", 0) == 0 || m.rfind("+AR", 0) == 0) {
      // LAR1/LAR2 <operand>: load a pointer into address register 1/2.
      // P#x.y literal -> raw byte*8+bit value; a direct operand (DB1.DBW 4,
      // M 10.0) -> packed pointer; no operand -> ACCU1 (as TARn).
      int ar = (m.size() >= 4 && isdigit(m[3])) ? m[3] - '0'
               : (m == "+AR2")                  ? 2
                                                : 1;
      mlir::Value ptr;
      if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real) {
        int64_t off = lexer.getInteger(); // P#x.y lexes as byte*8+bit
        lexer.next();
        ptr = constInt(builder.getI64Type(), off);
      } else if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        bool looksOperand =
            w.rfind("DB", 0) == 0 ||
            (isAreaLetter(w[0]) &&
             (w.size() == 1 || isdigit((unsigned char)w[1]) ||
              w[1] == 'B' || w[1] == 'W' || w[1] == 'D'));
        if (!looksOperand)
          ptr = nullptr; // bare LARn: take ACCU1 below
        else if (w.size() <= 2 && w[0] == 'L') { // LD/LW/LB: local data
          lexer.next();
          int32_t byte = 0;
          if (lexer.getCurToken() == Tok::integer) {
            byte = (int32_t)lexer.getInteger();
            lexer.next();
          }
          ptr = builder.create<awl::DirectOp>(
              loc(), mlir::scl::DirectArea::Marker, 0, byte, 0, 16);
        } else {
          ptr = parseOperand(false); // DB1.DBW 4, M 10.0, ...
          if (!ptr)
            return true;
        }
      }
      if (!ptr)
        ptr = castTo(popAcc(), builder.getI64Type()); // bare LARn = TARn
      builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(ar), ptr);
      return true;
    }
    if (m.rfind("TAR", 0) == 0) {
      // TAR1/TAR2 [operand]: AR := pointer operand, or ACCU1 if none
      int ar = (m.size() >= 4 && isdigit(m[3])) ? m[3] - '0' : 1;
      mlir::Value ptr;
      if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real) {
        int64_t off = lexer.getInteger(); // P#x.y
        lexer.next();
        ptr = constInt(builder.getI64Type(), off);
      } else if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        bool looksOperand =
            w.rfind("DB", 0) == 0 ||
            (isAreaLetter(w[0]) &&
             (w.size() == 1 || isdigit((unsigned char)w[1]) ||
              w[1] == 'B' || w[1] == 'W' || w[1] == 'D'));
        if (looksOperand) {
          if (w.size() <= 2 && w[0] == 'L') { // LD/LW/LB: local data
            lexer.next();
            int32_t byte = 0;
            if (lexer.getCurToken() == Tok::integer) {
              byte = (int32_t)lexer.getInteger();
              lexer.next();
            }
            ptr = builder.create<awl::DirectOp>(
                loc(), mlir::scl::DirectArea::Marker, 0, byte, 0, 16);
          } else {
            ptr = parseOperand(false);
            if (!ptr)
              return true;
          }
        }
      }
      if (!ptr)
        ptr = castTo(popAcc(), builder.getI64Type()); // no operand: ACCU1
      builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(ar), ptr);
      return true;
    }
    if (lexer.getCurToken() == Tok::lparen &&
        (m == "A" || m == "O" || m == "U" || m == "UN" || m == "ON")) {
      lexer.next();
      if (!ensureRlo())
        return false;
      brackets.push_back({rlo, m == "O" || m == "ON", false});
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
        m == "XOD" || m == "UD" || m == "UW")
      return parseWordLogic(m == "UD" ? "AD" : m == "UW" ? "OW" : m);
    if (m == "SP" || m == "SE" || m == "SS" || m == "SV" || m == "SI" ||
        m == "SA" || m == "SD") // SD = English off-delay (German SA)
      return parseTimerStart(m == "SD" ? "SA" : m);
    if (m == "ZV" || m == "ZR")
      return parseCounterZW(m == "ZV");
    if (m == "SPA" || m == "SPB" || m == "SPN" || m == "SPBN")
      return parseJump(m == "SPBN" ? "SPN" : m);
    if (m == "TAK") { // exchange ACCU1/ACCU2
      padAcc();
      std::swap(acc[acc.size() - 1], acc[acc.size() - 2]);
      return true;
    }
    if (m == "NEGI" || m == "NEGD") { // negate ACCU1
      mlir::Value a = popAcc();
      if (!a)
        return false;
      acc.push_back(builder.create<awl::NegOp>(loc(), a.getType(), a));
      return true;
    }
    if (m == "SLW" || m == "SRW" || m == "SSI" || m == "SSD" ||
        m == "SLD" || m == "SRD" || m == "RLD" || m == "RRD" ||
        m == "RLDA" || m == "RRDA")
      return parseShift(m);
    if (m == "ABS" || m == "NEGR" || m == "SIN" || m == "COS" || m == "TAN" ||
        m == "ASIN" || m == "ACOS" || m == "ATAN" || m == "SQRT" ||
        m == "EXP" || m == "LN" || m == "SQR")
      return parseRealMath(m);
    if (m == "TRUNC") { // real ACCU1 -> int (truncate)
      mlir::Value a = popAcc();
      acc.push_back(castTo(castTo(a, builder.getF32Type()), builder.getI32Type()));
      return true;
    }
    if (m == "DTR") { // DWORD ACCU1 -> real
      mlir::Value a = popAcc();
      acc.push_back(castTo(a, builder.getF32Type()));
      return true;
    }
    if (m == "FR") { // FR T n / FR C n: enable — reset the edge memory of
      // the timer/counter at runtime; the RLO is NOT modified
      mlir::Value v = parseOperand(false);
      if (!v)
        return false;
      if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(v.getType()))
        builder.create<awl::FrOp>(loc(), v, rloInput());
      return true;
    }
    if (m == "TDB") { // TDB [DB n]: transfer DB register — not modelled
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "DB" || w == "DI")
          lexer.next();
      }
      if (lexer.getCurToken() == Tok::integer)
        lexer.next();
      return true;
    }
    if (m == "OPN") { // OPN DB n / OPN DI n / OPN DB1: open data block (like AUF)
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "DB" || w == "DI")
          lexer.next();
        else if ((w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0) &&
                 w.size() > 2) {
          builder.create<awl::SetDbOp>(
              loc(), builder.getI32IntegerAttr(atoi(w.c_str() + 2)));
          lexer.next();
          return true;
        }
      }
      if (lexer.getCurToken() == Tok::integer) {
        int64_t db = lexer.getInteger();
        lexer.next();
        builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
      }
      return true;
    }
    if (m == "__STWRST") { // awlsim test directive: reset status word (RLO=0)
      auto i1 = builder.getI1Type();
      rlo = builder.create<awl::ConstOp>(loc(), i1, builder.getIntegerAttr(i1, 0));
      rloActive = true;
      return true;
    }
    if (m == "__SLEEP") { // awlsim test directive: advance the virtual clock
      int64_t ms = 0;
      if (lexer.getCurToken() == Tok::integer) {
        ms = lexer.getInteger();
        lexer.next();
      }
      builder.create<awl::SleepOp>(loc(), builder.getI32IntegerAttr((int32_t)ms));
      return true;
    }
    if (m == "LC") // LC Z n: load counter value into ACCU1
      return parseLoad();
    if (m == "FP" || m == "FN") // edge memory not modelled: approximated
      return parseRloLogic(m == "FP" ? "U" : "UN"); // by U/UN
    if (m == "SAVE" || m == "MCRA" || m == "MCRD")
      return true; // BR / master-control-relay bits are not modelled
    if (m == "BLD") { // BLD <n>: display code, no-op
      if (lexer.getCurToken() == Tok::integer)
        lexer.next();
      return true;
    }
    if (m == "MCR") { // MCR( ... MCR): master control relay — bracket only
      if (lexer.getCurToken() == Tok::lparen) {
        lexer.next();
        if (!ensureRlo())
          return false;
        brackets.push_back({rlo, false});
        rlo = builder.create<awl::ConstOp>(
            loc(), builder.getI1Type(),
            builder.getIntegerAttr(builder.getI1Type(), 1));
        rloActive = true;
        return true;
      }
      if (lexer.getCurToken() == Tok::rparen) {
        lexer.next();
        return parseCloseBracket();
      }
      return true;
    }
    if (m == "SPBB")
      return parseJump("SPB");
    if (m == "ENT" || m == "PUSH") { // push a copy of ACCU1
      acc.push_back(acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back());
      return true;
    }
    if (m == "POP" || m == "LEAVE") { // drop ACCU1
      if (!acc.empty())
        popAcc();
      return true;
    }
    if (m == "INC" || m == "DEC") { // ACCU1 +/-= n (default 1)
      int64_t n = 1;
      if (lexer.getCurToken() == Tok::integer) {
        n = lexer.getInteger();
        lexer.next();
      }
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto c = constInt(a.getType(), m == "INC" ? n : -n);
      acc.push_back(builder.create<awl::AddOp>(loc(), a.getType(), a, c));
      rloActive = false;
      return true;
    }
    if (m == "ITD" || m == "DTI") { // int <-> dword (sign/zero extend)
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      acc.push_back(castTo(a, m == "ITD" ? builder.getI32Type()
                                         : builder.getI16Type()));
      rloActive = false;
      return true;
    }
    if (m == "INVI" || m == "INVD") { // one's complement
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto ty = m == "INVI" ? builder.getI16Type() : builder.getI32Type();
      a = castTo(a, ty);
      acc.push_back(
          builder.create<awl::XOrOp>(loc(), ty, a, constInt(ty, -1)));
      rloActive = false;
      return true;
    }
    if (m == "RND") { // RND/RND+/RND-: real -> int (truncates; no rounding
      if (lexer.getCurToken() == Tok::plus || lexer.getCurToken() == Tok::minus)
        lexer.next(); // op in the dialect)
      mlir::Value a = acc.empty() ? constInt(builder.getF32Type(), 0) : popAcc();
      acc.push_back(castTo(castTo(a, builder.getF32Type()),
                           builder.getI32Type()));
      rloActive = false;
      return true;
    }
    if (m == "MOD") { // ACCU1 %% ACCU2: a - (a / b) * b
      padAcc();
      mlir::Value b = popAcc();
      mlir::Value a = popAcc();
      auto ty = (a.getType() == builder.getI16Type() &&
                 b.getType() == builder.getI16Type())
                    ? builder.getI16Type()
                    : builder.getI32Type();
      a = castTo(a, ty);
      b = castTo(b, ty);
      mlir::Value q = builder.create<awl::DivOp>(loc(), ty, a, b);
      acc.push_back(builder.create<awl::SubOp>(
          loc(), ty, a, builder.create<awl::MulOp>(loc(), ty, q, b)));
      rloActive = false;
      return true;
    }
    if (m == "TAD") // exchange ACCU1/ACCU2 (dword view) — same as TAK
      return parseIdentInstruction("TAK", blockName);
    if ((m == "X" || m == "XN") && lexer.getCurToken() == Tok::lparen) {
      lexer.next();
      if (!ensureRlo())
        return false;
      brackets.push_back({rlo, false, true});
      rlo = builder.create<awl::ConstOp>(loc(), builder.getI1Type(),
                                          builder.getIntegerAttr(builder.getI1Type(), 1));
      rloActive = true;
      return true;
    }
    if (m == "X" || m == "XN") { // X/XN bit: rlo = rlo XOR bit (XN: negated)
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "TRUE" || w == "FALSE") {
          lexer.next();
          if (!ensureRlo())
            return false;
          auto i1 = builder.getI1Type();
          mlir::Value c = builder.create<awl::ConstOp>(
              loc(), i1, builder.getIntegerAttr(i1, w == "TRUE" ? 1 : 0));
          rlo = builder.create<awl::XOrOp>(loc(), i1, rlo, c);
          if (m == "XN")
            rlo = builder.create<awl::NotOp>(loc(), i1, rlo);
          return true;
        }
      }
      mlir::Value addr = parseBitOperand();
      if (!addr)
        return false;
      if (!ensureRlo())
        return false;
      auto i1 = builder.getI1Type();
      mlir::Value b = builder.create<awl::LoadOp>(loc(), addr);
      rlo = builder.create<awl::XOrOp>(loc(), i1, rlo, castTo(b, i1));
      if (m == "XN")
        rlo = builder.create<awl::NotOp>(loc(), i1, rlo);
      return true;
    }
    if (m == "SPL")
      return parseJump("SPA"); // jump via label list: treated as SPA
    if (m == "SPBNB" || m == "SPBI" || m == "SPBIN" || m == "JNB" ||
        m == "JNBI" || m == "SPO" || m == "SOS" || m == "SPU")
      return parseJump("SPN");
    if (m == "JB" || m == "JBI")
      return parseJump("SPB");
    if (m == "LOOP") { // jump if ACCU1 != 0
      if (lexer.getCurToken() != Tok::ident)
        return error("expected label after LOOP");
      std::string label = upperIdent();
      lexer.next();
        mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto ty = a.getType();
      auto c = builder.create<awl::CmpNeOp>(loc(), builder.getI1Type(),
                                            castTo(a, ty), constInt(ty, 0));
      builder.create<awl::JumpIfOp>(loc(), c, builder.getStringAttr(label));
      rloActive = false;
      return true;
    }
    if (m == "SPZ" || m == "SPPZ" || m == "SPMZ" || m == "SPS" || m == "SPP" ||
        m == "SPM") {
      // jump if ACCU1 == 0 / > 0 / < 0
      if (lexer.getCurToken() != Tok::ident)
        return error("expected label after " + m);
      std::string label = upperIdent();
      lexer.next();
        mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto ty = a.getType();
      auto i1 = builder.getI1Type();
      mlir::Value c;
      if (m == "SPZ")
        c = builder.create<awl::CmpEqOp>(loc(), i1, castTo(a, ty), constInt(ty, 0));
      else if (m == "SPPZ" || m == "SPS" || m == "SPP")
        c = builder.create<awl::CmpGtOp>(loc(), i1, castTo(a, ty), constInt(ty, 0));
      else
        c = builder.create<awl::CmpLtOp>(loc(), i1, castTo(a, ty), constInt(ty, 0));
      builder.create<awl::JumpIfOp>(loc(), c, builder.getStringAttr(label));
      rloActive = false;
      return true;
    }
    if (m == "BTI" || m == "BTD") { // BCD -> binary
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto ty = m == "BTI" ? builder.getI16Type() : builder.getI32Type();
      acc.push_back(bcdToInt(castTo(a, ty), ty, m == "BTI" ? 4 : 8));
      rloActive = false;
      return true;
    }
    if (m == "ITB" || m == "DTB") { // binary -> BCD
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      auto ty = m == "ITB" ? builder.getI16Type() : builder.getI32Type();
      acc.push_back(intToBcd(castTo(a, ty), ty, m == "ITB" ? 4 : 8));
      rloActive = false;
      return true;
    }
    if (m == "TAW") { // ACCU1 -> status word (approximated: marker word 0)
      mlir::Value v = acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back();
      auto stw = builder.create<awl::DirectOp>(
          loc(), mlir::scl::DirectArea::Marker, 0, 0, 0, 16);
      builder.create<awl::StoreOp>(loc(), castTo(v, builder.getI16Type()), stw);
      rloActive = false;
      return true;
    }
    if (m.rfind("__ASSERT", 0) == 0)
      return parseAssert();
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

  mlir::Value constInt(mlir::Type ty, int64_t v) {
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(ty))
      return builder.create<awl::ConstOp>(loc(), ty,
                                          builder.getFloatAttr(ft, (double)v));
    return builder.create<awl::ConstOp>(loc(), ty, builder.getIntegerAttr(ty, v));
  }

  // The S7 accumulators are initialized to 0; pad a short stack so that
  // arithmetic/comparisons on an "empty" accumulator still produce SSA.
  void padAcc() {
    if (acc.empty())
      acc.push_back(constInt(builder.getI32Type(), 0));
    if (acc.size() == 1)
      acc.insert(acc.begin(), constInt(builder.getI32Type(), 0));
  }

  // BCD <-> binary conversion built from awl arith ops (the awl dialect has
  // no shift ops): BCD digit i of v is (v / 16^i) & 15.
  mlir::Value bcdToInt(mlir::Value v, mlir::Type ty, int digits) {
    mlir::Value n = constInt(ty, 0);
    int64_t pow16 = 1, pow10 = 1;
    for (int i = 0; i < digits; ++i) {
      mlir::Value q =
          builder.create<awl::DivOp>(loc(), ty, v, constInt(ty, pow16));
      mlir::Value d =
          builder.create<awl::AndOp>(loc(), ty, q, constInt(ty, 15));
      mlir::Value term =
          builder.create<awl::MulOp>(loc(), ty, d, constInt(ty, pow10));
      n = builder.create<awl::AddOp>(loc(), ty, n, term);
      pow16 *= 16;
      pow10 *= 10;
    }
    return n;
  }
  mlir::Value intToBcd(mlir::Value v, mlir::Type ty, int digits) {
    mlir::Value bcd = constInt(ty, 0);
    mlir::Value m = v;
    int64_t pow16 = 1;
    for (int i = 0; i < digits; ++i) {
      mlir::Value q =
          builder.create<awl::DivOp>(loc(), ty, m, constInt(ty, 10));
      mlir::Value d = builder.create<awl::SubOp>(
          loc(), ty, m,
          builder.create<awl::MulOp>(loc(), ty, q, constInt(ty, 10)));
      bcd = builder.create<awl::AddOp>(
          loc(), ty, bcd,
          builder.create<awl::MulOp>(loc(), ty, d, constInt(ty, pow16)));
      m = q;
      pow16 *= 16;
    }
    return bcd;
  }

  // RLO value for a timer/counter instruction input: the current RLO if a
  // logic chain is active, else the preserved value, else 0 at block start
  mlir::Value rloInput() {
    if (rloActive && rlo)
      return rlo;
    if (rlo)
      return rlo;
    return constInt(builder.getI1Type(), 0);
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

  // The four ACCU registers always hold a value (0 at block start), so an
  // "empty" parse-time stack yields 0 instead of an underflow error.
  mlir::Value popAcc() {
    if (acc.empty())
      return constInt(builder.getI32Type(), 0);
    mlir::Value v = acc.back();
    acc.pop_back();
    return v;
  }

  // MARK: operands

  static bool isAreaLetter(char c) {
    // German E/A/Z and English I/Q/C; L = local data (mapped to marker)
    return c == 'E' || c == 'A' || c == 'M' || c == 'T' || c == 'Z' ||
           c == 'I' || c == 'Q' || c == 'C' || c == 'L';
  }
  static mlir::scl::DirectArea areaOf(char c) {
    switch (c) {
    case 'E': case 'I': return mlir::scl::DirectArea::Input;
    case 'A': case 'Q': return mlir::scl::DirectArea::Output;
    case 'M': return mlir::scl::DirectArea::Marker;
    case 'T': return mlir::scl::DirectArea::Timer;
    case 'Z': case 'C': return mlir::scl::DirectArea::Counter;
    }
    return mlir::scl::DirectArea::Marker;
  }

  // Fallback for an identifier in operand position that is not an area
  // operand: awlsim symbol names (SYM_M_0.0 -> M 0.0) are decoded, anything
  // else (quoted names, string literals) maps to a hashed marker bit.
  mlir::Value symbolicOperand(const std::string &w) {
    uint32_t h = 5381;
    for (char c : w)
      h = h * 33 + (unsigned char)c;
    int32_t byte = (int32_t)(h % 4096), bit = (int32_t)((h >> 12) % 8);
    auto area = mlir::scl::DirectArea::Marker;
    if (w.rfind("SYM_", 0) == 0 && w.size() > 4 && isAreaLetter(w[4])) {
      area = areaOf(w[4]);
      byte = w.size() > 5 ? atoi(w.c_str() + 5) : 0;
      bit = 0;
    }
    while (lexer.getCurToken() == Tok::dot) { // "Data_1".VAR0 [i] .member
      lexer.next();
      if (lexer.getCurToken() == Tok::integer) {
        bit = (int32_t)lexer.getInteger();
        lexer.next();
      } else if (lexer.getCurToken() == Tok::ident ||
                 lexer.getCurToken() == Tok::localvar) {
        lexer.next();
      }
      while (lexer.getCurToken() == Tok::lbracket) {
        lexer.next();
        while (lexer.getCurToken() != Tok::rbracket &&
               lexer.getCurToken() != Tok::eof)
          lexer.next();
        if (lexer.getCurToken() == Tok::rbracket)
          lexer.next();
      }
    }
    return builder.create<awl::DirectOp>(loc(), area, 0, byte, bit, 1);
  }

  // Parse an operand: E/A/M/T/Z [B|W|D] byte [.bit] or DBn.DB[XBWD] byte [.bit]
  // `bitContext`: for T/Z operands without a size letter, `U T0`-style bit
  // access yields the status bit (width 1, bit 0); otherwise the 16-bit value.
  /// Parse `[AR1, P#x.y]` (or AR2) into an awl.indirect op yielding the
  /// packed pointer (i64). `widthBits` records the access width for the
  /// load/store that consumes the pointer (lastIndirectWidth).
  mlir::Value parseIndirectBracket(int32_t widthBits) {
    int32_t ar = 1, offBits = 0;
    lexer.next(); // [
    while (lexer.getCurToken() != Tok::rbracket &&
           lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w.rfind("AR", 0) == 0 && w.size() == 3 && isdigit(w[2]))
          ar = w[2] - '0';
        lexer.next();
      } else if (lexer.getCurToken() == Tok::integer) {
        // P#x.y lexes as one integer token: byte*8 + bit
        offBits = (int32_t)lexer.getInteger();
        lexer.next();
      } else {
        lexer.next(); // comma etc.
      }
    }
    if (lexer.getCurToken() == Tok::rbracket)
      lexer.next();
    lastIndirectWidth = widthBits;
    return builder.create<awl::IndirectOp>(
        loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar),
        builder.getI32IntegerAttr(offBits));
  }

  // DB number for a DB operand without an explicit number: inside a
  // DATA_BLOCK init section that is the DATA_BLOCK's own number.
  int32_t effDb(int32_t db) const {
    return db == 0 && dbOverride >= 0 ? dbOverride : db;
  }

  mlir::Value parseOperand(bool bitContext = false) {
    // operand starting with '[': fully indirect U [AR1, P#0.0]
    if (lexer.getCurToken() == Tok::lbracket)
      return parseIndirectBracket(bitContext ? 1 : 16);
    // #name: awlsim local variable, mapped to a marker bit by name hash
    if (lexer.getCurToken() == Tok::localvar) {
      std::string name = std::string(lexer.getIdent());
      lexer.next();
      uint32_t h = 5381;
      for (char c : name)
        h = h * 33 + (unsigned char)c;
      return builder.create<awl::DirectOp>(
          loc(), mlir::scl::DirectArea::Marker, 0, (int32_t)(h % 4096),
          (int32_t)((h >> 12) % 8), 1);
    }
    if (lexer.getCurToken() != Tok::ident)
      return (error("expected operand"), nullptr);
    std::string w = upperIdent();
    bool quoted = lexer.curQuoted();
    lexer.next();
    if (quoted)
      return symbolicOperand(w); // "symbol" / 'literal' in operand position
    // pseudo operands DBLG/DBNO/DILG (DB length/number): not modelled,
    // load 0
    if (w == "DBLG" || w == "DBNO" || w == "DILG" || w == "DINO") {
      auto ty = builder.getI16Type();
      return builder.create<awl::ConstOp>(loc(), ty,
                                          builder.getIntegerAttr(ty, 0));
    }

    // STW: status word — approximated by marker word 0
    if (w == "STW")
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Marker,
                                            0, 0, 0, 16);
    // width letter first = indirect access: L B [AR1, P#0.0] — the address
    // register contents are not modelled; use the P# byte offset from the
    // brackets in the marker area
    if (w == "B" || w == "W" || w == "D") {
      int32_t width = w[0] == 'B' ? 8 : w[0] == 'W' ? 16 : 32;
      if (lexer.getCurToken() == Tok::lbracket)
        return parseIndirectBracket(width); // B/W/D [AR1, P#x.y]
      int32_t byte = 0;
      if (lexer.getCurToken() == Tok::integer) {
        byte = (int32_t)lexer.getInteger();
        lexer.next();
      }
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Marker,
                                            0, byte, 0, width);
    }
    // DBX/DBB/DBW/DBD without a DB number: instance DB of the enclosing FB
    // (db = 0), e.g. `L DBW 4`
    if (w.rfind("DBX", 0) == 0 || w.rfind("DBB", 0) == 0 ||
        w.rfind("DBW", 0) == 0 || w.rfind("DBD", 0) == 0) {
      int32_t width = w[2]=='X'?1:w[2]=='B'?8:w[2]=='W'?16:32;
      int32_t byte = w.size() > 3 ? atoi(w.c_str() + 3) : -1, bit = 0;
      bool hasBit = false;
      double realBit = -1;
      if (byte < 0 && lexer.getCurToken() == Tok::lbracket)
        return parseIndirectBracket(width); // DBW [AR1, P#x.y]
      if (byte < 0 && lexer.getCurToken() == Tok::integer) {
        byte = (int32_t)lexer.getInteger();
        lexer.next();
      } else if (byte < 0 && lexer.getCurToken() == Tok::real) {
        double r = lexer.getReal();
        byte = (int32_t)r;
        realBit = r - byte;
        lexer.next();
      } else if (byte < 0) {
        return (error("expected byte address in operand '" + w + "'"), nullptr);
      }
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
      while (lexer.getCurToken() == Tok::lbracket) {
        lexer.next();
        while (lexer.getCurToken() != Tok::rbracket && lexer.getCurToken() != Tok::eof)
          lexer.next();
        if (lexer.getCurToken() == Tok::rbracket)
          lexer.next();
      }
      if (width == 1 && !hasBit)
        return (error("DBX bit access requires a bit number"), nullptr);
      if (width != 1 && hasBit)
        return (error("only DBX takes a bit number"), nullptr);
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::DataBlock,
                                            effDb(0), byte, bit, width);
    }

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
      bool isDbAccess = dw.size() >= 3 && dw.rfind("DB", 0) == 0 &&
                        (dw[2] == 'X' || dw[2] == 'B' || dw[2] == 'W' ||
                         dw[2] == 'D');
      if (!isDbAccess) {
        // symbolic DB member (DB1.VAR0, DB1.DBVAR_UDT1.VAR6_BOOL, [i]
        // indices): DB contents are runtime memory, not modelled;
        // approximate with DBX 0.0 / DBW 0 of that DB
        while (true) {
          while (lexer.getCurToken() == Tok::lbracket) {
            lexer.next();
            while (lexer.getCurToken() != Tok::rbracket &&
                   lexer.getCurToken() != Tok::eof)
              lexer.next();
            if (lexer.getCurToken() == Tok::rbracket)
              lexer.next();
          }
          if (lexer.getCurToken() == Tok::dot) {
            lexer.next();
            if (lexer.getCurToken() == Tok::ident ||
                lexer.getCurToken() == Tok::localvar)
              lexer.next();
            continue;
          }
          break;
        }
        return builder.create<awl::DirectOp>(
            loc(), mlir::scl::DirectArea::DataBlock, effDb(db), 0, 0,
            bitContext ? 1 : 16);
      }
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
      while (lexer.getCurToken() == Tok::lbracket) {
        lexer.next();
        while (lexer.getCurToken() != Tok::rbracket && lexer.getCurToken() != Tok::eof)
          lexer.next();
        if (lexer.getCurToken() == Tok::rbracket)
          lexer.next();
      }
      if (width == 1 && !hasBit)
        return (error("DBX bit access requires a bit number"), nullptr);
      if (width != 1 && hasBit)
        return (error("only DBX takes a bit number"), nullptr);
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::DataBlock,
                                            effDb(db), byte, bit, width);
    }

    // E/A/M/T/Z [B|W|D] [byte] [.bit]; P[E|A][B|W|D] = peripheral I/Q
    bool isPeriph = w.size() >= 2 && w[0] == 'P' &&
                    (w[1] == 'E' || w[1] == 'A' || w[1] == 'I' || w[1] == 'Q');
    if (w.size() < 1 || !(isAreaLetter(w[0]) || isPeriph))
      return symbolicOperand(w); // awlsim symbol/string in operand position
    auto area = isPeriph ? (w[1] == 'E' || w[1] == 'I' ? mlir::scl::DirectArea::Input
                                                      : mlir::scl::DirectArea::Output)
                         : areaOf(w[0]); // 'L' (local data) -> Marker
    size_t p = isPeriph ? 2 : 1;
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
    if (byte < 0 && lexer.getCurToken() == Tok::lbracket)
      // indirect access M [AR1, P#0.0] / Q [AR2, P#x.y]
      return parseIndirectBracket(width);
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
    // awlsim extension: member index suffix, e.g. DB100.DBVAR2[2].V2
    while (lexer.getCurToken() == Tok::lbracket) {
      lexer.next();
      while (lexer.getCurToken() != Tok::rbracket && lexer.getCurToken() != Tok::eof)
        lexer.next();
      if (lexer.getCurToken() == Tok::rbracket)
        lexer.next();
    }
    if (width == 1 && !hasBit)
      return (error("bit access '" + w + "' requires a bit number (.bit)"), nullptr);
    if (width != 1 && hasBit)
      return (error("only bit access takes a bit number"), nullptr);
    return builder.create<awl::DirectOp>(loc(), area, 0, byte, bit, width);
  }

  static bool isBit(mlir::Value v) {
    auto dt = mlir::dyn_cast<mlir::scl::DirectType>(v.getType());
    return dt && dt.getWidth() == 1;
  }

  mlir::Value parseBitOperand() {
    mlir::Value v = parseOperand(true);
    if (!v)
      return nullptr;
    // i64: indirect bit operand [AR1, P#x.y] (packed pointer)
    if (!isBit(v) && !v.getType().isInteger(64))
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
      mlir::Value addr = parseOperand(false);
      if (!addr)
        return false;
      if (addr.getType().isInteger(64)) {
        // indirect load L/LC [AR1, P#x.y] through the packed pointer
        int32_t w = lastIndirectWidth;
        if (w == 1) {
          v = builder.create<awl::LoadBitIndirectOp>(
              loc(), builder.getI1Type(), addr);
        } else {
          mlir::Value loaded = builder.create<awl::LoadIndirectOp>(
              loc(), builder.getI64Type(), addr, builder.getI32IntegerAttr(w));
          v = castTo(loaded, builder.getIntegerType(w));
        }
      } else {
        // pseudo operands (DBLG/DBNO/DILG) already yield a constant
        v = mlir::isa<mlir::scl::DirectType>(addr.getType())
                ? (mlir::Value)builder.create<awl::LoadOp>(loc(), addr)
                : addr;
      }
    }
    if (!v)
      return false;
    acc.push_back(v);
    // L does not modify the RLO (S7 semantics): the current RLO value is
    // preserved for a following timer/counter/logic instruction
    return true;
  }

  bool parseStore() {
    mlir::Value addr = parseOperand(false);
    if (!addr)
      return false;
    // T does not pop ACCU1 (S7 semantics): several T can store the same value
    mlir::Value v = acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back();
    if (addr.getType().isInteger(64)) {
      // indirect store T [AR1, P#x.y]
      builder.create<awl::StoreIndirectOp>(
          loc(), castTo(v, builder.getI64Type()), addr,
          builder.getI32IntegerAttr(lastIndirectWidth));
    } else if (mlir::isa<mlir::scl::DirectType>(addr.getType())) {
      builder.create<awl::StoreOp>(loc(), v, addr);
    } else {
      popAcc(); // store to a pseudo operand: value dropped
    }
    rloActive = false;
    return true;
  }

  bool parseRloLogic(const std::string &m) {
    // U/O TRUE/FALSE: constant RLO operand (awlsim extension)
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "TRUE" || w == "FALSE") {
        lexer.next();
        if (!ensureRlo())
          return false;
        auto i1 = builder.getI1Type();
        mlir::Value c = builder.create<awl::ConstOp>(
            loc(), i1, builder.getIntegerAttr(i1, w == "TRUE" ? 1 : 0));
        if (m == "UN" || m == "ON")
          c = builder.create<awl::NotOp>(loc(), i1, c);
        bool isOr = (m == "O" || m == "ON");
        rlo = isOr ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, rlo, c)
                   : (mlir::Value)builder.create<awl::AndOp>(loc(), i1, rlo, c);
        return true;
      }
    }
    mlir::Value addr = parseBitOperand();
    if (!addr)
      return false;
    if (!ensureRlo())
      return false;
    bool isOr = (m == "O" || m == "ON");
    bool neg = (m == "UN" || m == "ON");
    auto i1 = builder.getI1Type();
    if (addr.getType().isInteger(64)) {
      // U/O [AR1, P#x.y]: indirect bit through the packed pointer
      mlir::Value bit =
          builder.create<awl::LoadBitIndirectOp>(loc(), i1, addr);
      if (neg)
        bit = builder.create<awl::NotOp>(loc(), i1, bit);
      rlo = isOr ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, rlo, bit)
                 : (mlir::Value)builder.create<awl::AndOp>(loc(), i1, rlo, bit);
      return true;
    }
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
    if (!rloActive) // e.g. MCR) without a preceding MCR(
      return true;
    mlir::Value group = rlo;
    auto i1 = builder.getI1Type();
    rlo = b.isXor ? (mlir::Value)builder.create<awl::XOrOp>(loc(), i1, b.savedRlo, group)
          : b.isOr ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, b.savedRlo, group)
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
    if (addr.getType().isInteger(64))
      builder.create<awl::WriteBitIndirectOp>(loc(), rlo, addr);
    else
      builder.create<awl::AssignOp>(loc(), rlo, addr);
    rloActive = false;
    return true;
  }

  bool parseSetReset(bool set) {
    mlir::Value v = parseOperand(false);
    if (!v)
      return false;
    auto dt = v.getType().dyn_cast<mlir::scl::DirectType>();
    if (!dt && v.getType().isInteger(64)) {
      // S/R [AR1, P#x.y]: indirect bit store
      auto i1 = builder.getI1Type();
      builder.create<awl::WriteBitIndirectOp>(
          loc(), builder.create<awl::ConstOp>(
                     loc(), i1, builder.getIntegerAttr(i1, set ? 1 : 0)),
          v);
      rlo = builder.create<awl::ConstOp>(
          loc(), i1, builder.getIntegerAttr(i1, set ? 1 : 0));
      rloActive = true;
      return true;
    }
    if (!dt)
      return error("expected a direct or indirect operand");
    // `S C n` sets the counter to ACC1, `R C n` resets it to 0
    if (dt.getArea() == mlir::scl::DirectArea::Counter && dt.getWidth() != 1) {
      mlir::Value val;
      int op;
      if (set) {
        // S C n reads ACCU1 as the new count but does not pop it (S7)
        val = acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back();
        op = 2; // SCL_RT_COUNTER_SET
      } else {
        auto i16 = builder.getI16Type();
        val = builder.create<awl::ConstOp>(loc(), i16,
                                            builder.getIntegerAttr(i16, 0));
        op = 3; // SCL_RT_COUNTER_RESET
      }
      builder.create<awl::CounterOp>(
          loc(), mlir::TypeRange{}, mlir::ValueRange{val, v, rloInput()},
          builder.getNamedAttr("op", builder.getI32IntegerAttr(op)));
      // S/R on a counter do not modify the RLO (awlsim semantics)
      return true;
    }
    if (dt.getArea() == mlir::scl::DirectArea::Timer && dt.getWidth() != 1) {
      // R T n resets the timer at runtime
      if (set)
        return error("S on a timer operand is not valid");
      builder.create<awl::TimerResetOp>(loc(), v);
      // R T n does not modify the RLO
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

  // SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA [n]: shift/rotate ACCU1 by n
  // (immediate, or ACCU1 if no immediate was given: `L val`, `L n`, `SLD`)
  bool parseShift(const std::string &m) {
    padAcc();
    mlir::Value v = popAcc();
    bool throughCarry = m == "RLDA" || m == "RRDA";
    mlir::Value amount = constInt(builder.getI32Type(), 1);
    if (!throughCarry) {
      if (lexer.getCurToken() == Tok::integer) {
        amount = constInt(builder.getI32Type(), (int64_t)lexer.getInteger());
        lexer.next();
      } else {
        amount = popAcc(); // no immediate: shift count is in ACCU2
        amount = castTo(amount, builder.getI32Type());
      }
    }
    v = castTo(v, builder.getI32Type());
    std::string kind = m;
    for (auto &c : kind)
      c = (char)tolower((unsigned char)c);
    acc.push_back(builder.create<awl::ShiftOp>(
        loc(), builder.getI32Type(), builder.getStringAttr(kind), v, amount));
    rloActive = false;
    return true;
  }

  // ABS/NEGR/SIN/COS/TAN/ASIN/ACOS/ATAN/SQRT/EXP/LN: unary real math on ACCU1
  bool parseRealMath(const std::string &m) {
    mlir::Value a = popAcc();
    if (!a)
      return false;
    a = castTo(a, builder.getF32Type());
    std::string kind = m;
    for (auto &c : kind)
      c = (char)tolower((unsigned char)c);
    acc.push_back(builder.create<awl::RealMathOp>(
        loc(), builder.getF32Type(), builder.getStringAttr(kind), a));
    rloActive = false;
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
    // symbolic/indirect operands: coerce to a valid timer index
    if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(t.getType())) {
      if (dt.getArea() != mlir::scl::DirectArea::Timer)
        t = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Timer,
                                          0, 0, 0, 16);
    } else {
      t = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Timer,
                                        0, 0, 0, 16);
    }
    // SclRtTimerMode: SP=0, SE=1, SS=2, SV=3, SI=4, SA=5
    int mode = m == "SP" ? 0 : m == "SE" ? 1 : m == "SS" ? 2
               : m == "SV" ? 3 : m == "SI" ? 4 : 5;
    builder.create<awl::TimerStartOp>(
        loc(), mlir::TypeRange{}, mlir::ValueRange{dur, t, rloInput()},
        builder.getNamedAttr("mode", builder.getI32IntegerAttr(mode)));
    return true;
  }

  // ZV/ZR Z n: increment/decrement counter n
  bool parseCounterZW(bool up) {
    mlir::Value c = parseOperand(false);
    if (!c)
      return false;
    // symbolic/indirect operands: coerce to a valid counter index
    if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(c.getType())) {
      if (dt.getArea() != mlir::scl::DirectArea::Counter)
        c = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Counter,
                                          0, 0, 0, 16);
    } else {
      c = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Counter,
                                        0, 0, 0, 16);
    }
    auto i16 = builder.getI16Type();
    mlir::Value zero = builder.create<awl::ConstOp>(
        loc(), i16, builder.getIntegerAttr(i16, 0));
    builder.create<awl::CounterOp>(
        loc(), mlir::TypeRange{}, mlir::ValueRange{zero, c, rloInput()},
        builder.getNamedAttr("op", builder.getI32IntegerAttr(up ? 0 : 1)));
    return true;
  }

  enum class ArithKind { Add, Sub, Mul, Div };

  bool parseArith(ArithKind k, char w) {
    mlir::Type ty = widthType(w);
    padAcc();
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
    // AW/OW/XOW operate on words (i16), AD/OD/XOD/UD on double words (i32)
    auto ty = (m.size() == 3 && m[2] == 'D') ? builder.getI32Type()
                                             : builder.getI16Type();
    mlir::Value b, a;
    if (lexer.getCurToken() == Tok::integer || lexer.getCurToken() == Tok::real) {
      // immediate operand: ACCU1 op imm
      b = parseConst();
      if (acc.empty())
        acc.push_back(constInt(builder.getI32Type(), 0));
      a = popAcc();
    } else {
      padAcc();
      b = popAcc();
      a = popAcc();
    }
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
    mlir::Value a, b;
    if (lexer.getCurToken() == Tok::integer || lexer.getCurToken() == Tok::real) {
      // immediate operand: ACC1 cmp imm
      b = castTo(parseConst(), ty);
      if (acc.empty())
        acc.push_back(constInt(builder.getI32Type(), 0));
      a = castTo(popAcc(), ty);
    } else {
      padAcc();
      a = castTo(acc[acc.size() - 2], ty); // ACC2
      b = castTo(acc[acc.size() - 1], ty); // ACC1
    }
    if (!a || !b)
      return false;
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

  // __ASSERT<cmp>[R] <lhs>, <rhs> — awlsim test oracle. lhs is __ACCU n or
  // a memory operand (loaded); __STW bit asserts are skipped (the sim does
  // not track status-word bits). Lowered to awl.assert / awl.assert_r.
  bool parseAssert() {
    uint32_t id = (uint32_t)lexer.curLine();
    Tok cmp = lexer.getCurToken();
    if (cmp != Tok::eq && cmp != Tok::ne && cmp != Tok::lt && cmp != Tok::le &&
        cmp != Tok::gt && cmp != Tok::ge) {
      lexer.nextLine(); // not an assert form we know: skip the line
      return true;
    }
    lexer.next();
    bool isReal = false;
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "R")
        isReal = true;
      if (w == "R" || w == "I" || w == "D") // optional width letter
        lexer.next();
    }
    int op;
    switch (cmp) {
    case Tok::eq: op = 0; break;
    case Tok::ne: op = 1; break;
    case Tok::gt: op = 2; break;
    case Tok::lt: op = 3; break;
    case Tok::ge: op = 4; break;
    default: op = 5; break;
    }
    mlir::Value actual;
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "__STW" || w == "BIE" || w == "OV" || w == "OS" || w == "VKE" ||
          w == "A0" || w == "A1" || w == "RLO" || w == "__DBR") {
        lexer.nextLine(); // status-word / DB-register bits not tracked: skip
        return true;
      }
      if (w == "__AR") { // address register value (packed or raw P# pointer)
        lexer.next();
        int n = 1;
        if (lexer.getCurToken() == Tok::integer) {
          n = (int)lexer.getInteger();
          lexer.next();
        }
        actual = builder.create<awl::ArGetOp>(loc(), builder.getI64Type(),
                                              builder.getI32IntegerAttr(n));
        goto haveLhs;
      }
      if (w == "__ACCU") {
        lexer.next();
        int n = 1;
        if (lexer.getCurToken() == Tok::integer) {
          n = (int)lexer.getInteger();
          lexer.next();
        }
        actual = (n >= 1 && (size_t)n <= acc.size())
                     ? acc[acc.size() - n]
                     : constInt(builder.getI32Type(), 0);
      } else {
        mlir::Value addr = parseOperand(false);
        if (!addr) {
          lexer.nextLine();
          return true;
        }
        actual = mlir::isa<mlir::scl::DirectType>(addr.getType())
                     ? (mlir::Value)builder.create<awl::LoadOp>(loc(), addr)
                     : addr;
      }
    haveLhs:;
    } else if (lexer.getCurToken() == Tok::integer ||
               lexer.getCurToken() == Tok::real ||
               lexer.getCurToken() == Tok::minus) {
      // constant lhs (e.g. __ASSERT== 1, 2 or a pointer literal P#E 2.1, 17)
      bool bcd = lexer.curBcdLit();
      mlir::Value c = parseConst();
      if (!c) {
        lexer.nextLine();
        return true;
      }
      if (bcd) // C#122 in an assert compares the decoded count
        c = bcdToInt(c, c.getType(), 3);
      actual = c;
    } else {
      lexer.nextLine(); // unsupported lhs form: skip the line
      return true;
    }
    if (lexer.getCurToken() != Tok::comma) {
      lexer.nextLine();
      return true;
    }
    lexer.next();
    mlir::Value expected;
    // rhs may be another accumulator: __ASSERT== __ACCU 1, __ACCU 2
    if (lexer.getCurToken() == Tok::ident && upperIdent() == "__ACCU") {
      lexer.next();
      int n = 1;
      if (lexer.getCurToken() == Tok::integer) {
        n = (int)lexer.getInteger();
        lexer.next();
      }
      expected = (n >= 1 && (size_t)n <= acc.size())
                     ? acc[acc.size() - n]
                     : constInt(builder.getI32Type(), 0);
    } else if (lexer.getCurToken() == Tok::ident) {
      // memory operand rhs, e.g. __ASSERT== __ACCU 1, MD 8
      mlir::Value addr = parseOperand(false);
      if (!addr) {
        lexer.nextLine();
        return true;
      }
      expected = mlir::isa<mlir::scl::DirectType>(addr.getType())
                     ? (mlir::Value)builder.create<awl::LoadOp>(loc(), addr)
                     : addr;
    } else {
      bool bcd = lexer.curBcdLit();
      mlir::Value c = parseConst();
      if (!c) {
        lexer.nextLine();
        return true;
      }
      if (bcd) // C#122 in an assert compares the decoded count
        c = bcdToInt(c, c.getType(), 3);
      expected = c;
    }
    auto idAttr = builder.getI32IntegerAttr((int32_t)id);
    auto opAttr = builder.getI8IntegerAttr((int8_t)op);
    // awlsim compares as real when the R suffix is given or either side
    // is a real literal (e.g. __ASSERT== __ACCU 1, 0.0)
    bool real = isReal || mlir::isa<mlir::FloatType>(actual.getType()) ||
                mlir::isa<mlir::FloatType>(expected.getType());
    if (real) {
      actual = castTo(actual, builder.getF64Type());
      expected = castTo(expected, builder.getF64Type());
      builder.create<awl::AssertROp>(loc(), idAttr, opAttr, actual, expected);
    } else {
      actual = castTo(actual, builder.getI64Type());
      expected = castTo(expected, builder.getI64Type());
      builder.create<awl::AssertOp>(loc(), idAttr, opAttr, actual, expected);
    }
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
      return error("expected FC/FB/SFC after CALL");
    std::string orig = upperIdent();
    std::string kind = orig;
    lexer.next();
    // awlsim writes the block number attached: CALL SFC47(...)
    int64_t suffixed = 0;
    bool hasSuffix = false;
    for (const char *p : {"SFC", "SFB", "FC", "FB"}) {
      size_t l = strlen(p);
      if (orig.size() > l && orig.compare(0, l, p) == 0 &&
          llvm::all_of(llvm::drop_begin(orig, l),
                       [](char c) { return isdigit((unsigned char)c); })) {
        suffixed = atoll(orig.c_str() + l);
        hasSuffix = true;
        kind = p;
        break;
      }
    }
    if (kind != "FC" && kind != "FB" && kind != "SFC" && kind != "SFB") {
      // symbolic block name: CALL "GE_DT" (params) — callee referenced by name
      if (lexer.getCurToken() == Tok::comma) { // CALL "FB", "Instance DB" (..)
        lexer.next();
        if (lexer.getCurToken() == Tok::ident)
          lexer.next();
      }
      if (lexer.getCurToken() == Tok::lparen) {
        int depth = 0;
        do {
          if (lexer.getCurToken() == Tok::lparen)
            ++depth;
          else if (lexer.getCurToken() == Tok::rparen)
            --depth;
          else if (lexer.getCurToken() == Tok::eof)
            return error("unterminated CALL parameter list");
          lexer.next();
        } while (depth > 0);
      }
      llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), orig)));
      builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                  cond ? mlir::ValueRange{rloVal}
                                       : mlir::ValueRange{},
                                  attrs);
      if (!cond)
        rloActive = false;
      return true;
    }
    if (kind == "SFC" || kind == "SFB") {
      // system block call: CALL SFC/SFB [-]n [(params)] — accepted
      // syntactically; SFCs/SFBs are host-provided (runtime stub), a
      // documented gap for the JIT
      bool neg = false;
      if (lexer.getCurToken() == Tok::minus) {
        lexer.next();
        neg = true;
      }
      int64_t n;
      if (hasSuffix) {
        n = suffixed;
      } else {
        if (lexer.getCurToken() != Tok::integer)
          return error("expected SFC/SFB number after CALL");
        n = lexer.getInteger();
        lexer.next();
      }
      if (neg)
        n = -n;
      // SFB with instance DB: CALL SFB n, DB m
      if (kind == "SFB" && lexer.getCurToken() == Tok::comma) {
        lexer.next();
        if (lexer.getCurToken() == Tok::ident && upperIdent() == "DB")
          lexer.next();
        if (lexer.getCurToken() == Tok::integer)
          lexer.next();
      }
      llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  "SFC" + std::to_string(n))));
      attrs.push_back(
          builder.getNamedAttr("sfc", builder.getI32IntegerAttr((int32_t)n)));
      if (lexer.getCurToken() == Tok::lparen) {
        int depth = 0;
        do {
          if (lexer.getCurToken() == Tok::lparen)
            ++depth;
          else if (lexer.getCurToken() == Tok::rparen)
            --depth;
          else if (lexer.getCurToken() == Tok::eof)
            return error("unterminated SFC parameter list");
          lexer.next();
        } while (depth > 0);
      }
      builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                  cond ? mlir::ValueRange{rloVal}
                                       : mlir::ValueRange{},
                                  attrs);
      return true;
    }
    int64_t num;
    if (hasSuffix) {
      num = suffixed;
    } else if (lexer.getCurToken() == Tok::lbracket) {
      // indirect block call: UC FC [MW 50] — block number is runtime data,
      // not modelled; call block 0
      lexer.next();
      while (lexer.getCurToken() != Tok::rbracket &&
             lexer.getCurToken() != Tok::eof)
        lexer.next();
      if (lexer.getCurToken() == Tok::rbracket)
        lexer.next();
      num = 0;
    } else {
      if (lexer.getCurToken() != Tok::integer)
        return error("expected block number after " + kind);
      num = lexer.getInteger();
      lexer.next();
    }
    std::string callee = kind + std::to_string(num);
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), callee)));
    if (kind == "FB" && lexer.getCurToken() == Tok::comma) {
      // CALL FB n, DB m: the instance data block number (optional in awlsim)
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
    // optional parameter list: CALL FC 1 (P1 := ..., P2 => ...)
    if (lexer.getCurToken() == Tok::lparen) {
      int depth = 0;
      do {
        if (lexer.getCurToken() == Tok::lparen)
          ++depth;
        else if (lexer.getCurToken() == Tok::rparen)
          --depth;
        else if (lexer.getCurToken() == Tok::eof)
          return error("unterminated CALL parameter list");
        lexer.next();
      } while (depth > 0);
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
