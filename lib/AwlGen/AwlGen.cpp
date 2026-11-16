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

// Named DBs from the command line / external symbol tables (awlsim
// .awlpro "Symbol table"), consulted before the source-declared namedDbs.
static std::map<std::string, int32_t> gCliNamedDbs;

void awlRegisterNamedDb(llvm::StringRef name, int32_t db) {
  std::string n = name.upper();
  gCliNamedDbs[n] = db;
}

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
  int getLitWidth() const { return cur_.litWidth; }
  double getReal() const { return cur_.realVal; }
  int line() const { return lineNo; }
  llvm::StringRef raw() const { return cur; } // unconsumed input
  int col() const { return colNo; }
  // line of the current token (lineNo itself is ahead due to lookahead)
  int curLine() const { return cur_.line; }
  bool curQuoted() const { return cur_.quoted; }
  char curQuoteChar() const { return cur_.quoteChar; }
  bool curBcdLit() const { return cur_.bcdLit; }
  bool curPtrLocal() const { return cur_.ptrLocal; }
  llvm::StringRef curPtrSym() const { return cur_.ptrSym; }
  bool curPtrLit() const { return cur_.ptrLit; }
  int curPtrTypeCode() const { return cur_.ptrTypeCode; }
  int curPtrCount() const { return cur_.ptrCount; }

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
    char quoteChar = '\''; // which quote: '...' literal vs "..." name
    bool bcdLit = false; // C#... counter literal (raw BCD pattern)
    bool ptrLocal = false; // P##name: pointer to a block-local variable
    llvm::StringRef ptrSym;  // P#DB1.SYM...: symbolic member part (raw text)
    bool ptrLit = false;     // token came from a P#x.y pointer literal
    int ptrTypeCode = -1;    // P#M 10.0 BYTE 7: ANY type code of the literal
    int ptrCount = 0;        // ...and repetition count
    int litWidth = 0;    // B#/W#/L# typed literal: 8/16/32
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
          unit = 1; // ms
          i += 2;
        } else if (i < s.size() && s[i] == 'H') {
          unit = 3600000;
          ++i;
        } else if (i < s.size() && s[i] == 'D') {
          unit = 86400000;
          ++i;
        } else if (i < s.size() && s[i] == 'M') {
          unit = 60000;
          ++i;
        } else if (i < s.size() && s[i] == 'S') {
          unit = 1000;
          ++i;
        }
        if (!anyDigit || unit < 0)
          ok = false;
        else
          total += num * unit;
      }
      if (!ok || total < 0 || total > 9990LL * 1000) { // max 9990s in ms
        llvm::errs() << file << ":" << lineNo << ":" << colNo
                     << ": bad S5TIME literal 'S5T#" << s << "'\n";
        t.tok = Tok::eof;
        return t;
      }
      // awlsim Timer_seconds_to_s5t: finest timebase that covers the value
      // (total is in ms): <=9.99s -> 10ms base, <=99.9s -> 100ms,
      // <=999s -> 1s, else 10s
      int code;
      int64_t val;
      if (total <= 9990) {
        code = 0;
        val = total / 10;
      } else if (total <= 99900) {
        code = 1;
        val = total / 100;
      } else if (total <= 999000) {
        code = 2;
        val = total / 1000;
      } else {
        code = 3;
        val = total / 10000;
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
        // awlsim named constants (exact names — PINF/NINF must not match PI)
        if (t.ident == "__CNST_PI")
          t.realVal = 3.14159265358979;
        else if (t.ident == "__CNST_E")
          t.realVal = 2.71828182845905;
        else if (t.ident == "__CNST_PINF")
          t.realVal = __builtin_huge_val();
        else if (t.ident == "__CNST_NINF")
          t.realVal = -__builtin_huge_val();
        else if (t.ident == "__CNST_PNAN" || t.ident == "__CNST_NNAN") {
          // awlsim floatConst: pNaNDWord = 0x7FFFFFFF, nNaNDWord = 0xFFFFFFFF
          uint32_t bits = t.ident == "__CNST_PNAN" ? 0x7FFFFFFFu : 0xFFFFFFFFu;
          float f;
          std::memcpy(&f, &bits, 4);
          t.realVal = f;
        }
        else {
          t.intVal = 0;
          t.tok = Tok::integer;
          return t;
        }
        t.tok = Tok::real;
        return t;
      }
      // awlsim test directives (__FEATURE, __STWRST, ...): skip the line.
      // __ASSERT... is kept as an ident — the parser lowers it to a runtime
      // oracle check (awl.assert), see parseAssert.
      if (t.ident.starts_with("__") && !t.ident.starts_with("__ASSERT") &&
          t.ident != "__ACCU" && t.ident != "__STW" && t.ident != "__AR" &&
          t.ident != "__DBR" && t.ident != "__SLEEP" &&
          t.ident != "__STWRST" && t.ident != "__FEATURE") {
        while (!cur.empty() && cur.front() != '\n')
          cur = cur.drop_front();
        return scan();
      }
      // typed integer literals: B#.. (i8), W#.. (i16), L#.. (i32); the value
      // may itself be base-prefixed (B#16#91) or C-style hex (L#0x2A)
      if (t.ident.size() == 1 &&
          (t.ident == "B" || t.ident == "W" || t.ident == "L") &&
          cur.size() >= 2 && cur.front() == '#' &&
          (isdigit((unsigned char)cur[1]) || cur[1] == '-')) {
        cur = cur.drop_front(); // '#'
        bool neg = false;
        if (!cur.empty() && cur.front() == '-') {
          cur = cur.drop_front();
          neg = true;
        }
        int64_t lead = 0;
        while (!cur.empty() && isdigit((unsigned char)cur.front())) {
          lead = lead * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        int64_t v = lead;
        if (!cur.empty() && cur.front() == '#') {
          cur = cur.drop_front();
          int b = (int)lead;
          v = 0;
          if (b == 16) {
            if (cur.size() >= 2 && cur.front() == '0' &&
                tolower((unsigned char)cur[1]) == 'x')
              cur = cur.drop_front(2);
            while (!cur.empty() && isxdigit((unsigned char)cur.front())) {
              char h = cur.front();
              int d = isdigit((unsigned char)h) ? h - '0' : tolower((unsigned char)h) - 'a' + 10;
              v = v * 16 + d;
              cur = cur.drop_front();
            }
          } else if (b == 8 || b == 2) {
            while (!cur.empty() && cur.front() >= '0' &&
                   cur.front() < (char)('0' + b)) {
              v = v * b + (cur.front() - '0');
              cur = cur.drop_front();
            }
          } else {
            while (!cur.empty() && isdigit((unsigned char)cur.front())) {
              v = v * 10 + (cur.front() - '0');
              cur = cur.drop_front();
            }
          }
        }
        if (neg)
          v = -v;
        t.intVal = v;
        t.tok = Tok::integer;
        t.litWidth = t.ident == "B" ? 8 : t.ident == "W" ? 16 : 32;
        return t;
      }
      // DATE/TIME literals: D#1990-1-1 (days since 1990-01-01), T#2s (ms),
      // TOD#h:m:s.ms (ms since midnight), DT#y-mo-d-h:mi:s.ms (8 BCD bytes)
      if ((t.ident == "D" || t.ident == "T" || t.ident == "DT" ||
           t.ident == "TOD" || t.ident == "DATE" || t.ident == "TIME" ||
           t.ident == "TIME_OF_DAY" || t.ident == "DATE_AND_TIME") &&
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
          // components <digits><unit>, units MS/S/M/H/D -> total ms
          bool neg = false;
          size_t i = 0;
          if (!s.empty() && s[0] == '-') {
            neg = true;
            i = 1;
          }
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
            } else if (i < s.size() && s[i] == 'D') {
              unit = 86400000;
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
          if (neg)
            v = -v;
        } else if (t.ident == "TOD" || t.ident == "TIME_OF_DAY") {
          // TOD#h:m:s.ms -> ms since midnight (awlsim float msec semantics)
          int h = 0, mi = 0;
          double fsec = 0.0;
          if (sscanf(s.c_str(), "%d:%d:%lf", &h, &mi, &fsec) >= 2) {
            int sec = (int)fsec;
            int ms = (int)(fsec * 1000) - sec * 1000;
            v = ((int64_t)h * 3600 + mi * 60 + sec) * 1000 + ms;
          }
        } else if (t.ident == "DT" || t.ident == "DATE_AND_TIME") {
          // DT#y-mo-d-h:mi:s.ms -> 8 bytes, BCD-packed, little-endian in
          // the integer (byte 0 = year BCD, ... byte 7 = ms-low + dow)
          int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
          double fsec = 0.0;
          if (sscanf(s.c_str(), "%d-%d-%d-%d:%d:%lf", &y, &mo, &d, &h, &mi,
                     &fsec) >= 3) {
            // awlsim semantics: msec = int(fsec*1000) - int(fsec)*1000
            sec = (int)fsec;
            int ms = (int)(fsec * 1000) - sec * 1000;
            auto bcd = [](int n) { return ((n / 10) << 4) | (n % 10); };
            auto daysFromCivil = [](int yy, int m, int dd) -> int64_t {
              yy -= m <= 2;
              int64_t era = (yy >= 0 ? yy : yy - 399) / 400;
              unsigned yoe = (unsigned)(yy - era * 400);
              unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + dd - 1;
              unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
              return era * 146097 + (int64_t)doe - 719468;
            };
            int64_t days = daysFromCivil(y, mo, d);
            int dow = (int)((days + 4) % 7) + 1; // 1=Sunday .. 7=Saturday (1970-01-01 = Thursday = 5)
            uint64_t b[8] = {(uint64_t)bcd(y % 100), (uint64_t)bcd(mo),
                             (uint64_t)bcd(d), (uint64_t)bcd(h),
                             (uint64_t)bcd(mi), (uint64_t)bcd(sec),
                             (uint64_t)bcd(ms / 10),
                             (uint64_t)bcd(ms % 10) * 16 + (uint64_t)dow};
            v = 0;
            for (int i = 0; i < 8; ++i)
              v |= (int64_t)b[i] << (8 * (7 - i));
          }
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
        // width for zero-extension into ACCU: DATE=16-bit, T#/TOD=32-bit,
        // DT/DATE_AND_TIME stays 64-bit (litWidth 0 -> i64 in parseConst)
        t.litWidth = (t.ident == "D" || t.ident == "DATE")
                         ? 16
                         : (t.ident == "DT" || t.ident == "DATE_AND_TIME")
                               ? 0
                               : 32;
        return t;
      }
      // Pointer literal P#... — S7 32-bit pointer format:
      //   area<<24 | db<<16 | byte<<3 | bit
      // area codes: P=0x80 E/I=0x81 A/Q=0x82 M=0x83 DB=0x84 L=0x86.
      // Without an area letter (bare P#10.6) the value is the raw
      // byte*8+bit offset (area 0), used for AR arithmetic offsets.
      if (t.ident == "P" && !cur.empty() && cur.front() == '#') {
        cur = cur.drop_front();
        if (!cur.empty() && cur.front() == '#') {
          // P##<ident>: pointer to a symbolic local — resolved by the
          // parser (area DI for FBs / L otherwise, member offset)
          cur = cur.drop_front();
          auto begin = cur.begin();
          while (!cur.empty() && (isalnum(cur.front()) || cur.front() == '_'))
            cur = cur.drop_front();
          t.ident = llvm::StringRef(begin, cur.begin() - begin);
          t.tok = Tok::localvar;
          t.ptrLocal = true;
          return t;
        }
        auto skipWs = [&]() {
          while (!cur.empty() && (cur.front() == ' ' || cur.front() == '\t'))
            cur = cur.drop_front();
        };
        auto skipQuoted = [&]() {
          if (!cur.empty() && cur.front() == '"') {
            cur = cur.drop_front();
            while (!cur.empty() && cur.front() != '"' && cur.front() != '\n')
              cur = cur.drop_front();
            if (!cur.empty() && cur.front() == '"')
              cur = cur.drop_front();
          }
        };
        // optional area: P#DB50.... / P#DBX... / P#M... / P#E...
        int64_t area = 0, db = 0;
        skipWs();
        if (cur.size() >= 2 && toupper((unsigned char)cur[0]) == 'D' &&
            (toupper((unsigned char)cur[1]) == 'B' ||
             toupper((unsigned char)cur[1]) == 'I')) {
          // P#DB... = DB area 0x84, P#DI... = DI area 0x85 (instance DB)
          area = toupper((unsigned char)cur[1]) == 'B' ? 0x84 : 0x85;
          cur = cur.drop_front(2);
          while (!cur.empty() && isdigit((unsigned char)cur.front())) {
            db = db * 10 + (cur.front() - '0');
            cur = cur.drop_front();
          }
          // DBX: db number 0 = current DB register (area stays 0x84,
          // awlsim packs P#DBX 8.6 as DW#16#84000046); P#DIX is the DI
          // register (0x85). Remaining letters (DBB/DBW/DBD) are skipped.
          while (!cur.empty() && isalpha((unsigned char)cur.front()))
            cur = cur.drop_front(); // skips the X of DBX (db stays 0)
        } else if (!cur.empty() && isalpha((unsigned char)cur.front())) {
          char a = (char)toupper((unsigned char)cur.front());
          cur = cur.drop_front();
          switch (a) {
          case 'P': area = 0x80; break;
          case 'E': case 'I': area = 0x81; break;
          case 'A': case 'Q': area = 0x82; break;
          case 'M': area = 0x83; break;
          case 'L': area = 0x86; break;
          case 'V': area = 0x87; break; // VL: parent L-stack frame
          default: break; // not an area letter: symbolic name, skip below
          }
          if (area == 0)
            while (!cur.empty() &&
                   (isalnum((unsigned char)cur.front()) || cur.front() == '_'))
              cur = cur.drop_front();
        }
        // symbolic member parts: P#DB42.DTVAR1, P#"DB".VAR, [i]
        skipWs();
        const char *symBegin = cur.begin();
        while (true) {
          skipWs();
          if (!cur.empty() && cur.front() == '"') {
            skipQuoted();
          } else if (!cur.empty() && cur.front() == '.') {
            cur = cur.drop_front();
            skipQuoted();
            while (!cur.empty() &&
                   (isalnum((unsigned char)cur.front()) || cur.front() == '_'))
              cur = cur.drop_front();
          } else if (!cur.empty() && cur.front() == '[') {
            int depth = 0;
            do {
              if (cur.front() == '[')
                ++depth;
              else if (cur.front() == ']')
                --depth;
              cur = cur.drop_front();
            } while (depth > 0 && !cur.empty());
          } else if (!cur.empty() && isdigit((unsigned char)cur.front())) {
            break; // byte number follows
          } else {
            break;
          }
        }
        const char *symEnd = cur.begin();
        if (symEnd > symBegin)
          t.ptrSym = llvm::StringRef(symBegin, symEnd - symBegin);
        skipWs();
        int64_t byte = 0;
        while (!cur.empty() && isdigit((unsigned char)cur.front())) {
          byte = byte * 10 + (cur.front() - '0');
          cur = cur.drop_front();
        }
        int64_t bit = 0;
        if (!cur.empty() && cur.front() == '.') {
          cur = cur.drop_front();
          if (!cur.empty() && isdigit((unsigned char)cur.front())) {
            bit = cur.front() - '0';
            cur = cur.drop_front();
          }
        }
        // ANY-pointer type/length suffix: P#M 10.0 BYTE 7 / P#M 11.1 BOOL 13
        // — consume a known type word plus an optional length number
        skipWs();
        static const char *typeWords[] = {"BOOL", "BYTE", "WORD", "DWORD",
                                          "CHAR", "INT", "DINT", "REAL",
                                          "DATE", "TOD", "TIME", "S5T",
                                          "STRING", "TIMER", "COUNTER"};
        static const int typeCodes[] = {0x01, 0x02, 0x04, 0x06, 0x03, 0x05,
                                        0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
                                        0x13, 0x1D, 0x1C};
        for (size_t ti = 0; ti < sizeof(typeWords) / sizeof(*typeWords);
             ++ti) {
          const char *tw = typeWords[ti];
          size_t tl = strlen(tw);
          if (cur.size() >= tl && cur.substr(0, tl).equals_insensitive(tw)) {
            cur = cur.drop_front(tl);
            t.ptrTypeCode = typeCodes[ti];
            skipWs();
            int64_t cnt = 0;
            bool hasCnt = false;
            while (!cur.empty() && isdigit((unsigned char)cur.front())) {
              cnt = cnt * 10 + (cur.front() - '0');
              cur = cur.drop_front();
              hasCnt = true;
            }
            t.ptrCount = hasCnt ? (int)cnt : 1;
            break;
          }
        }
        t.intVal = (area << 24) | (db << 16) | (byte * 8 + bit);
        t.tok = Tok::integer;
        t.ptrLit = true;
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
        // paren literal: B#(1, 2) = 16#0102, W#(1,2,3,4) = 32#01020304 —
        // the comma-separated numbers combine big-endian
        if (!cur.empty() && cur.front() == '(') {
          cur = cur.drop_front();
          int64_t v = 0;
          while (!cur.empty() && cur.front() != ')') {
            if (isdigit((unsigned char)cur.front())) {
              int64_t x = 0;
              while (!cur.empty() && isdigit((unsigned char)cur.front())) {
                x = x * 10 + (cur.front() - '0');
                cur = cur.drop_front();
              }
              v = v * 256 + x;
            } else {
              cur = cur.drop_front();
            }
          }
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
        } else {
          while (!cur.empty() && isdigit(cur.front())) {
            v = v * 10 + (cur.front() - '0');
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
      t.quoteChar = '\'';
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
      t.quoteChar = '"';
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

// MARK: data model (S7 type layouts for UDTs / DBs / FB-FC interfaces)

struct Layout;

struct MemberInfo {
  int32_t byte = 0, bit = 0, widthBits = 0; // offset within the parent
  bool isBit = false;
  uint8_t section = 0; // 0=IN 1=OUT 2=INOUT 3=STAT 4=TEMP
  std::string type;      // base type name (BOOL, INT, ARRAY, UDT name, ...)
  int32_t count = 1;     // ARRAY element count (all dims multiplied)
  int32_t stride = 0;    // ARRAY element size in bytes
  std::string elemType;  // ARRAY element type
  int32_t lo = 1;        // ARRAY lower bound of the first dimension
  int32_t elemSize = 0;  // ARRAY element size in bytes
  std::vector<std::pair<int32_t, int32_t>> dims; // ARRAY (lo, hi) per dim
  std::shared_ptr<Layout> nested;     // inline STRUCT member layout
  std::shared_ptr<Layout> elemNested; // ARRAY OF STRUCT element layout
};

struct Layout {
  std::map<std::string, MemberInfo> members;
  std::map<std::string, std::string> memberInit; // raw ":= ..." initializer text
  int32_t size = 0;      // total size in bytes (rounded to even)
  int32_t bitCursor = 0; // absolute bit offset for the next member (BOOL packing)
};

// block header keywords (TITLE = ..., AUTHOR : ..., VERSION : ...):
// such a line is skipped entirely
static bool isHeaderKwLine(const std::string &w) {
  return w == "TITLE" || w == "AUTHOR" || w == "FAMILY" || w == "NAME" ||
         w == "VERSION" || w == "NETWORK" || w == "KNOW_HOW_PROTECT";
}

static std::string upperStr(llvm::StringRef s) {
  std::string r(s);
  for (auto &c : r)
    c = (char)toupper((unsigned char)c);
  return r;
}

// A type name is an identifier, optionally followed by a number
// ("UDT 1"); the name is the uppercased ident + " " + number.
static std::string typeNameOf(Lexer &lx) {
  std::string t = lx.getCurToken() == Tok::ident ? upperStr(lx.getIdent()) : "";
  lx.next();
  if (lx.getCurToken() == Tok::integer) {
    t += " " + std::to_string(lx.getInteger());
    lx.next();
  }
  return t;
}


// Registry of FB layouts, set during the pre-scan so that typeInfo can size
// FB/SFB-typed members (multi-instance variables). The pre-scan runs twice:
// the first pass collects all block layouts, the second recomputes member
// offsets with the FB sizes known.
static std::map<int32_t, Layout> *gFbLayouts = nullptr;

// Standard instance-DB layout of a system function block (SFB 0-5:
// CTU/CTD/CTUD counters, TON/TOF/TP timers)
static Layout sfbLayoutOf(int32_t n) {
  Layout L;
  // sections: 0 = IN, 1 = OUT, 3 = STAT (the CALL param direction comes
  // from the section, awlsim-style)
  auto bit = [&](const char *name, int32_t byte, int32_t b, uint8_t sec) {
    MemberInfo m;
    m.type = "BOOL";
    m.isBit = true;
    m.widthBits = 1;
    m.byte = byte;
    m.bit = b;
    m.section = sec;
    L.members[name] = m;
  };
  auto word = [&](const char *name, int32_t byte, const char *ty, int32_t wb,
                  uint8_t sec) {
    MemberInfo m;
    m.type = ty;
    m.widthBits = wb;
    m.byte = byte;
    m.section = sec;
    L.members[name] = m;
  };
  if (n == 0) { // SFB0 CTU (awlsim system_sfb_0)
    bit("CU", 0, 0, 0);
    bit("R", 0, 1, 0);
    word("PV", 2, "INT", 16, 0);
    bit("Q", 4, 0, 1);
    word("CV", 6, "INT", 16, 1);
    bit("CUO", 8, 0, 3);
    L.size = 10;
  } else if (n == 1) { // SFB1 CTD (awlsim system_sfb_1)
    bit("CD", 0, 0, 0);
    bit("LOAD", 0, 1, 0);
    word("PV", 2, "INT", 16, 0);
    bit("Q", 4, 0, 1);
    word("CV", 6, "INT", 16, 1);
    bit("CDO", 8, 0, 3);
    L.size = 10;
  } else if (n == 2) { // SFB2 CTUD (awlsim system_sfb_2)
    bit("CU", 0, 0, 0);
    bit("CD", 0, 1, 0);
    bit("R", 0, 2, 0);
    bit("LOAD", 0, 3, 0);
    word("PV", 2, "INT", 16, 0);
    bit("QU", 4, 0, 1);
    bit("QD", 4, 1, 1);
    word("CV", 6, "INT", 16, 1);
    bit("CUO", 8, 0, 3);
    bit("CDO", 8, 1, 3);
    L.size = 10;
  } else { // SFB3 TON / SFB4 TOF / SFB5 TP (awlsim system_sfb_3 layout;
    // TIME is a 32-bit type there, total instance size 22)
    bit("IN", 0, 0, 0);
    word("PT", 2, "TIME", 32, 0);
    bit("Q", 6, 0, 1);
    word("ET", 8, "TIME", 32, 1);
    word("STATE", 12, "BYTE", 8, 3);
    word("STIME", 14, "TIME", 32, 3);
    word("ATIME", 18, "TIME", 32, 3);
    L.size = 22;
  }
  return L;
}

// size/alignment/width of a (non-array) type; BOOL is bit-packed (size 0)
static void typeInfo(const std::string &ty, const std::map<std::string, Layout> &udts,
                     int32_t &size, int32_t &align, int32_t &widthBits,
                     bool &isBit) {
  isBit = false;
  if (ty == "BOOL") { size = 0; align = 1; widthBits = 1; isBit = true; return; }
  if (ty == "BYTE" || ty == "CHAR") { size = 1; align = 1; widthBits = 8; return; }
  if (ty == "WORD" || ty == "INT" || ty == "S5TIME" || ty == "TIME" ||
      ty == "DATE" || ty == "COUNTER" || ty == "TIMER") {
    size = 2; align = 2; widthBits = 16; return;
  }
  if (ty == "DWORD" || ty == "DINT" || ty == "REAL" || ty == "TOD" ||
      ty == "TIME_OF_DAY") {
    size = 4; align = 2; widthBits = 32; return;
  }
  if (ty == "DT" || ty == "DATE_AND_TIME") {
    size = 8; align = 2; widthBits = 64; return; // 8 BCD bytes
  }
  if (ty == "ANY") { size = 10; align = 2; widthBits = 80; return; } // ANY hdr
  if (ty == "POINTER") { size = 6; align = 2; widthBits = 48; return; }
  if (ty.rfind("STRING", 0) == 0) { // STRING[n]: n+2 bytes, even-aligned
    // (awlsim: element size is (n+2) rounded up to a multiple of 2)
    int n = 254;
    if (ty.size() > 6)
      n = atoi(ty.c_str() + 6);
    size = (n + 2 + 1) & ~1; align = 2; widthBits = size * 8; return;
  }
  auto it = udts.find(ty);
  if (it != udts.end()) {
    size = it->second.size;
    align = size >= 2 ? 2 : 1;
    widthBits = size * 8;
    return;
  }
  if (ty.rfind("FB ", 0) == 0 && gFbLayouts) {
    // multi-instance FB variable: occupies the referenced FB's instance size
    auto fit = gFbLayouts->find(atoi(ty.c_str() + 3));
    if (fit != gFbLayouts->end()) {
      size = fit->second.size;
      align = size >= 2 ? 2 : 1;
      widthBits = size * 8;
      return;
    }
  }
  if (ty.rfind("SFB ", 0) == 0) {
    Layout sl = sfbLayoutOf(atoi(ty.c_str() + 4));
    size = sl.size;
    align = size >= 2 ? 2 : 1;
    widthBits = size * 8;
    return;
  }
  size = 2; align = 2; widthBits = 16; // unknown type: word
}

// Parse a STRUCT/VAR body until one of the end keywords; assigns offsets
// per S7 rules (BOOL bit-packed; WORD/DWORD even-aligned).
// Compound types passed by reference (pointer header in the slot):
// ANY (10-byte header), POINTER/STRING/DATE_AND_TIME (6-byte DB-pointer).
// BLOCK_* refs and POINTER-typed... note POINTER itself is passed by value.
static bool isByRefType(const std::string &t) {
  return t == "ANY" || t == "DATE_AND_TIME" || t.rfind("STRING", 0) == 0;
}

static void parseStructBody(Lexer &lx, Layout &layout,
                            const std::map<std::string, Layout> &udts,
                            std::initializer_list<const char *> endKws,
                            int section = 0, bool fcBlock = false) {
  // continue after members collected by a previous section (VAR_INPUT,
  // VAR_OUTPUT, ... all share one address space)
  int32_t bit = layout.bitCursor;
  while (lx.getCurToken() != Tok::eof) {
    if (lx.getCurToken() == Tok::semi) { lx.next(); continue; }
    if (lx.getCurToken() == Tok::ident) {
      std::string w = upperStr(lx.getIdent());
      bool isEnd = false;
      for (const char *e : endKws)
        if (w == e) isEnd = true;
      if (isEnd) { lx.next(); goto done; }
      if (w.rfind("END_", 0) == 0) { lx.next(); goto done; }
      if (w == "STRUCT") { lx.next(); continue; } // struct header keyword
      if (isHeaderKwLine(w)) { lx.nextLine(); continue; } // TITLE/VERSION/...
      // member?
      Tok save = lx.getCurToken();
      std::string name = w;
      lx.next();
      if (lx.getCurToken() != Tok::colon) {
        // header line etc.: skip to next line
        while (lx.getCurToken() != Tok::semi && lx.getCurToken() != Tok::eof)
          lx.next();
        continue;
      }
      lx.next(); // :
      MemberInfo m;
      std::string t = typeNameOf(lx);
      if (t == "ARRAY" && lx.getCurToken() == Tok::lbracket) {
        lx.next();
        // all dimensions: ARRAY [lo .. hi, lo .. hi, ...] OF elem
        while (lx.getCurToken() != Tok::rbracket && lx.getCurToken() != Tok::eof) {
          int64_t lo = 0, hi = -1;
          if (lx.getCurToken() == Tok::integer) { lo = lx.getInteger(); lx.next(); }
          while (lx.getCurToken() == Tok::dot) lx.next();
          if (lx.getCurToken() == Tok::integer) { hi = lx.getInteger(); lx.next(); }
          m.dims.push_back({(int32_t)lo, (int32_t)hi});
          if (lx.getCurToken() == Tok::comma) { lx.next(); continue; }
          break;
        }
        while (lx.getCurToken() != Tok::rbracket && lx.getCurToken() != Tok::eof)
          lx.next();
        if (lx.getCurToken() == Tok::rbracket) lx.next();
        if (lx.getCurToken() == Tok::ident && upperStr(lx.getIdent()) == "OF")
          lx.next();
        std::string et = typeNameOf(lx);
        if (et.empty())
          et = "INT";
        if (et == "STRING" && lx.getCurToken() == Tok::lbracket) {
          // ARRAY [..] OF STRING[n]: keep the dimension in the elem type
          lx.next();
          if (lx.getCurToken() == Tok::integer) {
            et = "STRING" + std::to_string(lx.getInteger());
            lx.next();
          }
          while (lx.getCurToken() != Tok::rbracket && lx.getCurToken() != Tok::eof)
            lx.next();
          if (lx.getCurToken() == Tok::rbracket)
            lx.next();
        }
        int32_t esize, ealign, ewidth;
        bool ebit;
        typeInfo(et, udts, esize, ealign, ewidth, ebit);
        if (et == "STRUCT") { // ARRAY [..] OF STRUCT <members> END_STRUCT
          auto en = std::make_shared<Layout>();
          parseStructBody(lx, *en, udts, {"END_STRUCT"});
          m.elemNested = en;
          esize = en->size;
          ewidth = en->size * 8;
          ebit = false;
        }
        m.type = "ARRAY";
        m.elemType = et;
        m.count = 1;
        for (auto &d : m.dims)
          m.count *= (int32_t)(d.second - d.first + 1);
        if (m.count < 0) m.count = 0;
        m.lo = m.dims.empty() ? 1 : m.dims[0].first;
        if (ebit) {
          // ARRAY OF BOOL: bit-packed, one bit per element
          m.elemSize = 0;
          m.stride = 0;
          m.widthBits = 1;
          m.isBit = true;
          m.byte = bit / 8;
          m.bit = bit % 8;
          bit += m.count;
        } else {
          m.elemSize = esize ? esize : 2;
          m.stride = m.elemSize;
          m.widthBits = ewidth;
          m.isBit = false;
          bit = (bit + 7) & ~7;              // byte align
          if (m.stride >= 2) bit = (bit + 15) & ~15; // even align
          m.byte = bit / 8;
          bit += m.count * m.stride * 8;
        }
      } else {
        if (t == "STRING" && lx.getCurToken() == Tok::lbracket) {
          lx.next();
          if (lx.getCurToken() == Tok::integer) {
            t = "STRING" + std::to_string(lx.getInteger());
            lx.next();
          }
          while (lx.getCurToken() != Tok::rbracket && lx.getCurToken() != Tok::eof)
            lx.next();
          if (lx.getCurToken() == Tok::rbracket) lx.next();
        }
        if (t == "STRUCT") { // inline anonymous struct member
          auto nested = std::make_shared<Layout>();
          parseStructBody(lx, *nested, udts, {"END_STRUCT"});
          m.type = "STRUCT";
          m.nested = nested;
          int32_t sz = nested->size;
          bit = (bit + 7) & ~7;
          if (sz >= 2) bit = (bit + 15) & ~15;
          m.byte = bit / 8;
          m.bit = 0;
          m.widthBits = sz * 8;
          m.isBit = false;
          bit += sz * 8;
          m.section = (uint8_t)section;
          layout.members[name] = m;
          bool capturedInit = false;
          if (lx.getCurToken() == Tok::colon || lx.getCurToken() == Tok::assign) {
            llvm::StringRef raw = lx.raw();
            size_t pos = raw.find(';');
            std::string slice = pos == llvm::StringRef::npos ? raw.str() : raw.take_front(pos).str();
            size_t i = 0;
            while (i < slice.size() && (slice[i] == ':' || slice[i] == '=' ||
                                        slice[i] == ' ' || slice[i] == '\t'))
              ++i;
            layout.memberInit[name] = slice.substr(i);
            capturedInit = true;
          }
          // skip the initializer text only when one was captured; otherwise
          // the next token may already be the next member (e.g. after
          // END_STRUCT without a trailing ';')
          if (capturedInit)
            while (lx.getCurToken() != Tok::semi && lx.getCurToken() != Tok::eof)
              lx.next();
          continue;
        }
        m.type = t;
        int32_t sz, al, wb;
        bool isb;
        typeInfo(t, udts, sz, al, wb, isb);
        // by-ref compound params carry a pointer header in their slot
        // (6 bytes, 10 for ANY) — reserve it so following members shift
        // (FC params live in the caller's L frame; in FBs only IN_OUT
        // compound params are pointer-overridden in the instance DB)
        if ((isByRefType(t) || t == "ARRAY") && !isb) {
          int32_t hdr = t == "ANY" ? 10 : 6;
          if (fcBlock)
            sz = std::max(sz, hdr); // FC: header slot, declared size if larger
          else if (section == 2)
            sz = hdr; // FB IN_OUT: the field is OVERRIDDEN by a DB-pointer
        }
        m.widthBits = wb;
        m.isBit = isb;
        if (isb) {
          m.byte = bit / 8;
          m.bit = bit % 8;
          ++bit;
        } else {
          bit = (bit + 7) & ~7;                 // leave bit packing, byte align
          if (sz >= 2) bit = (bit + 15) & ~15;  // WORD/DWORD even-aligned
          m.byte = bit / 8;
          m.bit = 0;
          bit += sz * 8;
        }
      }
      m.section = (uint8_t)section;
      layout.members[name] = m;
      // capture the raw initializer text (up to ';'), then skip to ';'
      bool capturedInit = false;
      if (lx.getCurToken() == Tok::colon || lx.getCurToken() == Tok::assign) {
        llvm::StringRef raw = lx.raw();
        size_t pos = raw.find(';');
        std::string slice =
            pos == llvm::StringRef::npos ? raw.str() : raw.take_front(pos).str();
        // strip the leading ":" / ":=" / "=" remnants
        size_t i = 0;
        while (i < slice.size() && (slice[i] == ':' || slice[i] == '=' ||
                                    slice[i] == ' ' || slice[i] == '\t'))
          ++i;
        layout.memberInit[name] = slice.substr(i);
        capturedInit = true;
      }
      if (capturedInit)
        while (lx.getCurToken() != Tok::semi && lx.getCurToken() != Tok::eof)
          lx.next();
      continue;
    }
    lx.next();
    continue;
  }
done:
  int32_t off = (bit + 7) / 8;
  if (off % 2) ++off;
  layout.size = off;
  layout.bitCursor = bit;
}

// MARK: Parser + codegen

class AwlGenImpl {
public:
  AwlGenImpl(mlir::MLIRContext &context, llvm::StringRef source,
             llvm::StringRef filename)
      : builder(&context), lexer(source, filename), file(filename),
        sourceText(source) {}

  llvm::StringRef sourceText;
  // data model collected by a pre-scan over the whole file (declarations may
  // appear after the blocks that reference them)
  std::map<std::string, Layout> udtLayouts;  // TYPE "name" ... END_TYPE
  std::map<int32_t, Layout> dbLayouts;       // DATA_BLOCK DB n (STRUCT or FB ref)
  std::map<int32_t, Layout> fbLayouts;       // FUNCTION_BLOCK FB n interface
  std::map<int32_t, Layout> fcLayouts;       // FUNCTION FC n parameters
  int indirectSeq = 0; // unique label prefix for indirect-call dispatch
  std::map<int32_t, Layout> sfbCache; // SFB n instance layouts
  std::map<int32_t, int32_t> dbFbRef;        // DATA_BLOCK DB n -> "FB n" line
  std::vector<std::string> typeDeclSpans; // TYPE decl text, for fixpoint
  std::map<std::string, Layout> tempLayouts; // VAR_TEMP of OB/FC/FB (L-stack)
  std::map<std::string, Layout> namedLayouts; // FUNCTION(_BLOCK) quoted names
  std::map<std::string, int32_t> namedDbs;  // DATA_BLOCK "Name" -> DB number
  int32_t nextNamedDb = 100;                 // numbers assigned to quoted DBs
  // quoted block name -> {kind, num}, for CALL #var through a quoted-typed
  // multi-instance variable (MULTI : "Function Block 2")
  std::map<std::string, std::pair<std::string, int32_t>> namedBlockNums;
  std::string forcedKind; // set by parseModule for FUNCTION(_BLOCK) quoted
  const Layout *curLayout = nullptr;         // interface of the block being parsed
  const Layout *curTempLayout = nullptr;     // VAR_TEMP of the block being parsed
  int32_t blockVlLitOff = 0;   // running offset of VL literal copies in this block
  int32_t blockVlLitEnd = 0;   // end of the VL literal area (for the lframe size)
  bool curBlockFb = false;
  int32_t curBlockNum = 0;
  // base of the FC parameter area within the block's L frame; must match
  // kParamBase in lib/Runtime/SimRuntime.cpp (scl_rt_param_in/out)
  static constexpr int32_t kLocalBase = 8192;

  /// Pre-scan the file for TYPE / FUNCTION_BLOCK / FUNCTION / DATA_BLOCK
  /// declarations and collect their member layouts (single forward pass,
  /// plus a fixpoint re-parse of TYPE declarations for forward references).
  void preScan() {
    // idempotent re-run: clear everything collected by a previous pass
    udtLayouts.clear();
    dbLayouts.clear();
    fbLayouts.clear();
    fcLayouts.clear();
    tempLayouts.clear();
    namedLayouts.clear();
    namedBlockNums.clear();
    dbFbRef.clear();
    typeDeclSpans.clear();
    gFbLayouts = &fbLayouts;
    // Pre-pass: collect the TYPE declarations and parse them to a fixpoint,
    // so forward UDT references (TYPE UDT 1 uses UDT 2 declared later)
    // resolve before the main pass computes DB/FB layouts.
    {
      Lexer sx(sourceText, file);
      sx.start();
      while (sx.getCurToken() != Tok::eof) {
        if (sx.getCurToken() == Tok::ident && upperStr(sx.getIdent()) == "TYPE") {
          // span from the start of the "TYPE" token to the end of the source
          // (token idents point into the source buffer)
          const char *tokBegin = sx.getIdent().begin();
          typeDeclSpans.push_back(std::string(
              llvm::StringRef(tokBegin, sourceText.end() - tokBegin)));
        }
        sx.next();
      }
    }
    for (int round = 0; round < 4 && !typeDeclSpans.empty(); ++round) {
      std::map<std::string, int32_t> before;
      for (auto &kv : udtLayouts)
        before[kv.first] = kv.second.size;
      for (auto &span : typeDeclSpans) {
        Lexer sub(span, file);
        sub.start();
        if (sub.getCurToken() != Tok::ident || upperStr(sub.getIdent()) != "TYPE")
          continue;
        sub.next();
        std::string name = typeNameOf(sub);
        udtLayouts[name] = Layout();
        Layout &L = udtLayouts[name];
        if (sub.getCurToken() == Tok::ident &&
            upperStr(sub.getIdent()) == "STRUCT")
          sub.next();
        parseStructBody(sub, L, udtLayouts, {"END_TYPE"});
      }
      std::map<std::string, int32_t> after;
      for (auto &kv : udtLayouts)
        after[kv.first] = kv.second.size;
      if (after == before)
        break;
    }
    Lexer lx(sourceText, file);
    lx.start();
    while (lx.getCurToken() != Tok::eof) {
      if (lx.getCurToken() != Tok::ident) {
        lx.next();
        continue;
      }
      std::string w = upperStr(lx.getIdent());
      if (w == "TYPE") {
        lx.next();
        std::string name = typeNameOf(lx);
        udtLayouts[name] = Layout(); // fresh (the pre-pass left one)
        Layout &L = udtLayouts[name];
        if (lx.getCurToken() == Tok::ident &&
            upperStr(lx.getIdent()) == "STRUCT")
          lx.next();
        parseStructBody(lx, L, udtLayouts, {"END_TYPE"});
        continue;
      }
      if (w == "ORGANIZATION_BLOCK") {
        lx.next();
        // ORGANIZATION_BLOCK OB n ... END_ORGANIZATION_BLOCK; the OB's VAR
        // sections are TEMP (L-stack) variables
        if (lx.getCurToken() == Tok::ident)
          lx.next(); // OB
        int32_t num = 0;
        if (lx.getCurToken() == Tok::ident)
          num = (int32_t)blockNumFromName(lx.getIdent());
        else if (lx.getCurToken() == Tok::integer)
          num = (int32_t)lx.getInteger();
        lx.next();
        std::string name = "OB" + std::to_string(num);
        while (lx.getCurToken() != Tok::eof) {
          if (lx.getCurToken() == Tok::ident) {
            std::string s = upperStr(lx.getIdent());
            if (s.rfind("END_", 0) == 0) {
              lx.next();
              break;
            }
            if (s.rfind("VAR", 0) == 0) {
              lx.next();
              parseStructBody(lx, tempLayouts[name], udtLayouts, {"END_VAR"}, 4);
              continue;
            }
            if (s == "BEGIN") {
              lx.next();
              break;
            }
          }
          lx.next();
        }
        // skip the body up to END_ORGANIZATION_BLOCK
        while (lx.getCurToken() != Tok::eof) {
          if (lx.getCurToken() == Tok::ident &&
              upperStr(lx.getIdent()).rfind("END_", 0) == 0) {
            lx.next();
            break;
          }
          lx.next();
        }
        continue;
      }
      if (w == "FUNCTION_BLOCK" || w == "FUNCTION") {
        lx.next();
        std::string kind;
        std::string blockName;
        Layout *Lptr = nullptr;
        if (lx.getCurToken() == Tok::ident && lx.curQuoted()) {
          // FUNCTION "NAME" / FUNCTION_BLOCK "NAME"
          blockName = upperStr(lx.getIdent());
          kind = w == "FUNCTION" ? "FC" : "FB";
          lx.next();
          Lptr = &namedLayouts[blockName];
          namedBlockNums[blockName] = {kind, (int32_t)blockNumFromName(blockName)};
        } else {
          kind = lx.getCurToken() == Tok::ident ? upperStr(lx.getIdent()) : "";
          lx.next();
          int32_t num = 0;
          if (lx.getCurToken() == Tok::ident)
            num = (int32_t)blockNumFromName(lx.getIdent());
          else if (lx.getCurToken() == Tok::integer)
            num = (int32_t)lx.getInteger();
          lx.next();
          blockName = kind + std::to_string(num);
          Lptr = kind == "FC" ? &fcLayouts[num] : &fbLayouts[num];
        }
        // FUNCTION FC n : <retType> — the return value is an implicit
        // RET_VAL output parameter (awlsim adds it after VAR_OUTPUT)
        std::string retType;
        if (kind == "FC" && lx.getCurToken() == Tok::colon) {
          lx.next();
          retType = typeNameOf(lx);
        }
        Layout &L = *Lptr;
        // Interface sections are collected per section and merged in the
        // canonical S7 order VAR_INPUT, VAR_OUTPUT, VAR_IN_OUT, VAR —
        // independent of the order the sections appear in the source
        // (awlsim blockinterface.getAllFields uses the same order).
        Layout secLayouts[5];
        bool secSeen[5] = {false, false, false, false, false};
        while (lx.getCurToken() != Tok::eof) {
          if (lx.getCurToken() == Tok::ident) {
            std::string s = upperStr(lx.getIdent());
            if (s.rfind("END_", 0) == 0) {
              lx.next();
              break;
            }
            if (s.rfind("VAR", 0) == 0) {
              lx.next();
              int sec = s == "VAR_INPUT" ? 0 : s == "VAR_OUTPUT" ? 1
                        : s == "VAR_IN_OUT" ? 2 : s == "VAR_TEMP" ? 4 : 3;
              // TEMP vars live in the block's L-stack frame, not in the
              // interface (param) address space
              Layout &dst = sec == 4 ? tempLayouts[blockName] : secLayouts[sec];
              parseStructBody(lx, dst, udtLayouts, {"END_VAR"}, sec,
                              kind == "FC");
              if (sec != 4)
                secSeen[sec] = true;
              continue;
            }
            if (s == "BEGIN") {
              lx.next();
              break;
            }
          }
          lx.next();
        }
        bool retValAdded = false;
        for (int sec : {0, 1, 2, 3}) {
          if (secSeen[sec]) {
            int32_t baseBit = L.bitCursor;
            for (auto &kv : secLayouts[sec].members) {
              MemberInfo m = kv.second;
              int32_t absBit = baseBit + m.byte * 8 + m.bit;
              if (!m.isBit) {
                // re-align at the section boundary per S7 rules (BOOL stays
                // bit-packed; WORD/DWORD are even-aligned)
                absBit = (absBit + 7) & ~7;
                if (m.widthBits >= 16)
                  absBit = (absBit + 15) & ~15;
              }
              m.byte = absBit / 8;
              m.bit = absBit % 8;
              L.members[kv.first] = m;
              // total member extent in bits (arrays: count * elem stride);
              // by-reference compound params carry a pointer header in
              // their slot (6 bytes, 10 for ANY), so reserve at least that
              int32_t totalBits =
                  m.type == "ARRAY"
                      ? (m.isBit ? m.count : m.count * m.stride * 8)
                      : (m.widthBits ? m.widthBits : 1);
              // FC params are passed via the caller's L frame, so compound
              // by-ref params carry a pointer header in their slot (6 bytes,
              // 10 for ANY). In FBs only IN_OUT compound params are
              // pointer-overridden in the instance DB (awlsim
              // blockinterface __buildField); IN/OUT fields keep their
              // declared size.
              bool hdrSlot = kind == "FC"
                                 ? (isByRefParam(&m) || m.type == "ARRAY")
                                 : (m.section == 2 &&
                                    (isByRefParam(&m) || m.type == "ARRAY"));
              if (hdrSlot) {
                int32_t hdrBits = m.type == "ANY" ? 80 : 48;
                totalBits = kind == "FC" ? std::max(totalBits, hdrBits)
                                         : hdrBits; // FB IN_OUT: override
              }
              int32_t endBit = absBit + totalBits;
              if (endBit > L.bitCursor)
                L.bitCursor = endBit;
            }
            for (auto &kv : secLayouts[sec].memberInit)
              L.memberInit[kv.first] = kv.second;
          }
          // the implicit RET_VAL of an FC goes right after VAR_OUTPUT
          // (awlsim appends it to the OUT fields, even when empty)
          if (sec == 1 && !retValAdded && kind == "FC" && !retType.empty() &&
              retType != "VOID") {
            int32_t sz, al, wb;
            bool isb;
            typeInfo(retType, udtLayouts, sz, al, wb, isb);
            int32_t bit = (L.bitCursor + 7) & ~7;
            if (sz >= 2)
              bit = (bit + 15) & ~15;
            MemberInfo m;
            m.type = retType;
            m.byte = bit / 8;
            m.bit = 0;
            m.widthBits = wb;
            m.section = 1;
            L.members["RET_VAL"] = m;
            L.bitCursor = bit + (sz > 0 ? sz : 1) * 8; // BOOL still takes a byte
            retValAdded = true;
          }
        }
        {
          int32_t off = (L.bitCursor + 7) / 8;
          if (off % 2)
            ++off;
          L.size = off;
        }
        // skip the body up to END_FUNCTION*
        while (lx.getCurToken() != Tok::eof) {
          if (lx.getCurToken() == Tok::ident &&
              upperStr(lx.getIdent()).rfind("END_", 0) == 0) {
            lx.next();
            break;
          }
          lx.next();
        }
        continue;
      }
      if (w == "DATA_BLOCK") {
        lx.next();
        int32_t db = parseDbHeader(lx);
        while (lx.getCurToken() != Tok::eof) {
          if (lx.getCurToken() == Tok::ident) {
            std::string s = upperStr(lx.getIdent());
            if (s == "END_DATA_BLOCK") {
              lx.next();
              break;
            }
            if (s == "STRUCT") {
              lx.next();
              parseStructBody(lx, dbLayouts[db], udtLayouts, {"END_STRUCT"});
              continue;
            }
            if (s == "VAR") {
              lx.next();
              parseStructBody(lx, dbLayouts[db], udtLayouts, {"END_VAR"});
              continue;
            }
            if (s == "FB" || s == "FC" || s == "SFB") {
              lx.next();
              int32_t n = 0;
              if (lx.getCurToken() == Tok::ident)
                n = (int32_t)blockNumFromName(lx.getIdent());
              else if (lx.getCurToken() == Tok::integer)
                n = (int32_t)lx.getInteger();
              lx.next();
              if (s == "FB")
                dbFbRef[db] = n;
              else if (s == "SFB")
                dbLayouts[db] = sfbLayoutOf(n);
              continue;
            }
          }
          lx.next();
        }
        continue;
      }
      lx.next();
    }
    // resolve DATA_BLOCK DB n with `FB m` reference to the FB's layout
    for (auto &kv : dbFbRef) {
      auto it = fbLayouts.find(kv.second);
      if (it != fbLayouts.end())
        dbLayouts[kv.first] = it->second;
    }
  }

  mlir::OwningOpRef<mlir::ModuleOp> run() {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    preScan();  // pass 1: collect all block layouts (FB sizes unknown yet)
    preScan();  // pass 2: recompute member offsets with FB sizes known
    lexer.start();
    if (!parseModule())
      return nullptr;
    // The JIT driver invokes "_mlir_ciface_OB1"; synthesize the wrapper for
    // OB 1 (same convention as the SCL tests).
    if (sawOB1) {
      auto wrapLoc = loc();
      builder.setInsertionPointToEnd(module.getBody());
      auto wrap = builder.create<awl::BlockOp>(
          wrapLoc, builder.getStringAttr("_mlir_ciface_OB1"), mlir::IntegerAttr(),
          mlir::IntegerAttr(), mlir::UnitAttr());
      auto *entry = builder.createBlock(&wrap.getBody());
      builder.setInsertionPointToStart(entry);
      SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), "OB1")));
      builder.create<awl::CallOp>(wrapLoc, mlir::TypeRange{}, mlir::ValueRange{},
                                  attrs);
      builder.create<awl::BeOp>(wrapLoc);
    }
    // Register the referenced extent of every DB used in the file (for
    // ANY-pointer bounds checks on DBs without a DATA_BLOCK declaration);
    // goes into __scl_db_init, which the JIT driver runs before OB 1.
    if (!dbRefBytes.empty()) {
      if (!dbInitBlock) {
        builder.setInsertionPointToEnd(module.getBody());
        dbInitBlock = builder.create<awl::BlockOp>(
            loc(), builder.getStringAttr("__scl_db_init"),
            mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::UnitAttr());
        builder.createBlock(&dbInitBlock.getBody());
      }
      OpBuilder::InsertionGuard guard(builder);
      Block *entry = &dbInitBlock.getBody().front();
      bool hadTerm = !entry->empty() && isa<awl::BeOp>(entry->back());
      if (hadTerm)
        builder.setInsertionPoint(&entry->back());
      else
        builder.setInsertionPointToEnd(entry);
      for (auto &kv : dbRefBytes)
        builder.create<awl::SetDbRefOp>(loc(), builder.getI32IntegerAttr(kv.first),
                                        builder.getI32IntegerAttr(kv.second));
      builder.setInsertionPointToEnd(entry);
      if (entry->empty() || !isa<awl::BeOp>(entry->back()))
        builder.create<awl::BeOp>(loc());
    }
    // Same for the DATA_BLOCK init block, if one was emitted.
    if (dbInitBlock) {
      auto wrapLoc = loc();
      builder.setInsertionPointToEnd(module.getBody());
      auto wrap = builder.create<awl::BlockOp>(
          wrapLoc, builder.getStringAttr("_mlir_ciface___scl_db_init"),
          mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::UnitAttr());
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

  // Accumulator model (per block): ACC stack and the RLO logic state.
  // awlsim models the status word bits VKE/OR/STA/NER; `rlo` is the VKE.
  // The other three are parse-time SSA values too (null = default: OR=0,
  // STA=1, NER=0 — the state after BE / at block start).
  llvm::SmallVector<mlir::Value, 4> acc;
  mlir::Value rlo;        // VKE
  mlir::Value stOr, stSta, stNer; // OR/STA/NER (null = default)
  bool rloActive = false;
  // true while the current statement is a logic-chain op (U/O/A/NOT/X/...):
  // awlsim sets NER=0 on all other instructions, so the next statement's
  // first U/O starts a fresh chain (VKE = operand)
  bool stmtLogic = false;
  struct Bracket {
    mlir::Value savedRlo; // VKE at the bracket open
    mlir::Value savedNer, savedOr;
    bool isOr;
    bool isXor = false;
    bool neg = false;     // UN(/ON(/XN( — negated inner group
    bool isMcr = false;   // MCR( ... )MCR: savedRlo is a pushed mcrStack VKE
  };
  llvm::SmallVector<Bracket, 4> brackets;
  // compile-time approximation of awlsim's runtime mcrActive flag (MCRA/MCRD)
  bool mcrActive = false;
  bool sawOB1 = false;
  std::vector<std::string> obBlockNames; // OB block names, for JIT wrappers
  // awl.block "__scl_db_init" collecting DATA_BLOCK initial values, created
  // lazily; the JIT driver invokes it once before the first scan cycle.
  awl::BlockOp dbInitBlock = nullptr;
  // highest referenced byte (db -> bytes) of each DBn operand in the file
  std::map<int32_t, int32_t> dbRefBytes;
  // auto-assigned symbolic members of undeclared DBs (awlsim symbol tables
  // assign offsets in order of first use; so does the AWL test harness)
  std::map<int32_t, int32_t> autoDbCursor;
  std::map<std::pair<int32_t, std::string>, int32_t> autoDbSyms;
  // resolved type name of the most recent parsed operand (e.g. REAL, so
  // that `L #REALVAR` loads a float, matching the member's data type)
  std::string lastOperandType;
  // width (bits) of the most recent indirect [ARn, P#x.y] operand
  int32_t lastIndirectWidth = 0;
  bool lastOperandIndirect = false; // last parseOperand returned an indirect addr
  int32_t lastPtrCount = 0; // resolved ARRAY element count of a P#DBn.SYM literal
  std::string lastPtrElemType;  // resolved ARRAY element type of a P# literal
  int32_t lastPtrElemBits = 0;  // resolved ARRAY element size in bits
  int32_t ptrParamSlot = -1; // >= 0: parseOperand returned a POINTER-param
                              // VL slot direct at this byte (header store)
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
    bool sawAnyBlock = false;
    while (lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() != Tok::ident) {
        lexer.next(); // stray token between blocks (e.g. after a BEA): skip
        continue;
      }
      std::string w = upperIdent();
      if (w == "ORGANIZATION_BLOCK") {
        // long block form: ORGANIZATION_BLOCK OB 1 ... END_ORGANIZATION_BLOCK
        lexer.next();
        sawAnyBlock = true;
        if (!parseBlockBody(/*longForm=*/true))
          return false;
        continue;
      }
      if (w == "DATA_BLOCK") {
        sawAnyBlock = true;
        if (!parseDataBlock())
          return false;
        continue;
      }
      if (w == "FUNCTION" || w == "FUNCTION_BLOCK") {
        // FB/FC declaration with interface + body: parse as a block so
        // CALL can invoke it (layouts were collected by the pre-scan)
        forcedKind = w == "FUNCTION" ? "FC" : "FB";
        lexer.next();
        sawAnyBlock = true;
        if (!parseBlockBody(/*longForm=*/false))
          return false;
        forcedKind.clear();
        continue;
      }
      if (w == "TYPE") {
        lexer.next();
        while (lexer.getCurToken() != Tok::eof) {
          if (lexer.getCurToken() == Tok::ident && upperIdent() == "END_TYPE") {
            lexer.next();
            break;
          }
          lexer.next();
        }
        continue;
      }
      if (w == "OB" || w == "FB" || w == "FC") {
        sawAnyBlock = true;
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
        sawAnyBlock = true;
        if (!parseBlockBody(/*longForm=*/false, /*implicitOB1=*/true))
          return false;
        continue;
      }
      lexer.next(); // stray token after the blocks: skip
    }
    if (!sawAnyBlock && !sawOB1) {
      // empty bare file: wrap in an implicit empty OB 1 (awlsim does the same)
      if (!parseBlockBody(/*longForm=*/false, /*implicitOB1=*/true))
        return false;
    }
    return true;
  }

  /// Parse a DATA_BLOCK DB n [STRUCT ... END_STRUCT] [BEGIN ... END_DATA_BLOCK].
  /// The STRUCT declaration only sizes the DB (runtime memory, ADR 002); the
  /// BEGIN section holds initial values (`DBn.DBW 2 := 16#00FF`), which are
  /// emitted as stores into an awl.block "__scl_db_init" that the JIT driver
  /// runs once before the first scan.
  // Parse the DATA_BLOCK header token after the DATA_BLOCK keyword: DB n,
  // a bare number, or a quoted name (registered in namedDbs).
  int32_t parseDbHeader(Lexer &lx) {
    int32_t db = 0;
    if (lx.getCurToken() == Tok::ident && upperStr(lx.getIdent()) == "DB") {
      lx.next();
      if (lx.getCurToken() == Tok::integer) {
        db = (int32_t)lx.getInteger();
        lx.next();
      }
    } else if (lx.getCurToken() == Tok::integer) {
      db = (int32_t)lx.getInteger();
      lx.next();
    } else if (lx.getCurToken() == Tok::ident && lx.curQuoted()) {
      std::string name = upperStr(lx.getIdent());
      auto it = namedDbs.find(name);
      if (it == namedDbs.end())
        it = namedDbs.emplace(name, nextNamedDb++).first;
      db = it->second;
      lx.next();
    }
    return db;
  }

  bool parseDataBlock() {
    lexer.next(); // DATA_BLOCK
    int32_t db = parseDbHeader(lexer);
    while (lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() == Tok::ident &&
          upperIdent() == "END_DATA_BLOCK") {
        // declaration-only DB (no BEGIN section): still size it and apply
        // declaration defaults (STRING max-len bytes, member := value)
        return parseDbInitSection(db);
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
          loc(), builder.getStringAttr("__scl_db_init"), mlir::IntegerAttr(),
          mlir::IntegerAttr(), mlir::UnitAttr());
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
    { // configure the DB size from its declaration layout (DBLG/DILG)
      auto lit = dbLayouts.find(db);
      if (lit != dbLayouts.end() && lit->second.size > 0)
        builder.create<awl::SetDbSizeOp>(
            loc(), builder.getI32IntegerAttr(db),
            builder.getI32IntegerAttr(lit->second.size));
      // declaration initializers (STRUCT/VAR member := value, incl. UDT
      // defaults) first
      if (lit != dbLayouts.end()) {
        std::vector<std::string> assigns;
        collectInitAssigns(lit->second, "", udtLayouts, assigns);
        for (const std::string &assign : assigns) {
          Lexer sub(assign, "<dbinit>"); // assign must outlive sub
          sub.start();
          Lexer savedLexer = lexer;
          lexer = sub;
          parseDbInitLine();
          lexer = savedLexer;
        }
        emitStringMaxlenInits(lit->second, db, 0);
      }
    }
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

  // "[i, j]" index text for flat element k of an array member
  static std::string arrayIndexText(const MemberInfo &mi, int64_t k) {
    std::string idx;
    int64_t rem = k;
    for (size_t d = 0; d < mi.dims.size(); ++d) {
      int64_t radix = 1;
      for (size_t j = d + 1; j < mi.dims.size(); ++j)
        radix *= mi.dims[j].second - mi.dims[j].first + 1;
      int64_t i = mi.dims.empty() ? 0 : rem / radix;
      if (!mi.dims.empty())
        rem %= radix;
      idx += (d ? ", " : "[") +
             std::to_string(mi.dims.empty() ? 0
                                            : mi.dims[d].first + (int32_t)i);
    }
    idx += "]";
    return idx;
  }

  // Collect all declaration initializers of a layout as "PREFIX.NAME[i].sub
  // := value;" texts, recursing into UDT / inline-STRUCT members (per array
  // element for arrays).
  // STRING[n] members: initialize the max-len byte (byte 0 of the string)
  // to the declared length; the actual-length byte stays 0 (awlsim
  // initializes STRING declarations this way at DB load).
  void emitStringMaxlenInits(const Layout &lay, int32_t db,
                             int32_t baseByte) {
    auto maxlenOf = [](const std::string &ty) {
      return ty.size() > 6 ? atoi(ty.c_str() + 6) : 254;
    };
    for (auto &kv : lay.members) {
      const MemberInfo *mi = &kv.second;
      if (mi->type.rfind("STRING", 0) == 0) {
        auto addr = mkDirect(mlir::scl::DirectArea::DataBlock, db,
                             baseByte + mi->byte, 0, 8);
        builder.create<awl::StoreOp>(
            loc(), constInt(builder.getI32Type(), maxlenOf(mi->type)), addr);
        continue;
      }
      if (mi->type == "ARRAY" && mi->elemType.rfind("STRING", 0) == 0) {
        int32_t maxlen = maxlenOf(mi->elemType);
        for (int32_t k = 0; k < mi->count; ++k) {
          auto addr = mkDirect(mlir::scl::DirectArea::DataBlock, db,
                               baseByte + mi->byte + k * mi->stride, 0, 8);
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), maxlen), addr);
        }
        continue;
      }
      const Layout *sub = nullptr;
      if (mi->type == "ARRAY") {
        if (mi->elemNested)
          sub = mi->elemNested.get();
        else {
          auto u = udtLayouts.find(mi->elemType);
          if (u != udtLayouts.end())
            sub = &u->second;
        }
        if (sub)
          for (int32_t k = 0; k < mi->count; ++k)
            emitStringMaxlenInits(*sub, db, baseByte + mi->byte + k * mi->stride);
        continue;
      }
      if (mi->nested)
        sub = mi->nested.get();
      else {
        auto u = udtLayouts.find(mi->type);
        if (u != udtLayouts.end())
          sub = &u->second;
      }
      if (sub)
        emitStringMaxlenInits(*sub, db, baseByte + mi->byte);
    }
  }

  static void collectInitAssigns(const Layout &lay, const std::string &prefix,
                                 const std::map<std::string, Layout> &udts,
                                 std::vector<std::string> &out) {
    for (auto &kv : lay.members) {
      const MemberInfo *mi = &kv.second;
      std::string base = prefix.empty() ? kv.first : prefix + "." + kv.first;
      auto initIt = lay.memberInit.find(kv.first);
      if (initIt != lay.memberInit.end()) {
        for (auto &a : expandMemberInit(kv.first, initIt->second, mi)) {
          std::string suffix = a.substr(kv.first.size()); // "[i] := v"/" := v"
          out.push_back(base + suffix);
        }
      }
      const Layout *sub = nullptr;
      if (mi->type == "ARRAY") {
        if (mi->elemNested)
          sub = mi->elemNested.get();
        else {
          auto u = udts.find(mi->elemType);
          if (u != udts.end())
            sub = &u->second;
        }
        if (sub && !sub->memberInit.empty()) {
          for (int32_t k = 0; k < mi->count; ++k)
            collectInitAssigns(*sub, base + arrayIndexText(*mi, k), udts, out);
          continue;
        }
      } else {
        if (mi->nested)
          sub = mi->nested.get();
        else {
          auto u = udts.find(mi->type);
          if (u != udts.end())
            sub = &u->second;
        }
      }
      if (sub && !sub->memberInit.empty())
        collectInitAssigns(*sub, base, udts, out);
    }
  }

  // Expand a declaration initializer into one "NAME[i,...] := value;" text
  // per element. Handles "v, v, ..." and the "N (v, ...)" repeat form.
  // Array indices are emitted with the first dimension most significant
  // (row-major), matching consumeArrayIndices.
  static std::vector<std::string> expandMemberInit(const std::string &name,
                                                   const std::string &slice,
                                                   const MemberInfo *mi) {
    std::vector<std::string> out;
    // strip outer whitespace
    std::string s = slice;
    auto isws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    size_t b = 0, e = s.size();
    while (b < e && isws(s[b])) ++b;
    while (e > b && isws(s[e - 1])) --e;
    s = s.substr(b, e - b);
    int repeat = 1;
    std::string list = s;
    // "N (v, v, ...)" repeat form
    {
      size_t i = 0;
      int64_t n = 0;
      bool digits = false;
      while (i < s.size() && isdigit((unsigned char)s[i])) {
        n = n * 10 + (s[i++] - '0');
        digits = true;
      }
      size_t j = i;
      while (j < s.size() && isws(s[j])) ++j;
      if (digits && j < s.size() && s[j] == '(' && s.back() == ')') {
        repeat = (int)n;
        list = s.substr(j + 1, s.size() - j - 2);
      }
    }
    // split the value list on top-level commas
    std::vector<std::string> vals;
    {
      int depth = 0;
      size_t start = 0;
      for (size_t i = 0; i <= list.size(); ++i) {
        if (i == list.size() || (list[i] == ',' && depth == 0)) {
          std::string v = list.substr(start, i - start);
          size_t vb = 0, ve = v.size();
          while (vb < ve && isws(v[vb])) ++vb;
          while (ve > vb && isws(v[ve - 1])) --ve;
          if (ve > vb)
            vals.push_back(v.substr(vb, ve - vb));
          start = i + 1;
        } else if (list[i] == '(') {
          ++depth;
        } else if (list[i] == ')') {
          --depth;
        }
      }
    }
    if (vals.empty())
      vals.push_back("0");
    if (!mi || mi->dims.empty()) {
      out.push_back(name + " := " + vals[0] + ";");
      return out;
    }
    // per-dimension sizes
    std::vector<int32_t> n(mi->dims.size());
    int32_t total = 1;
    for (size_t d = 0; d < mi->dims.size(); ++d) {
      n[d] = mi->dims[d].second - mi->dims[d].first + 1;
      if (n[d] < 1) n[d] = 1;
      total *= n[d];
    }
    int64_t count = (int64_t)repeat * (int64_t)vals.size();
    if (count > total)
      count = total;
    for (int64_t k = 0; k < count; ++k) {
      std::string idx;
      int64_t rem = k;
      for (size_t d = 0; d < mi->dims.size(); ++d) {
        int64_t radix = 1;
        for (size_t j = d + 1; j < mi->dims.size(); ++j)
          radix *= n[j];
        int64_t i = rem / radix;
        rem %= radix;
        idx += (d ? ", " : "[") + std::to_string(mi->dims[d].first + (int32_t)i);
      }
      idx += "]";
      out.push_back(name + idx + " := " + vals[k % vals.size()] + ";");
    }
    return out;
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
      // a known member of this DB's layout always wins over the operand
      // shape (T1_2 is a member name, not a T-area operand)
      auto lit0 = dbLayouts.find(effDb(dbOverride));
      if (lit0 != dbLayouts.end() && lit0->second.members.count(w0))
        looksOperand = false;
      if (!looksOperand)
        symName = w0; // symbolic member name (e.g. TIMER_VAR := T 24)
    }
    mlir::Value addr;
    if (symName.empty()) {
      addr = parseOperand(false);
    } else {
      lexer.next(); // member name
      int32_t byte = 0, bit = 0, width = 16;
      auto lit = dbLayouts.find(effDb(dbOverride));
      if (lit != dbLayouts.end()) {
        auto it = lit->second.members.find(symName);
        if (it != lit->second.members.end()) {
          byte = it->second.byte;
          bit = it->second.bit;
          width = it->second.widthBits ? it->second.widthBits : 16;
          if (lexer.getCurToken() == Tok::lbracket)
            addBits(byte, bit, consumeArrayIndices(&it->second));
          // .sub-member chains (UDT / inline STRUCT / array elements),
          // arbitrarily deep: VAR10.VAR10_3[1].VAR10_3_2[2]
          const MemberInfo *cur = &it->second;
          while (lexer.getCurToken() == Tok::dot) {
            lexer.next();
            if (lexer.getCurToken() != Tok::ident &&
                lexer.getCurToken() != Tok::localvar)
              break;
            std::string sub = upperStr(lexer.getIdent());
            lexer.next();
            const Layout *next = nullptr;
            if (cur->nested)
              next = cur->nested.get();
            else if (cur->type == "ARRAY" && cur->elemNested)
              next = cur->elemNested.get();
            else if (cur->type != "ARRAY") {
              auto uit = udtLayouts.find(cur->type);
              if (uit != udtLayouts.end())
                next = &uit->second;
            } else {
              auto uit = udtLayouts.find(cur->elemType);
              if (uit != udtLayouts.end())
                next = &uit->second;
            }
            if (!next)
              break;
            auto it2 = next->members.find(sub);
            if (it2 == next->members.end())
              break;
            byte += it2->second.byte;
            bit = it2->second.bit;
            width = it2->second.widthBits ? it2->second.widthBits : 16;
            cur = &it2->second;
            if (lexer.getCurToken() == Tok::lbracket)
              addBits(byte, bit, consumeArrayIndices(cur));
          }
        }
      }
      addr = mkDirect(mlir::scl::DirectArea::DataBlock, effDb(dbOverride), byte,
                      bit, width);
      // remember the member's type for the RHS handler (e.g. STRING literal;
      // for ARRAY OF STRING the element type is what matters)
      auto lit2 = dbLayouts.find(effDb(dbOverride));
      if (lit2 != dbLayouts.end()) {
        auto it2 = lit2->second.members.find(symName);
        if (it2 != lit2->second.members.end()) {
          lastOperandType = it2->second.type == "ARRAY"
                               ? it2->second.elemType
                               : it2->second.type;
        }
      }
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
      if (lexer.curQuoted() && lexer.curQuoteChar() == '\'' &&
          lastOperandType.rfind("STRING", 0) == 0 &&
          mlir::isa<mlir::scl::DirectType>(addr.getType())) {
        // VAR := 'literal': store current length at byte 1 and the chars
        // from byte 2 (byte 0 = max length, set at declaration init)
        auto dt = mlir::cast<mlir::scl::DirectType>(addr.getType());
        std::string raw = lexer.getIdent().str();
        lexer.next();
        auto putB = [&](int32_t off, int32_t v) {
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), v),
              mkDirect(dt.getArea(), dt.getDb(), dt.getByte() + off, 0, 8));
        };
        putB(1, (int32_t)raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
          putB(2 + (int32_t)i, (unsigned char)raw[i]);
        // awlsim zero-fills the rest of the string up to the declared maxlen
        int32_t maxlen = lastOperandType.size() > 6
                             ? atoi(lastOperandType.c_str() + 6)
                             : 254;
        for (int32_t i = (int32_t)raw.size(); i < maxlen; ++i)
          putB(2 + i, 0);
        return true;
      }
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
    } else if (isPseudoValue(addr)) {
      // store to a pseudo operand: value dropped
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
    std::string quotedName; // FUNCTION "NAME" / FUNCTION_BLOCK "NAME"
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
      bool quotedKind = lexer.curQuoted();
      kind = upperIdent();
      lexer.next();
      if (quotedKind) {
        // FUNCTION "NAME" : VOID / FUNCTION_BLOCK "NAME" — the quoted token
        // is the whole block name (called via CALL "NAME")
        quotedName = kind;
        kind = forcedKind.empty() ? "FC" : forcedKind;
        num = 0;
      } else if (lexer.getCurToken() == Tok::ident) {
        num = blockNumFromName(lexer.getIdent());
      } else if (lexer.getCurToken() == Tok::integer) {
        num = lexer.getInteger();
      } else {
        return error("expected block number after " + kind);
      }
      auto skipReturnType = [&]() {
        // FC return type annotation: `FC 1 : VOID`, `FC 65535 : UDT 65535`
        if (lexer.getCurToken() == Tok::colon) {
          lexer.next();
          while (lexer.getCurToken() == Tok::ident ||
                 lexer.getCurToken() == Tok::integer) {
            std::string rw = lexer.getCurToken() == Tok::ident
                                 ? upperStr(lexer.getIdent())
                                 : "";
            if (!rw.empty() && (rw.rfind("VAR", 0) == 0 || rw == "BEGIN" ||
                                rw.rfind("END_", 0) == 0 ||
                                isHeaderKw(rw)))
              break;
            lexer.next();
          }
        }
      };
      if (!quotedKind) {
        lexer.next();
        skipReturnType();
      } else {
        skipReturnType();
      }
    }
    std::string name = !quotedName.empty() ? quotedName
                                           : kind + std::to_string(num);
    if (kind == "OB" && num == 1)
      sawOB1 = true;
    if (kind == "OB" && quotedName.empty())
      obBlockNames.push_back(name); // for the JIT startup-OB wrappers
    curBlockNum = (int32_t)num;
    curBlockFb = kind == "FB";
    blockVlLitOff = 0;
    blockVlLitEnd = 0;
    if (!quotedName.empty()) {
      auto it = namedLayouts.find(quotedName);
      curLayout = it != namedLayouts.end() ? &it->second : nullptr;
    } else if (kind == "FB") {
      auto it = fbLayouts.find((int32_t)num);
      curLayout = it != fbLayouts.end() ? &it->second : nullptr;
    } else if (kind == "FC") {
      auto it = fcLayouts.find((int32_t)num);
      curLayout = it != fcLayouts.end() ? &it->second : nullptr;
    } else {
      curLayout = nullptr;
    }
    {
      auto it = tempLayouts.find(name);
      curTempLayout = it != tempLayouts.end() ? &it->second : nullptr;
    }

    auto blockLoc = loc();
    builder.setInsertionPointToEnd(module.getBody());
    auto blockOp = builder.create<awl::BlockOp>(
        blockLoc, builder.getStringAttr(name), mlir::IntegerAttr(),
        mlir::IntegerAttr(),
        kind == "FB" ? builder.getUnitAttr() : mlir::UnitAttr());
    // FC interface parameters live in the block's own L frame at
    // kLocalBase (frame-relative), so recursive calls are isolated; the
    // frame must cover both the TEMP vars and the parameter area
    int32_t tempSize = curTempLayout ? curTempLayout->size : 0;
    int32_t paramBytes = (kind == "FC" && curLayout) ? curLayout->size : 0;
    if (paramBytes > 0)
      blockOp->setAttr("paramBytes", builder.getI32IntegerAttr(paramBytes));
    int32_t frame = tempSize;
    if (paramBytes > 0 && kLocalBase + paramBytes > frame)
      frame = kLocalBase + paramBytes;
    if (frame > 0)
      blockOp->setAttr("lframe", builder.getI32IntegerAttr(frame));
    auto *entry = builder.createBlock(&blockOp.getBody());
    builder.setInsertionPointToStart(entry);
    reloadAcc(); // ACCU1..4 persist from the runtime (awlsim cpu.accu1..4)
    rlo = nullptr;
    stOr = stSta = stNer = nullptr; // defaults: OR=0, STA=1, NER=0
    rloActive = false;
    brackets.clear();

    // header lines between the block header and BEGIN (TITLE, VERSION, ...)
    while (lexer.getCurToken() == Tok::ident &&
           isHeaderKw(upperStr(lexer.getIdent())))
      lexer.nextLine();
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
      // header lines inside a block body (TITLE = ..., VERSION : ...)
      if (lexer.getCurToken() == Tok::ident &&
          isHeaderKw(upperStr(lexer.getIdent()))) {
        lexer.nextLine();
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
        // A label starts a new statement (ADR 003): RLO does not cross the
        // label boundary, so the next U/O/NOT starts from the neutral RLO
        // instead of an SSA value from a predecessor block (which would not
        // dominate the label's block after the lowering splits at labels).
        // The accumulators DO persist (awlsim keeps them in runtime
        // registers): reload the parse-time stack from the runtime ACCUs.
        reloadAcc();
        rlo = nullptr;
        stOr = stSta = stNer = nullptr;
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
      stmtLogic = false;
      if (!parseInstruction(name))
        return false;
      if (!stmtLogic)
        rloActive = false; // non-logic statement ends the logic chain
      if (!blockEnded)
        syncAcc(); // keep the runtime ACCU registers current for label reloads
      // note: awlsim does NOT reset NER/OR on non-logic instructions (only
      // BE/CLR/SET/brackets do), so stNer/stOr persist across statements
      // the STW logic bits mirror the parse-time state at the end of each
      // logic statement (awlsim: VKE/STA/NER/OR are runtime status bits)
      if (rloActive && rlo) {
        builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(1),
                                          rlo);              // VKE
        builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(0),
                                          nerD());            // NER
        builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(2),
                                          staD());            // STA
        builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(3),
                                          orD());             // OR
      }
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
    // VL literal copies for by-ref params live in this block's L frame past
    // the parameter area; grow the frame to cover them
    if (blockVlLitEnd > 0) {
      int32_t frame = curTempLayout ? curTempLayout->size : 0;
      int32_t paramBytes = (kind == "FC" && curLayout) ? curLayout->size : 0;
      if (kLocalBase + paramBytes > frame)
        frame = kLocalBase + paramBytes;
      if (blockVlLitEnd > frame)
        frame = blockVlLitEnd;
      blockOp->setAttr("lframe", builder.getI32IntegerAttr(frame));
    }
    blockEnded = false;
    return true;
  }

  bool blockEnded = false;

  int insnLine = 0; // line of the current instruction keyword

  bool parseInstruction(const std::string &blockName) {
    auto t = lexer.getCurToken();
    if (t == Tok::semi) { // statement separator
      lexer.next();
      return true;
    }
    if (t == Tok::ident) {
      std::string m = upperIdent();
      insnLine = lexer.curLine();
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
      // `+ <immediate>` (awlsim insn_pl, TYPE_PL): plain int literal -> word
      // add with setWord, L#.. literal -> dword add with setDWord, real ->
      // real op. Unlike +I/+D, the STW is NOT updated.
      if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real || lexer.getCurToken() == Tok::minus) {
        Lexer ahead = lexer;
        if (ahead.getCurToken() == Tok::minus)
          ahead.next();
        bool isReal = lexer.getCurToken() == Tok::real;
        bool isD = !isReal && ahead.getLitWidth() == 32;
        mlir::Value b = parseConst();
        if (!b)
          return false;
        padAcc();
        mlir::Value a = popAcc();
        if (!a)
          return false;
        auto i16 = builder.getI16Type();
        auto i32 = builder.getI32Type();
        mlir::Value r;
        if (isReal) {
          auto f32 = builder.getF32Type();
          mlir::Value as = castTo(a, f32), bs = castTo(b, f32);
          switch (k) {
          case ArithKind::Add: r = builder.create<awl::AddOp>(loc(), f32, as, bs); break;
          case ArithKind::Sub: r = builder.create<awl::SubOp>(loc(), f32, as, bs); break;
          case ArithKind::Mul: r = builder.create<awl::MulOp>(loc(), f32, as, bs); break;
          case ArithKind::Div: r = builder.create<awl::DivOp>(loc(), f32, as, bs); break;
          }
        } else {
          mlir::Value as = isD ? castTo(a, i32) : castTo(castTo(a, i16), i32);
          mlir::Value bs = isD ? castTo(b, i32) : castTo(castTo(b, i16), i32);
          mlir::Value r32;
          switch (k) {
          case ArithKind::Add: r32 = builder.create<awl::AddOp>(loc(), i32, as, bs); break;
          case ArithKind::Sub: r32 = builder.create<awl::SubOp>(loc(), i32, as, bs); break;
          case ArithKind::Mul: r32 = builder.create<awl::MulOp>(loc(), i32, as, bs); break;
          case ArithKind::Div: r32 = builder.create<awl::DivOp>(loc(), i32, as, bs); break;
          }
          r = isD ? r32
                  : builder.create<awl::OrOp>(
                        loc(), i32,
                        builder.create<awl::AndOp>(loc(), i32, a,
                            constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
                        builder.create<awl::AndOp>(loc(), i32, r32,
                            constInt(i32, 0xFFFF)).getResult()).getResult();
        }
        pushAcc(r);
        rloActive = false;
        return true;
      }
      if (lexer.getCurToken() != Tok::ident)
        return error("expected I/D/R after arithmetic operator");
      std::string w = upperIdent();
      lexer.next();
      if (w.rfind("AR", 0) == 0 && w.size() == 3 && isdigit(w[2])) {
        // + AR1 P#x.y: address register arithmetic — ARn += offset (bits)
        int ar = w[2] - '0';
        mlir::Value off;
        if (lexer.getCurToken() == Tok::integer ||
            lexer.getCurToken() == Tok::real) {
          off = constInt(builder.getI64Type(),
                         consumePtrLiteral() & 0xFFFFFF); // byte*8+bit part
        } else {
          // bare + ARn: add ACCU1's low 24 bits
          off = builder.create<awl::AndOp>(
              loc(), builder.getI64Type(), zextTo64(popAcc()),
              constInt(builder.getI64Type(), 0xFFFFFF));
        }
        auto np = builder.create<awl::ArAddOp>(
            loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar), off);
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(ar), np);
        rloActive = false;
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
    // SPEC[AWL] IEC 61131-3 STL block end (BE/BEA/BEU), Siemens AWL manual "Block end"
    if (m == "BE" || m == "BEA" || m == "BEU") {
      builder.create<awl::BeOp>(loc());
      blockEnded = true;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL conditional block end (BEB/BEC), Siemens AWL manual
    if (m == "BEB" || m == "BEC") {
      // BEB: end block if RLO = 1; BEC: end block if RLO = 0
      if (!ensureRlo())
        return false;
      auto op = builder.create<awl::BeIfOp>(loc(), rlo, mlir::UnitAttr());
      if (m == "BEC")
        op->setAttr("negate", builder.getUnitAttr());
      rloActive = false;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL no-operation (NOP)
    if (m == "NOP") {
      if (lexer.getCurToken() == Tok::integer)
        lexer.next(); // NOP 0 / NOP 1
      builder.create<awl::NopOp>(loc());
      rloActive = false;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL load (L), Siemens AWL manual "Load and transfer" — see parseLoad
    if (m == "L")
      return parseLoad();
    // SPEC[AWL] IEC 61131-3 STL transfer (T), Siemens AWL manual "Load and transfer" — see parseStore
    if (m == "T")
      return parseStore();
    // SPEC[AWL] IEC 61131-3 STL boolean logic ops (U/UN/A/AN/O/ON) + STW comparisons (U ==0 etc.), Siemens AWL manual "Logic operations"
    if ((m == "U" || m == "UN" || m == "A" || m == "AN" || m == "O" ||
         m == "ON") &&
        lexer.getCurToken() != Tok::lparen) {
      stmtLogic = true;
      // bare U/O/... (no operand, e.g. at end of a chain): RLO unchanged.
      // An ident followed by an address token is an operand (U A 0.0,
      // UN L 0.0); only if no address follows and the ident is an
      // instruction keyword is the logic op bare (O / L STW).
      if (lexer.getCurToken() == Tok::semi || lexer.getCurToken() == Tok::rparen)
        return bareLogicEnd(m); // bare `O ;`
      if (lexer.getCurToken() == Tok::eq || lexer.getCurToken() == Tok::ne ||
          lexer.getCurToken() == Tok::lt || lexer.getCurToken() == Tok::le ||
          lexer.getCurToken() == Tok::gt || lexer.getCurToken() == Tok::ge) {
        // `U ==0` / `U <>0` / `U >0` / `U <0` / `U >=0` / `U <=0`: STW
        // condition from the A0/A1 bits (0=Z,1=NZ,2=POS,3=NEG,4=GE,5=LE)
        Tok ct = lexer.getCurToken();
        lexer.next();
        if (lexer.getCurToken() == Tok::integer)
          lexer.next();
        int cond = ct == Tok::eq ? 0
                   : ct == Tok::ne ? 1
                   : ct == Tok::gt ? 2
                   : ct == Tok::lt ? 3
                   : ct == Tok::ge ? 4
                                   : 5;
        rlo = builder.create<awl::StwCondOp>(loc(), builder.getI1Type(),
                                              builder.getI32IntegerAttr(cond));
        rloActive = true;
        return true;
      }
      if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted() &&
          lexer.curLine() != insnLine)
        return bareLogicEnd(m); // next line starts a new instruction: bare op
      if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperStr(lexer.getIdent()); // not consumed yet
        Tok nxt = lexer.peekToken();
        bool addrFollows = nxt == Tok::integer || nxt == Tok::real ||
                           nxt == Tok::dot || nxt == Tok::lbracket;
        if (!addrFollows &&
            (w == "L" || w == "T" || w == "U" || w == "UN" || w == "O" ||
             w == "ON" || w == "A" || w == "AN" || w == "NOT" || w == "SET" ||
             w == "CLR" || w == "BE" || w == "NOP" || w == "CALL" ||
             w == "SPA" || w == "SPB" || w == "SPN")) {
          lexer.next(); // consume the keyword: bare logic op
          return bareLogicEnd(m);
        }
      }
      return parseRloLogic(m == "AN" ? "UN" : m);
    }
    // SPEC[AWL] IEC 61131-3 STL NOT (negate RLO)
    if (m == "NOT") {
      stmtLogic = true;
      // NOT inverts the *current* RLO, even across non-logic statements
      // (awlsim insn_not: STA=1, VKE ^= 1, NER unchanged)
      if (!ensureRlo())
        return false;
      rlo = builder.create<awl::NotOp>(loc(), builder.getI1Type(), vkeD());
      stSta = i1Const(true);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL SET (RLO := 1)
    if (m == "SET") { // awlsim insn_set: OR=0, STA=1, VKE=1, NER=0
      stmtLogic = true;
      rlo = i1Const(true);
      stOr = i1Const(false);
      stSta = i1Const(true);
      stNer = i1Const(false);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL CLR (RLO := 0)
    if (m == "CLR") { // awlsim insn_clr: OR=STA=VKE=NER=0
      stmtLogic = true;
      rlo = i1Const(false);
      stOr = i1Const(false);
      stSta = i1Const(false);
      stNer = i1Const(false);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Opening a data block" (AUF/OPN DB/DI, incl. AUF DB [LW n])
    if (m == "AUF") { // AUF DB n / AUF DI n: open data block -> set the
      // current DB/DI register (DB 0 operands then address it, ABI v5)
      bool di = false;
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "DB" || w == "DI") {
          di = w == "DI";
          lexer.next();
        } else if ((w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0) &&
                   w.size() > 2) {
          // attached number: AUF DB1 / OPN DI12
          int32_t n = atoi(w.c_str() + 2);
          if (w[0] == 'D' && w[1] == 'I')
            builder.create<awl::SetDiOp>(loc(), builder.getI32IntegerAttr(n));
          else
            builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr(n));
          lexer.next();
          return true;
        }
      }
      if (lexer.getCurToken() == Tok::integer) {
        int64_t db = lexer.getInteger();
        lexer.next();
        if (di)
          builder.create<awl::SetDiOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
        else
          builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
      } else if (lexer.getCurToken() == Tok::lbracket ||
                 lexer.getCurToken() == Tok::localvar) {
        // AUF DB [LW 0] / AUF #INPUT_DB
        if (!parseSetDbValue(di))
          return false;
      }
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Address register instructions" (LAR1/LAR2/+ARn; LAR loads the operand content as pointer, awlsim insn_larN)
    if (m.rfind("LAR", 0) == 0 || m.rfind("+AR", 0) == 0) {
      // LAR1/LAR2 <operand>: load a pointer into address register 1/2.
      // P#x.y literal -> raw byte*8+bit value; a direct operand (DB1.DBW 4,
      // M 10.0) -> packed pointer; no operand -> ACCU1 (as TARn).
      int ar = (m.size() >= 4 && isdigit(m[3])) ? m[3] - '0'
               : (m == "+AR2")                  ? 2
                                                : 1;
      mlir::Value ptr;
      int64_t offVal = -1; // +ARn with a P#x.y literal offset
      if (lexer.getCurToken() == Tok::localvar && lexer.curPtrLocal()) {
        ptr = parsePtrLocal(); // LAR1 P##name
      } else if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real) {
        int64_t off = consumePtrLiteral(); // P#x.y (sym resolved)
        ptr = constInt(builder.getI64Type(), off);
        offVal = off & 0xFFFFFF; // +ARn adds only the byte*8+bit part
      } else if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        if (w == "AR1" || w == "AR2") { // LARn ARm: copy the other AR
          lexer.next();
          ptr = builder.create<awl::ArGetOp>(
              loc(), builder.getI64Type(),
              builder.getI32IntegerAttr(w[2] - '0'));
        } else {
        bool looksOperand =
            w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0 ||
            (isAreaLetter(w[0]) &&
             (w.size() == 1 || isdigit((unsigned char)w[1]) ||
              w[1] == 'B' || w[1] == 'W' || w[1] == 'D'));
        if (!looksOperand || lexer.curLine() != insnLine)
          ptr = nullptr; // bare LARn (operand on a later line): take ACCU1
        else {
          // LARn <memory operand>: ARn := the operand's 32-bit CONTENT (a
          // pointer value, awlsim insn_larN: ar.set(fetch(op0, 32bit)))
          auto loadPtrValue = [&](mlir::Value addr) -> mlir::Value {
            if (!mlir::isa<mlir::scl::DirectType>(addr.getType()))
              return addr;
            auto dt = mlir::cast<mlir::scl::DirectType>(addr.getType());
            auto a32 = builder.create<awl::DirectOp>(
                loc(), dt.getArea(), dt.getDb(), dt.getByte(), dt.getBit(), 32);
            return zextTo64(builder.create<awl::LoadOp>(loc(), a32));
          };
          if (w.size() <= 2 && w[0] == 'L' &&
              lexer.peekToken() == Tok::integer) { // LD/LW/LB operand
            lexer.next();
            int32_t byte = 0;
            if (lexer.getCurToken() == Tok::integer) {
              byte = (int32_t)lexer.getInteger();
              lexer.next();
            }
            ptr = loadPtrValue(builder.create<awl::DirectOp>(
                loc(), mlir::scl::DirectArea::Local, 0, byte, 0, 32));
          } else {
            ptr = parseOperand(false); // DB1.DBW 4, M 10.0, ...
            if (!ptr)
              return true;
            ptr = loadPtrValue(ptr);
          }
        }
        }
      }
      if (!ptr)
        ptr = zextTo64(peekAcc()); // bare LARn = TARn (ACCU1 stays)
      if (m.rfind("+AR", 0) == 0) {
        // +ARn: ARn += offset in bits. A bare +ARn adds ACCU1's low 24 bits
        // (the byte*8+bit part of the pointer in ACCU1).
        mlir::Value off =
            offVal >= 0 ? constInt(builder.getI64Type(), offVal)
                        : builder.create<awl::AndOp>(
                              loc(), builder.getI64Type(), ptr,
                              constInt(builder.getI64Type(), 0xFFFFFF));
        auto np = builder.create<awl::ArAddOp>(
            loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar), off);
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(ar), np);
      } else {
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(ar), ptr);
      }
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Address register instructions" (TAR1/TAR2/TAR; TAR stores the AR as a raw S7 pointer, awlsim insn_tarN)
    if (m.rfind("TAR", 0) == 0) {
      // TAR1/TAR2 [operand]: AR := pointer operand, or ACCU1 if none
      int ar = (m.size() >= 4 && isdigit(m[3])) ? m[3] - '0' : 1;
      if (m == "TAR") {
        // bare TAR: swap AR1 and AR2
        auto a1 = builder.create<awl::ArGetOp>(loc(), builder.getI64Type(),
                                               builder.getI32IntegerAttr(1));
        auto a2 = builder.create<awl::ArGetOp>(loc(), builder.getI64Type(),
                                               builder.getI32IntegerAttr(2));
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(1), a2);
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(2), a1);
        return true;
      }
      mlir::Value ptr;
      bool arOperand = false;
      int operandAr = 0;
      if (lexer.getCurToken() == Tok::localvar && lexer.curPtrLocal()) {
        ptr = parsePtrLocal(); // TAR1 P##name
      } else if (lexer.getCurToken() == Tok::integer ||
                 lexer.getCurToken() == Tok::real) {
        int64_t off = consumePtrLiteral(); // P#x.y (sym resolved)
        ptr = constInt(builder.getI64Type(), off);
      } else if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        if (w == "AR1" || w == "AR2") { // TARn ARm: ARm := ARn
          lexer.next();
          arOperand = true;
          operandAr = w[2] - '0';
          ptr = builder.create<awl::ArGetOp>(
              loc(), builder.getI64Type(), builder.getI32IntegerAttr(operandAr));
        } else if ((w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0 ||
            (isAreaLetter(w[0]) &&
             (w.size() == 1 || isdigit((unsigned char)w[1]) ||
              w[1] == 'B' || w[1] == 'W' || w[1] == 'D'))) && lexer.curLine() == insnLine) {
          if (w.size() <= 2 && w[0] == 'L' &&
              lexer.peekToken() == Tok::integer) { // LD/LW/LB operand
            lexer.next();
            int32_t byte = 0;
            if (lexer.getCurToken() == Tok::integer) {
              byte = (int32_t)lexer.getInteger();
              lexer.next();
            }
            int32_t lwd = w == "LD" ? 32 : w == "LB" ? 8 : 16;
            ptr = builder.create<awl::DirectOp>(
                loc(), mlir::scl::DirectArea::Local, 0, byte, 0, lwd);
          } else {
            ptr = parseOperand(false);
            if (!ptr)
              return true;
          }
        }
      }
      if (arOperand) {
        // TARn ARm: ARm := ARn (awlsim insn_tarN stores arN into the operand)
        mlir::Value arv = builder.create<awl::ArGetOp>(
            loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar));
        builder.create<awl::ArSetOp>(loc(), builder.getI32IntegerAttr(operandAr),
                                     arv);
        return true;
      }
      mlir::Value arv = builder.create<awl::ArGetOp>(
          loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar));
      if (!ptr) {
        // bare TARn: ACCU1 := ARn, ACCU2 := old ACCU1, ... (awlsim
        // insn_tarN: accu2.copyFrom(accu1); accu1.copyFrom(arN))
        if (acc.empty())
          pushAcc(constInt(builder.getI32Type(), 0));
        else
          acc.erase(acc.begin()); // ACCU4 drops off (4-accu model)
        pushAcc(castTo(arv, builder.getI32Type()));
        return true;
      }
      // TARn <operand> stores ARn INTO the operand (awlsim insn_tarN:
      // store(oper, arN)) — the opposite direction of LARn
      if (mlir::isa<mlir::scl::DirectType>(ptr.getType())) {
        // TARn <memory> stores the AR as a raw 32-bit S7 pointer value
        // (awlsim stores arN as a pointer), not the packed internal form
        mlir::Value raw = builder.create<awl::ArRawOp>(
            loc(), builder.getI32Type(), builder.getI32IntegerAttr(ar));
        builder.create<awl::StoreOp>(loc(), raw, ptr);
      } else {
        // P#x.y literal / P##name / ARm: store through the packed pointer
        builder.create<awl::StoreIndirectOp>(
            loc(), arv, ptr, builder.getI32IntegerAttr(32));
      }
      return true;
    }
    if (lexer.getCurToken() == Tok::lparen &&
        (m == "A" || m == "O" || m == "U" || m == "UN" || m == "ON")) {
      // awlsim insn_ub/ob/...: push (VKE, NER, OR); set OR=0, STA=1, NER=0
      // (VKE itself is unchanged — the inner chain starts neutral via NER=0)
      stmtLogic = true;
      lexer.next();
      if (!ensureRlo())
        return false;
      Bracket b;
      b.savedRlo = vkeD();
      b.savedNer = nerD();
      b.savedOr = orD();
      b.isOr = (m == "O" || m == "ON");
      b.neg = (m == "UN" || m == "ON");
      brackets.push_back(b);
      stOr = i1Const(false);
      stSta = i1Const(true);
      stNer = i1Const(false);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL bit set/reset (S/R), Siemens AWL manual "Bit logic"
    if (m == "S")
      return parseSetReset(true);
    if (m == "R")
      return parseSetReset(false);
    // SPEC[AWL] IEC 61131-3 STL word/dword logic (AW/OW/XOW/AD/OD/XOD), Siemens AWL manual "Word logic"
    if (m == "AW" || m == "OW" || m == "XOW" || m == "AD" || m == "OD" ||
        m == "XOD" || m == "UD" || m == "UW" || m == "UB" || m == "OB")
      return parseWordLogic(m);
    // SPEC[AWL] Siemens AWL manual "Timer instructions" (SP/SE/SS/SV/SI/SD, S5TIME), runtime scl_rt_timer_*
    if (m == "SP" || m == "SE" || m == "SS" || m == "SV" || m == "SI" ||
        m == "SA" || m == "SD") // SD = English off-delay (German SA)
      return parseTimerStart(m == "SD" ? "SA" : m);
    // SPEC[AWL] Siemens AWL manual "Counter instructions" (ZV/ZR/CU/CD), runtime scl_rt_counter_op
    if (m == "ZV" || m == "ZR" || m == "CU" || m == "CD")
      return parseCounterZW(m == "ZV" || m == "CU");
    // SPEC[AWL] IEC 61131-3 STL unconditional/conditional jumps (SPA/SPB/SPN/SPBN/JU/JB/JNB/JBI/JC/LOOP), Siemens AWL manual "Jumps"
    if (m == "SPA" || m == "SPB" || m == "SPN" || m == "SPBN" || m == "JU")
      return parseJump(m == "JU" ? "SPA" : m);
    // SPEC[AWL] Siemens AWL manual "SAVE" (RLO -> BR status bit)
    if (m == "SAVE") { // SAVE: RLO -> BR (STW bit 8)
      if (!ensureRlo())
        return false;
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(8), rlo);
      if (lexer.getCurToken() == Tok::semi)
        lexer.next();
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual S5-compat SF (english alias of SA, off-delay timer start — awlsim english2german "SF" -> "SA")
    if (m == "SF")
      return parseTimerStart("SA");
    // SPEC[AWL] IEC 61131-3 STL TAK (exchange accumulators), Siemens AWL manual "Accumulator instructions"
    if (m == "TAK") { // exchange ACCU1/ACCU2
      padAcc();
      std::swap(acc[acc.size() - 1], acc[acc.size() - 2]);
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL NEGI/NEGD (two's-complement ACCU1); awlsim
    // insn_negi/insn_negd: negate the signed word/dword and store with
    // setWord/setDWord; STW A0/A1 from the truncated result, OV=0 (NEGD:
    // OV=OS=1 only for the 0x80000000 overflow)
    if (m == "NEGI" || m == "NEGD") {
      padAcc();
      mlir::Value a = popAcc();
      if (!a)
        return false;
      auto i16 = builder.getI16Type();
      auto i32 = builder.getI32Type();
      auto i1 = builder.getI1Type();
      mlir::Value r, trunc;
      mlir::Value ovf; // awlsim: OV=OS=1 when the untruncated value leaves the range
      if (m == "NEGI") {
        mlir::Value sw = castTo(castTo(a, i16), i32); // sign-extended word
        mlir::Value val = builder.create<awl::NegOp>(loc(), i32, sw).getResult();
        trunc = builder.create<awl::AndOp>(loc(), i32, val,
                                           constInt(i32, 0xFFFF)).getResult();
        r = builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::AndOp>(loc(), i32, a,
                                       constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
            trunc).getResult();
        ovf = builder.create<awl::OrOp>(
            loc(), i1,
            builder.create<awl::CmpGtOp>(loc(), i1, val, constInt(i32, 0x7FFF)).getResult(),
            builder.create<awl::CmpLtOp>(loc(), i1, val, constInt(i32, -32768))).getResult();
      } else {
        mlir::Value val = builder.create<awl::NegOp>(loc(), i32, castTo(a, i32)).getResult();
        r = val;
        trunc = val;
        ovf = builder.create<awl::CmpEqOp>(
            loc(), i1, castTo(a, i32), constInt(i32, (int64_t)(int32_t)0x80000000));
      }
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(5), ovf); // OV
      mlir::Value os = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(4));
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(4),
                                       builder.create<awl::OrOp>(loc(), i1, os, ovf).getResult()); // OS latches
      pushAcc(r);
      mlir::Value nz = builder.create<awl::CmpNeOp>(loc(), i1, trunc, constInt(i32, 0));
      mlir::Value sign = builder.create<awl::CmpNeOp>(
          loc(), i1,
          builder.create<awl::AndOp>(loc(), i32, trunc,
              constInt(i32, m == "NEGI" ? 0x8000 : (int64_t)(int32_t)0x80000000)).getResult(),
          constInt(i32, 0));
      builder.create<awl::StwSetBitOp>(
          loc(), builder.getI32IntegerAttr(7), // A1
          builder.create<awl::AndOp>(
              loc(), i1, nz,
              builder.create<awl::NotOp>(loc(), i1, sign).getResult()).getResult());
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(6), sign); // A0
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Shift and rotate" (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA), awl.shift
    if (m == "SLW" || m == "SRW" || m == "SSI" || m == "SSD" ||
        m == "SLD" || m == "SRD" || m == "RLD" || m == "RRD" ||
        m == "RLDA" || m == "RRDA")
      return parseShift(m);
    // SPEC[AWL] Siemens AWL manual "Real math" (ABS/NEGR/SIN/COS/TAN/...), awl.real_math (approx: float32 host math)
    if (m == "ABS" || m == "NEGR" || m == "SIN" || m == "COS" || m == "TAN" ||
        m == "ASIN" || m == "ACOS" || m == "ATAN" || m == "SQRT" ||
        m == "EXP" || m == "LN" || m == "SQR")
      return parseRealMath(m);
    // SPEC[AWL] Siemens AWL manual "Value conversions" TRUNC/RND (approx: truncation)
    if (m == "TRUNC" || m == "RND") {
      // real ACCU1 -> int; TRUNC truncates, RND rounds (RND+ up, RND- down).
      // Out of int32 range the ACCU is unchanged and OV/OS are set (runtime).
      int mode = 0; // TRUNC
      if (m == "RND") {
        mode = 1;
        if (lexer.getCurToken() == Tok::plus) {
          mode = 2;
          lexer.next();
        } else if (lexer.getCurToken() == Tok::minus) {
          mode = 3;
          lexer.next();
        }
      }
      mlir::Value a = acc.empty() ? constInt(builder.getF32Type(), 0) : popAcc();
      // out-of-range leaves ACCU1 unchanged (awlsim: the raw dword), so the
      // fallback value is the BIT pattern, not a float->int value cast
      mlir::Value orig = mlir::isa<mlir::FloatType>(a.getType())
                             ? castTo(bitcastTo(a, builder.getI32Type()),
                                      builder.getI64Type())
                             : castTo(a, builder.getI64Type());
      mlir::Value f = mlir::isa<mlir::IntegerType>(a.getType())
                          ? bitcastTo(castTo(a, builder.getI32Type()),
                                      builder.getF32Type())
                          : castTo(a, builder.getF32Type());
      pushAcc(builder.create<awl::F2IOp>(
          loc(), builder.getI32Type(), f, builder.getI32IntegerAttr(mode),
          orig));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Value conversions" DTR (dword->real), plus BTI/BTD/ITB/DTB BCD and ITD/DTI/INVI/INVD below
    if (m == "DTR") { // DWORD ACCU1 -> real (value conversion, awlsim insn_dtr)
      mlir::Value a = popAcc();
      pushAcc(castTo(castTo(a, builder.getI32Type()), builder.getF32Type()));
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Enabling" FR (timer/counter), runtime scl_rt_timer_fr
    if (m == "FR") { // FR T n / FR C n: enable — reset the edge memory of
      // the timer/counter at runtime; the RLO is NOT modified
      mlir::Value v = parseOperand(false);
      if (!v)
        return false;
      // Direct (T n / C n) or a dynamic i64 T/C pointer (#timer_var)
      builder.create<awl::FrOp>(loc(), v, rloInput());
      return true;
    }
    // SPEC[AWL] Siemens AWL manual TDB (swap DB/DI registers) (approx: not modelled)
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
    // SPEC[AWL] Siemens AWL manual "Opening a data block" OPN (= AUF)
    if (m == "OPN") { // OPN DB n / OPN DI n / OPN DB1: open data block (like AUF)
      bool di = false;
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "DB" || w == "DI") {
          di = w == "DI";
          lexer.next();
        } else if ((w.rfind("DB", 0) == 0 || w.rfind("DI", 0) == 0) &&
                   w.size() > 2) {
          int32_t n = atoi(w.c_str() + 2);
          if (w[0] == 'D' && w[1] == 'I')
            builder.create<awl::SetDiOp>(loc(), builder.getI32IntegerAttr(n));
          else
            builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr(n));
          lexer.next();
          return true;
        }
      }
      if (lexer.getCurToken() == Tok::integer) {
        int64_t db = lexer.getInteger();
        lexer.next();
        if (di)
          builder.create<awl::SetDiOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
        else
          builder.create<awl::SetDbOp>(loc(), builder.getI32IntegerAttr((int32_t)db));
      } else if (lexer.getCurToken() == Tok::lbracket ||
                 lexer.getCurToken() == Tok::localvar) {
        if (!parseSetDbValue(di))
          return false;
      }
      return true;
    }
    // SPEC[AWL] awlsim test directive __STWRST (reset status word)
    if (m == "__STWRST") { // awlsim statusword.reset(): ALL bits 0,
      // including NER/STA/OR (also in the parse-time RLO state)
      builder.create<awl::StwResetOp>(loc());
      auto i1 = builder.getI1Type();
      rlo = builder.create<awl::ConstOp>(loc(), i1, builder.getIntegerAttr(i1, 0));
      stOr = nullptr;
      stSta = nullptr;
      stNer = nullptr;
      rloActive = true;
      return true;
    }
    // SPEC[AWL] awlsim test directive __FEATURE t [, v] (nrAccus / OB-temp presets), runtime scl_rt_feature
    if (m == "__FEATURE") { // awlsim test directive: __FEATURE t [, v]
      int64_t target = 0, value = -1;
      if (lexer.getCurToken() == Tok::integer) {
        target = lexer.getInteger();
        lexer.next();
      }
      if (lexer.getCurToken() == Tok::comma) {
        lexer.next();
        if (lexer.getCurToken() == Tok::integer) {
          value = lexer.getInteger();
          lexer.next();
        }
      }
      // awlsim insn_feature: accu1 := feature value (nrAccus for target 0)
      pushAcc(builder.create<awl::FeatureOp>(
          loc(), builder.getI32Type(),
          builder.getI32IntegerAttr((int32_t)target),
          builder.getI32IntegerAttr((int32_t)value)));
      return true;
    }
    // SPEC[AWL] awlsim test directive __SLEEP ms (advance virtual clock), runtime scl_rt_sleep
    if (m == "__SLEEP") { // awlsim test directive: advance the virtual clock
      int64_t ms = 0;
      if (lexer.getCurToken() == Tok::integer) {
        ms = lexer.getInteger();
        lexer.next();
      }
      builder.create<awl::SleepOp>(loc(), builder.getI32IntegerAttr((int32_t)ms));
      return true;
    }
    // SPEC[AWL] Siemens AWL manual LC (load timer/counter current value, BCD-coded)
    if (m == "LC") {
      // LC T n: timer current value in S5TIME format (awlsim getTimevalS5T);
      // LC Z n and others load like L.
      if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted() &&
          upperIdent() == "T" && lexer.curLine() == insnLine) {
        std::string w = upperIdent();
        lexer.next();
        int32_t n = 0;
        if (lexer.getCurToken() == Tok::integer) {
          n = (int32_t)lexer.getInteger();
          lexer.next();
        }
        pushAcc(builder.create<awl::TimerLcOp>(
            loc(), builder.getI32Type(), builder.getI32IntegerAttr(n)));
        return true;
      }
      return parseLoad(); // LC Z n: load counter value into ACCU1
    }
    // SPEC[AWL] Siemens AWL manual FP/FN edge memory — the operand bit itself is the edge flag (awlsim insn_FP/FN): store flag := RLO, then new RLO = RLO & !oldFlag (FP) / !RLO & oldFlag (FN)
    if (m == "FP" || m == "FN") {
      stmtLogic = true;
      if (!ensureRlo())
        return false;
      mlir::Value addr = parseBitOperand();
      if (!addr)
        return false;
      auto i1 = builder.getI1Type();
      mlir::Value old = addr.getType().isInteger(64)
          ? (mlir::Value)builder.create<awl::LoadBitIndirectOp>(loc(), i1, addr)
          : (mlir::Value)builder.create<awl::LoadOp>(loc(), addr);
      if (addr.getType().isInteger(64))
        builder.create<awl::WriteBitIndirectOp>(loc(), rlo, addr);
      else
        builder.create<awl::AssignOp>(loc(), rlo, addr);
      rlo = m == "FP"
          ? (mlir::Value)builder.create<awl::AndOp>(
                loc(), i1, rlo,
                builder.create<awl::NotOp>(loc(), i1, old).getResult())
          : (mlir::Value)builder.create<awl::AndOp>(
                loc(), i1,
                builder.create<awl::NotOp>(loc(), i1, rlo).getResult(), old);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual MCR (master control relay) — MCRA/MCRD set the (compile-time tracked) mcrActive flag; SAVE (RLO -> BR) stays a no-op
    if (m == "MCRA") {
      mcrActive = true;
      return true;
    }
    if (m == "MCRD") {
      mcrActive = false;
      return true;
    }
    if (m == "SAVE")
      return true; // BR bit is not modelled
    // SPEC[AWL] Siemens AWL manual BLD <n> (display code — no-op)
    if (m == "BLD") { // BLD <n>: display code, no-op
      if (lexer.getCurToken() == Tok::integer)
        lexer.next();
      return true;
    }
    // SPEC[AWL] Siemens AWL manual MCR( / MCR ) nesting — the pushed VKE (savedRlo of the isMcr bracket) gates stores via mcrOn()/gatedRlo()
    if (m == "MCR") { // MCR( ... MCR): master control relay — bracket, store gate
      if (lexer.getCurToken() == Tok::lparen) {
        // awlsim insn_mcrb: push VKE onto the mcrStack; OR=0, STA=1, NER=0
        lexer.next();
        if (!ensureRlo())
          return false;
        Bracket b;
        b.savedRlo = vkeD(); // pushed mcrStack VKE (gates stores via mcrOn)
        b.savedNer = nerD();
        b.savedOr = orD();
        b.isOr = false;
        b.isMcr = true;
        brackets.push_back(b);
        stOr = i1Const(false);
        stSta = i1Const(true);
        stNer = i1Const(false);
        rloActive = true;
        return true;
      }
      if (lexer.getCurToken() == Tok::rparen) {
        lexer.next();
        return parseCloseBracket();
      }
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL SPBB/SPB-branch save (save RLO into BR)
    if (m == "SPBB")
      return parseJump(m);
    // SPEC[AWL] Siemens AWL manual ENT/PUSH (push accumulators), POP/LEAVE below
    if (m == "ENT") { // awlsim insn_ent: ACCU4:=ACCU3, ACCU3:=ACCU2
      entAcc();
      return true;
    }
    if (m == "PUSH") { // awlsim insn_push: ACCU2 := ACCU1 (only)
      padAcc();
      acc[acc.size() - 2] = acc.back();
      return true;
    }
    // SPEC[AWL] Siemens AWL manual POP/LEAVE (pop accumulators)
    if (m == "LEAVE") {
      // awlsim insn_leave: ACCU2 := ACCU3, ACCU3 := ACCU4 (ACCU1 unchanged)
      leaveAcc();
      return true;
    }
    if (m == "POP") { // awlsim insn_pop: ACCU1 := ACCU2 (only)
      padAcc();
      acc.back() = acc[acc.size() - 2];
      return true;
    }
    // SPEC[AWL] Siemens AWL manual INC/DEC (ACCU1 +/- n)
    // SPEC[AWL] Siemens AWL manual INC/DEC n — byte op on ACCU1 (awlsim setByte)
    if (m == "INC" || m == "DEC") {
      int64_t n = 1;
      if (lexer.getCurToken() == Tok::integer) {
        n = lexer.getInteger();
        lexer.next();
      }
      auto i32 = builder.getI32Type();
      mlir::Value a = acc.empty() ? constInt(i32, 0) : castTo(popAcc(), i32);
      mlir::Value sum = builder.create<awl::AddOp>(
          loc(), i32, a, constInt(i32, m == "INC" ? n : -n)).getResult();
      // setByte: low byte of the sum, rest of ACCU1 preserved
      pushAcc(builder.create<awl::OrOp>(
          loc(), i32,
          builder.create<awl::AndOp>(loc(), i32, a,
                                     constInt(i32, (int64_t)(int32_t)0xFFFFFF00)).getResult(),
          builder.create<awl::AndOp>(loc(), i32, sum,
                                     constInt(i32, 0xFF)).getResult()).getResult());
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Value conversions" ITD/DTI — ITD sign-extends the low word of ACCU1 to a dword (awlsim setDWord(getSignedWord))
    if (m == "ITD" || m == "DTI") {
      mlir::Value a = acc.empty() ? constInt(builder.getI32Type(), 0) : popAcc();
      if (m == "ITD")
        pushAcc(castTo(castTo(a, builder.getI16Type()), builder.getI32Type()));
      else
        pushAcc(castTo(a, builder.getI16Type()));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual INVI/INVD (one's complement) — INVI inverts the low word keeping the high word (awlsim setWord), INVD inverts the full dword
    if (m == "INVI" || m == "INVD") {
      auto i32 = builder.getI32Type();
      mlir::Value a = acc.empty() ? constInt(i32, 0) : castTo(popAcc(), i32);
      if (m == "INVD") {
        pushAcc(builder.create<awl::XOrOp>(loc(), i32, a, constInt(i32, -1)));
      } else {
        pushAcc(builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::AndOp>(loc(), i32, a,
                                       constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
            builder.create<awl::XOrOp>(
                loc(), i32,
                builder.create<awl::AndOp>(loc(), i32, a, constInt(i32, 0xFFFF)).getResult(),
                constInt(i32, 0xFFFF)).getResult()).getResult());
      }
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual MOD (remainder)
    if (m == "MOD") { // awlsim insn_mod: ACCU1 = dword remainder (sign of
      // the dividend); division by zero leaves ACCU1 unchanged and sets
      // A0=A1=OV=OS=1
      padAcc();
      mlir::Value b = popAcc();
      mlir::Value a = popAcc();
      if (!a || !b)
        return false;
      auto i32 = builder.getI32Type();
      auto i64 = builder.getI64Type();
      auto i1 = builder.getI1Type();
      mlir::Value as = castTo(a, i32), bs = castTo(b, i32);
      mlir::Value as64 = castTo(as, i64), bs64 = castTo(bs, i64);
      mlir::Value quo = builder.create<awl::DivOp>(loc(), i64, as64, bs64);
      mlir::Value rem = builder.create<awl::SubOp>(
          loc(), i64, as64,
          builder.create<awl::MulOp>(loc(), i64, quo, bs64).getResult()).getResult();
      mlir::Value isZero = builder.create<awl::CmpEqOp>(loc(), i1, bs, constInt(i32, 0));
      pushAcc(mux(isZero, castTo(b, i32), castTo(rem, i32)));
      auto div = builder.create<awl::StwDivOp>(loc(), bs, rem);
      div.setRangeAttr(builder.getI32IntegerAttr(0x7FFFFFFF));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual TAD (dword accumulator exchange)
    // SPEC[AWL] Siemens AWL manual TAD (swapEndianDWord(ACCU1): reverse all four bytes — awlsim insn_TAD)
    if (m == "TAD") {
      padAcc();
      auto i32 = builder.getI32Type();
      mlir::Value v = castTo(popAcc(), i32);
      auto shl = [&](mlir::Value x, int64_t n) {
        return builder.create<awl::ShiftOp>(loc(), i32,
            builder.getStringAttr("sld"), x, constInt(i32, n)).getResult();
      };
      auto shr = [&](mlir::Value x, int64_t n) {
        return builder.create<awl::ShiftOp>(loc(), i32,
            builder.getStringAttr("srd"), x, constInt(i32, n)).getResult();
      };
      auto band = [&](mlir::Value x, int64_t mask) {
        return builder.create<awl::AndOp>(loc(), i32, x, constInt(i32, mask)).getResult();
      };
      mlir::Value b0 = band(v, 0xFF), b1 = band(shr(v, 8), 0xFF),
                  b2 = band(shr(v, 16), 0xFF), b3 = band(shr(v, 24), 0xFF);
      pushAcc(builder.create<awl::OrOp>(
          loc(), i32,
          builder.create<awl::OrOp>(loc(), i32, shl(b0, 24), shl(b1, 16)).getResult(),
          builder.create<awl::OrOp>(loc(), i32, shl(b2, 8), b3).getResult()).getResult());
      return true;
    }
    if ((m == "X" || m == "XN") && lexer.getCurToken() == Tok::lparen) {
      lexer.next();
      if (!ensureRlo())
        return false;
      Bracket b;
      b.savedRlo = vkeD();
      b.savedNer = nerD();
      b.savedOr = orD();
      b.isOr = false;
      b.isXor = true;
      b.neg = (m == "XN");
      brackets.push_back(b);
      stOr = i1Const(false);
      stSta = i1Const(true);
      stNer = i1Const(false);
      rloActive = true;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL X/XN (XOR RLO with bit)
    if (m == "X" || m == "XN") { // X/XN bit: awlsim insn_X: VKE = (VKE & NER) ^ x (XN: ^ !x)
      stmtLogic = true;
      if (!ensureRlo())
        return false;
      auto i1 = builder.getI1Type();
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w == "TRUE" || w == "FALSE") {
          lexer.next();
          rloCombine(false, true, i1Const(w == "TRUE"), m == "XN");
          return true;
        }
      }
      mlir::Value addr = parseBitOperand();
      if (!addr)
        return false;
      mlir::Value b;
      if (addr.getType().isInteger(64))
        // X [AR1, P#x.y] / X #timer_var: indirect bit through the packed ptr
        b = builder.create<awl::LoadBitIndirectOp>(loc(), i1, addr);
      else if (addr.getType().isInteger(1))
        b = addr;
      else
        b = builder.create<awl::LoadOp>(loc(), addr);
      rloCombine(false, true, castTo(b, i1), m == "XN");
      return true;
    }
    // SPEC[AWL] Siemens AWL manual SPL (jump to label via ACCU1)
    if (m == "SPL") {
      // SPL is a jump table: `SPL <default>` followed by a list of
      // `SPA <label>` entries; the low byte of ACCU1 selects the entry
      // (awlsim: lookup = accu1.getByte() + 1, default if out of range)
      if (lexer.getCurToken() != Tok::ident)
        return error("expected label after SPL");
      std::string def = upperIdent();
      lexer.next();
      std::vector<std::string> table;
      {
        Lexer ahead = lexer;
        while (ahead.getCurToken() == Tok::ident &&
               upperStr(ahead.getIdent()) == "SPA") {
          ahead.next();
          if (ahead.getCurToken() != Tok::ident)
            break;
          table.push_back(upperStr(ahead.getIdent()));
          ahead.next();
        }
      }
      mlir::Value accVal =
          acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back();
      mlir::Value accB = zextAcc(castTo(accVal, builder.getI8Type()));
      auto i32 = builder.getI32Type();
      for (size_t i = 0; i < table.size(); ++i) {
        mlir::Value r = builder.create<awl::CmpEqOp>(
            loc(), builder.getI1Type(), accB, constInt(i32, (int64_t)i));
        builder.create<awl::JumpIfOp>(loc(), r, builder.getStringAttr(table[i]));
      }
      builder.create<awl::JumpOp>(loc(), builder.getStringAttr(def));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL conditional jumps JNB/JBI/SPBNB/SPBI/SPBIN (with BR save)
    if (m == "SPBNB" || m == "SPBI" || m == "SPBIN" || m == "JNB" ||
        m == "JNBI" || m == "SPO" || m == "SOS" || m == "SPU" || m == "JCN")
      return parseJump(m);
    // SPEC[AWL] IEC 61131-3 STL conditional jumps JB/JBI/JC (RLO/BR conditions)
    if (m == "JB" || m == "JBI" || m == "JC")
      return parseJump(m);
    // SPEC[AWL] IEC 61131-3 STL LOOP (decrement ACCU1, jump if != 0)
    if (m == "LOOP") { // awlsim insn_loop: ACCU1 word -= 1 (setWord),
      // jump if the result is != 0
      if (lexer.getCurToken() != Tok::ident)
        return error("expected label after LOOP");
      std::string label = upperIdent();
      lexer.next();
      padAcc();
      auto i32 = builder.getI32Type();
      // Read ACCU1 from the runtime at the LOOP site: across a back-edge the
      // parse-time acc still holds the SSA values loaded at the label, which
      // are stale after the first iteration.
      mlir::Value a = builder.create<awl::AccuGetOp>(
          loc(), i32, builder.getI32IntegerAttr(1)).getResult();
      auto i16 = builder.getI16Type();
      mlir::Value dec = builder.create<awl::SubOp>(
          loc(), i32, castTo(castTo(a, i16), i32), constInt(i32, 1)).getResult();
      mlir::Value r = builder.create<awl::OrOp>(
          loc(), i32,
          builder.create<awl::AndOp>(loc(), i32, a,
                                     constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
          builder.create<awl::AndOp>(loc(), i32, dec, constInt(i32, 0xFFFF)).getResult()).getResult();
      acc.back() = r; // fall-through sees the decremented ACCU1
      syncAcc();      // back-edge: the label reloads from the runtime accus
      auto c = builder.create<awl::CmpNeOp>(
          loc(), builder.getI1Type(),
          builder.create<awl::AndOp>(loc(), i32, r, constInt(i32, 0xFFFF)).getResult(),
          constInt(i32, 0));
      builder.create<awl::JumpIfOp>(loc(), c, builder.getStringAttr(label));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] IEC 61131-3 STL zero/plus/minus jumps (SPZ/SPPZ/SPMZ/SPS/SPP/SPM)
    // awlsim insn_spz/insn_spp/...: these test the STW A0/A1 bits, NOT ACCU1
    if (m == "SPZ" || m == "SPPZ" || m == "SPMZ" || m == "SPS" || m == "SPP" ||
        m == "SPM") {
      if (lexer.getCurToken() != Tok::ident)
        return error("expected label after " + m);
      std::string label = upperIdent();
      lexer.next();
      auto i1 = builder.getI1Type();
      mlir::Value c;
      int cond; // StwCondOp: 0=Z(!A0&!A1) 2=POS(!A0&A1) 3=NEG(A0&!A1) 4=GE(!A0) 5=LE(!A1)
      if (m == "SPZ")
        cond = 0;
      else if (m == "SPP")
        cond = 2;
      else if (m == "SPM")
        cond = 3;
      else if (m == "SPPZ")
        cond = 4;
      else if (m == "SPMZ")
        cond = 5;
      else // SPS: jump if OS, and clear OS (awlsim insn_sps)
        cond = -1;
      if (cond >= 0)
        c = builder.create<awl::StwCondOp>(loc(), i1, builder.getI32IntegerAttr(cond));
      else
        c = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(4)); // OS
      syncAcc();
      if (m == "SPS")
        builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(4),
                                         i1Const(false)); // OS := 0 (before the jump)
      builder.create<awl::JumpIfOp>(loc(), c, builder.getStringAttr(label));
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Value conversions" BTI/BTD (BCD -> binary);
    // awlsim insn_bti/insn_btd: 3/7 BCD digits in the low nibbles plus a sign
    // nibble (bit15/bit31); BTI stores with setWord, BTD with setDWord
    if (m == "BTI" || m == "BTD") {
      padAcc();
      mlir::Value a = popAcc();
      if (!a)
        return false;
      auto i32 = builder.getI32Type();
      bool word = m == "BTI";
      mlir::Value digits = bcdToInt(
          builder.create<awl::AndOp>(loc(), i32, a,
              constInt(i32, word ? 0xFFF : 0x0FFFFFFF)).getResult(),
          i32, word ? 3 : 7);
      mlir::Value s = builder.create<awl::AndOp>(loc(), i32,
          builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("srd"),
                                       a, constInt(i32, word ? 15 : 31)).getResult(),
          constInt(i32, 1)).getResult();
      mlir::Value negmask = builder.create<awl::SubOp>(loc(), i32, constInt(i32, 0), s).getResult();
      mlir::Value bin = builder.create<awl::SubOp>(
          loc(), i32,
          builder.create<awl::XOrOp>(loc(), i32, digits, negmask).getResult(),
          negmask).getResult(); // digits or -digits
      if (word)
        pushAcc(builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::AndOp>(loc(), i32, a,
                                       constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
            builder.create<awl::AndOp>(loc(), i32, bin, constInt(i32, 0xFFFF)).getResult()).getResult());
      else
        pushAcc(bin);
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual "Value conversions" ITB/DTB (binary -> BCD);
    // awlsim insn_itb/insn_dtb: sign nibble 0xF000/0xF0000000, |v| > 999/9999999
    // sets OV=OS=1 and leaves ACCU1 unchanged, else OV=0; A0/A1 untouched
    if (m == "ITB" || m == "DTB") {
      padAcc();
      mlir::Value a = popAcc();
      if (!a)
        return false;
      auto i16 = builder.getI16Type();
      auto i32 = builder.getI32Type();
      auto i1 = builder.getI1Type();
      bool word = m == "ITB";
      mlir::Value sv = word ? castTo(castTo(a, i16), i32) : castTo(a, i32);
      mlir::Value negmask = builder.create<awl::ShiftOp>(
          loc(), i32, builder.getStringAttr("ssd"), sv, constInt(i32, 31)).getResult(); // 0 / -1 (arithmetic)
      mlir::Value absv = builder.create<awl::SubOp>(
          loc(), i32,
          builder.create<awl::XOrOp>(loc(), i32, sv, negmask).getResult(),
          negmask).getResult();
      mlir::Value over = builder.create<awl::CmpGtOp>(
          loc(), i1, absv, constInt(i32, word ? 999 : 9999999));
      mlir::Value bcd = intToBcd(absv, i32, word ? 3 : 7);
      mlir::Value w = builder.create<awl::OrOp>(
          loc(), i32, bcd,
          builder.create<awl::AndOp>(loc(), i32, negmask,
                                     constInt(i32, word ? 0xF000 : (int64_t)(int32_t)0xF0000000)).getResult()).getResult();
      mlir::Value r = word
          ? builder.create<awl::OrOp>(loc(), i32,
                builder.create<awl::AndOp>(loc(), i32, a,
                                           constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
                builder.create<awl::AndOp>(loc(), i32, w, constInt(i32, 0xFFFF)).getResult()).getResult()
          : w;
      pushAcc(mux(over, a, r)); // out of range: ACCU1 unchanged
      // OV = over (else 0), OS = OS | over
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(5), over);
      mlir::Value os = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(4));
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(4),
                                       builder.create<awl::OrOp>(loc(), i1, os, over).getResult());
      rloActive = false;
      return true;
    }
    // SPEC[AWL] Siemens AWL manual TAW (swap the two bytes of ACCU1's low word — awlsim insn_TAW: accu1 = (accu1 & 0xFFFF0000) | swapEndianWord(accu1 & 0xFFFF))
    if (m == "TAW") {
      padAcc();
      mlir::Value v = popAcc();
      auto i32 = builder.getI32Type();
      v = castTo(v, i32);
      mlir::Value lo = builder.create<awl::AndOp>(
          loc(), i32, v, constInt(i32, 0xFFFF)).getResult();
      mlir::Value hi = builder.create<awl::AndOp>(
          loc(), i32, v, constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult();
      // swapEndianWord(lo) = ((lo << 8) | (lo >> 8)) & 0xFFFF, via shift ops
      mlir::Value swapped = builder.create<awl::AndOp>(
          loc(), i32,
          builder.create<awl::OrOp>(
              loc(), i32,
              builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("sld"),
                                           lo, constInt(i32, 8)).getResult(),
              builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("srd"),
                                           lo, constInt(i32, 8)).getResult()).getResult(),
          constInt(i32, 0xFFFF)).getResult();
      pushAcc(builder.create<awl::OrOp>(loc(), i32, hi, swapped).getResult());
      return true;
    }
    // SPEC[AWL] awlsim test directive __ASSERT<cmp> (value oracles, runtime scl_rt_assert_cmp[_r])
    if (m.rfind("__ASSERT", 0) == 0)
      return parseAssert();
    // SPEC[AWL] IEC 61131-3 STL block calls (CALL/UC/CC, incl. parameterized CALL and indirect UC FC [MW x]), see parseCall
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

  // reinterpret the bits (DTR, TRUNC/RND input): int <-> float, same width
  mlir::Value bitcastTo(mlir::Value v, mlir::Type t) {
    if (v.getType() == t)
      return v;
    auto op = builder.create<awl::CastOp>(loc(), t, v);
    op->setAttr("bitcast", builder.getUnitAttr());
    return op;
  }

  // loads of BYTE/WORD into an accumulator zero-extend (awlsim: W/B are
  // unsigned; the ACCU holds 0..65535)
  mlir::Value zextAcc(mlir::Value v) {
    if (v.getType().isInteger(8) || v.getType().isInteger(16)) {
      auto op = builder.create<awl::CastOp>(loc(), builder.getI32Type(), v);
      op->setAttr("zext", builder.getUnitAttr());
      return op;
    }
    return v;
  }

  // zero-extend a 32-bit value to i64 — pointer values are unsigned (an
  // S7 pointer like 0x84010000 must not be sign-extended when moved into
  // an address register via LARn/TARn)
  mlir::Value zextTo64(mlir::Value v) {
    if (v.getType().isInteger(32)) {
      auto op = builder.create<awl::CastOp>(loc(), builder.getI64Type(), v);
      op->setAttr("zext", builder.getUnitAttr());
      return op;
    }
    return castTo(v, builder.getI64Type());
  }

  mlir::Value constInt(mlir::Type ty, int64_t v) {
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(ty))
      return builder.create<awl::ConstOp>(loc(), ty,
                                          builder.getFloatAttr(ft, (double)v));
    return builder.create<awl::ConstOp>(loc(), ty, builder.getIntegerAttr(ty, v));
  }

  // mux(cond, a, b) = cond ? a : b on integer types (arithmetic select; both
  // sides are always evaluated — callers must ensure that is safe)
  mlir::Value mux(mlir::Value cond, mlir::Value a, mlir::Value b) {
    auto ty = a.getType();
    auto cop = builder.create<awl::CastOp>(loc(), ty, cond);
    cop.setZextAttr(builder.getUnitAttr());
    mlir::Value c = cop.getResult();
    mlir::Value nc = builder.create<awl::SubOp>(
        loc(), ty, constInt(ty, 1), c).getResult();
    return builder.create<awl::AddOp>(
        loc(), ty,
        builder.create<awl::MulOp>(loc(), ty, a, c).getResult(),
        builder.create<awl::MulOp>(loc(), ty, b, nc).getResult()).getResult();
  }
  // i1 variant: (a & cond) | (b & !cond)
  mlir::Value muxI1(mlir::Value cond, mlir::Value a, mlir::Value b) {
    auto i1 = builder.getI1Type();
    return builder.create<awl::OrOp>(
        loc(), i1,
        builder.create<awl::AndOp>(loc(), i1, a, cond).getResult(),
        builder.create<awl::AndOp>(loc(), i1, b,
                                  builder.create<awl::NotOp>(loc(), i1, cond).getResult()).getResult()).getResult();
  }

  // The S7 accumulators are initialized to 0; pad a short stack so that
  // arithmetic/comparisons on an "empty" accumulator still produce SSA.
  // The accumulator stack is always exactly 4 entries deep (ACCU4..ACCU1,
  // acc.back() = ACCU1), matching the awlsim 4-accu CPU: L shifts down,
  // arith pops shift up. This keeps the runtime ACCU sync/reload at labels
  // consistent.
  void padAcc() {
    // pad by repeating the bottom entry (ACCU4): awlsim arith shifts
    // ACCU3 := ACCU4, so the deepest register is duplicated, not zeroed
    while (acc.size() < 4)
      acc.insert(acc.begin(), acc.empty() ? constInt(builder.getI32Type(), 0)
                                           : acc.front());
  }
  // push a result at ACCU1 (arith/unary ops pop first, so the depth stays
  // <= 4; cap by dropping ACCU4 if a push happens at full depth)
  void pushAcc(mlir::Value v) {
    if (acc.size() >= 4)
      acc.erase(acc.begin());
    acc.push_back(v);
  }
  // L (awlsim insn_l): ACCU2 := ACCU1, ACCU1 := value; deeper accus unchanged
  void loadAcc(mlir::Value v) {
    padAcc();
    acc[acc.size() - 2] = acc[acc.size() - 1];
    acc.back() = v;
  }
  // ENT (awlsim insn_ent): ACCU4 := ACCU3, ACCU3 := ACCU2 (top two unchanged)
  void entAcc() {
    padAcc();
    for (size_t i = 0; i + 2 < acc.size(); ++i)
      acc[i] = acc[i + 1];
  }
  // LEAVE/POP (awlsim insn_leave): ACCU2 := ACCU3, ACCU3 := ACCU4
  // (acc[size-2] = acc[size-3], acc[size-3] = acc[size-4]; ACCU1 unchanged)
  void leaveAcc() {
    padAcc();
    for (size_t i = acc.size() - 1; i >= 2; --i)
      acc[i - 1] = acc[i - 2];
  }

  // awlsim keeps ACCU1..4 in runtime registers that persist across labels
  // and blocks. The frontend keeps a parse-time SSA stack for straight-line
  // code, syncs it to the runtime registers at every statement end, and
  // reloads it from the runtime at labels/block entry (SSA values cannot
  // cross the label's block boundary after lowering splits at labels).
  mlir::Value accuBits(mlir::Value v) {
    auto i32 = builder.getI32Type();
    if (v.getType() == i32)
      return v;
    if (v.getType().isa<mlir::FloatType>())
      return bitcastTo(v, i32);
    return castTo(v, i32);
  }
  // write the parse-time acc stack into the runtime ACCU registers
  // (acc.back() = ACCU1; acc[0] = ACCU4 when the stack is full)
  void syncAcc() {
    padAcc(); // all four ACCU registers are always written
    size_t n = std::min(acc.size(), (size_t)4);
    for (size_t k = 0; k < n; ++k) {
      mlir::Value v = accuBits(acc[acc.size() - 1 - k]);
      builder.create<awl::AccuSetOp>(
          loc(), builder.getI32IntegerAttr((int32_t)(k + 1)), v);
    }
  }
  // reload the parse-time acc stack from the runtime ACCU registers
  void reloadAcc() {
    acc.clear();
    for (int32_t i = 4; i >= 1; --i)
      acc.push_back(builder.create<awl::AccuGetOp>(
          loc(), builder.getI32Type(), builder.getI32IntegerAttr(i)));
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

  mlir::Value i1Const(bool v) {
    return builder.create<awl::ConstOp>(
        loc(), builder.getI1Type(),
        builder.getIntegerAttr(builder.getI1Type(), v ? 1 : 0));
  }
  // status-word logic state with defaults (block start / after BE: VKE=1,
  // OR=0, STA=1, NER=0)
  mlir::Value vkeD() { return rlo ? rlo : i1Const(true); }
  mlir::Value orD() { return stOr ? stOr : i1Const(false); }
  mlir::Value staD() { return stSta ? stSta : i1Const(true); }
  mlir::Value nerD() { return stNer ? stNer : i1Const(false); }

  bool ensureRlo() {
    // The RLO (VKE) value persists across statements; it defaults to 1 at
    // block start. The full awlsim logic formula is applied in
    // parseRloLogic/parseCloseBracket using the NER/OR/STA state bits.
    if (!rlo)
      rlo = i1Const(true);
    rloActive = true;
    return true;
  }

  // A logic instruction without an operand at the end of a chain. awlsim
  // insn_o __run_noOps (bare O): OR=(VKE|OR)&NER, STA=1, NER=VKE&NER, VKE
  // unchanged. Bare U/A have no awlsim semantics (they require an operand),
  // so they leave the state untouched.
  bool bareLogicEnd(const std::string &m) {
    if (!ensureRlo())
      return false;
    if (m == "O" || m == "ON") {
      auto i1 = builder.getI1Type();
      mlir::Value vke = vkeD(), orB = orD(), ner = nerD();
      stOr = builder.create<awl::AndOp>(
          loc(), i1,
          builder.create<awl::OrOp>(loc(), i1, vke, orB).getResult(),
          ner).getResult();
      stNer = builder.create<awl::AndOp>(loc(), i1, vke, ner).getResult();
      stSta = i1Const(true);
    }
    return true;
  }

  // awlsim MCR model: MCR( pushes the current VKE onto the mcrStack, )MCR
  // pops it; MCRA/MCRD set mcrActive; stores execute only if
  // mcrIsOn() = !mcrActive || all(mcrStack). The MCR( brackets save the pushed
  // VKE values, so the gate is an SSA AND of them.
  mlir::Value mcrOn() {
    mlir::Value g;
    auto i1 = builder.getI1Type();
    for (auto &b : brackets)
      if (b.isMcr)
        g = g ? (mlir::Value)builder.create<awl::AndOp>(loc(), i1, g, b.savedRlo)
              : b.savedRlo;
    return g;
  }
  // RLO ANDed with the MCR gate (awlsim store condition for = / S / R bits)
  mlir::Value gatedRlo() {
    if (!ensureRlo())
      return nullptr;
    if (!mcrActive)
      return rlo;
    mlir::Value g = mcrOn();
    if (!g)
      return rlo;
    return builder.create<awl::AndOp>(loc(), builder.getI1Type(), rlo, g).getResult();
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

  // ACCU1 without consuming it: TARn/LARn with no operand copy ACCU1 to an
  // address register but leave the accumulators untouched (awlsim insn_tar:
  // ARn := ACCU1, ACCUs unchanged).
  mlir::Value peekAcc() {
    if (acc.empty())
      return constInt(builder.getI32Type(), 0);
    return acc.back();
  }


  // MARK: operands

  // P##name as a value: S7 pointer to the local (area DI=0x85 in FBs,
  // L=0x86 otherwise, member offset from the block's layout)
  mlir::Value parsePtrLocal() {
    std::string name = upperStr(lexer.getIdent());
    lexer.next();
    int32_t byte = 0, bit = 0;
    bool isParam = false;
    if (curLayout) {
      auto it = curLayout->members.find(name);
      if (it != curLayout->members.end()) {
        byte = it->second.byte;
        bit = it->second.bit;
        isParam = true;
      }
    }
    if (!isParam && curTempLayout) {
      auto it = curTempLayout->members.find(name);
      if (it != curTempLayout->members.end()) {
        byte = it->second.byte;
        bit = it->second.bit;
      }
    }
    // FB interface -> instance DB (DI, 0x85); FC interface -> the L frame
    // at the parameter area offset (VL, 0x87); TEMP -> the block's own L
    // frame (0x86). S7 32-bit format: area<<24 | db<<16 | byte<<3 | bit —
    // for VL the DB field is unused, so rawToPacked decodes a 21-bit byte
    // there (the FC parameter area starts at kLocalBase = 8192 > 8191).
    int64_t area;
    if (curBlockFb)
      area = 0x85;
    else if (isParam) {
      area = 0x87;
      byte += kLocalBase;
    } else
      area = 0x86;
    return constInt(builder.getI64Type(),
                    (area << 24) | ((int64_t)byte * 8 + bit));
  }

  // create a direct operand, clamping whole-UDT/STRING widths to 32
  mlir::Value mkDirect(mlir::scl::DirectArea area, int32_t db, int32_t byte,
                       int32_t bit, int32_t width) {
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      width = 32;
    if (area == mlir::scl::DirectArea::DataBlock && db > 0) {
      int32_t hi = byte + (width + 7) / 8;
      auto &hw = dbRefBytes[db];
      if (hi > hw)
        hw = hi;
    }
    return builder.create<awl::DirectOp>(loc(), area, db, byte, bit, width);
  }

  // DBLG/DBNO/DILG/DINO yield an i64 runtime value (awl.db_info), not a
  // packed pointer — must not be treated as an indirect address
  static bool isPseudoValue(mlir::Value v) {
    return v.getType().isInteger(64) &&
           mlir::isa_and_nonnull<awl::DbInfoOp>(v.getDefiningOp());
  }

  // instruction keywords that can follow a bare LARn/TARn on the next line
  static bool isInsnKeyword(const std::string &w) {
    return w == "L" || w == "T" || w == "U" || w == "UN" || w == "O" ||
           w == "ON" || w == "A" || w == "AN" || w == "NOT" || w == "SET" ||
           w == "CLR" || w == "BE" || w == "BEA" || w == "BEB" || w == "BEC" ||
           w == "NOP" || w == "CALL" || w == "UC" || w == "CC" ||
           w == "SPA" || w == "SPB" || w == "SPN" || w == "JU" || w == "JC" ||
           w == "JCN" || w == "SAVE" || w == "LC" || w == "FR" || w == "S" ||
           w == "R" || w == "ZV" || w == "ZR";
  }

  static bool isAreaLetter(char c) {
    // German E/A/Z and English I/Q/C/L (L = local data, own area)
    return c == 'E' || c == 'A' || c == 'M' || c == 'T' || c == 'Z' ||
           c == 'I' || c == 'Q' || c == 'C' || c == 'L' || c == 'V';
  }
  static mlir::scl::DirectArea areaOf(char c) {
    switch (c) {
    case 'E': case 'I': return mlir::scl::DirectArea::Input;
    case 'A': case 'Q': return mlir::scl::DirectArea::Output;
    case 'M': return mlir::scl::DirectArea::Marker;
    case 'L': return mlir::scl::DirectArea::Local;
    case 'V': return mlir::scl::DirectArea::VLocal;
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
  // SPEC[AWL] Siemens AWL manual "Indirect addressing" ([AR1, P#x.y] area-relative, memory-indirect [LD n]/[#var] base = operand value, T/Z [..] = block number)
  mlir::Value parseIndirectBracket(
      int32_t widthBits,
      mlir::scl::DirectArea area = mlir::scl::DirectArea::Marker,
      int32_t forceS7 = -1) { // -1: derive the S7 area byte from `area`
    int32_t ar = 1, offBits = 0;
    mlir::Value ldBase; // [LW n] / [#var]: base pointer = the operand's VALUE
    lexer.next(); // [
    while (lexer.getCurToken() != Tok::rbracket &&
           lexer.getCurToken() != Tok::eof) {
      if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        if (w.rfind("AR", 0) == 0 && w.size() == 3 && isdigit(w[2])) {
          ar = w[2] - '0';
          lexer.next();
        } else if (isAreaLetter(w[0]) || w.rfind("DB", 0) == 0 ||
                   w.rfind("DI", 0) == 0) {
          // [LW 0] / [MD 4]: memory-indirect — the operand's CONTENT is the
          // base pointer (e.g. stored by TARn LD n)
          mlir::Value a = parseOperand(false);
          if (!a)
            return nullptr;
          ldBase = mlir::isa<mlir::scl::DirectType>(a.getType())
                       ? zextTo64(builder.create<awl::LoadOp>(loc(), a))
                       : a;
        } else {
          lexer.next();
        }
      } else if (lexer.getCurToken() == Tok::localvar) {
        // [#VAR]: the variable's value is the base pointer
        mlir::Value a = parseOperand(false);
        if (!a)
          return nullptr;
        ldBase = mlir::isa<mlir::scl::DirectType>(a.getType())
                     ? zextTo64(builder.create<awl::LoadOp>(loc(), a))
                     : a;
      } else if (lexer.getCurToken() == Tok::integer) {
        // P#x.y lexes as one integer token: byte*8 + bit
        offBits = (int32_t)consumePtrLiteral();
      } else {
        lexer.next(); // comma etc.
      }
    }
    if (lexer.getCurToken() == Tok::rbracket)
      lexer.next();
    lastIndirectWidth = widthBits;
    lastOperandIndirect = true;
    if (ldBase) {
      if (area == mlir::scl::DirectArea::Timer ||
          area == mlir::scl::DirectArea::Counter) {
        // T/Z [LW 0]: the value is the timer/counter NUMBER, not a pointer
        ldBase = builder.create<awl::PackAreaOp>(
            loc(), builder.getI64Type(),
            builder.getI32IntegerAttr(area == mlir::scl::DirectArea::Timer
                                          ? 'T'
                                          : 'C'),
            castTo(ldBase, builder.getI32Type()), builder.getI32IntegerAttr(0));
      } else {
        // a raw P#x.y value (area byte 0) in e.g. LD 0 takes the ACCESS
        // instruction's area (awlsim resolves such pointers with the
        // operand's area): set the S7 area byte, keep db/byte/bit fields
        int32_t s7 = forceS7 >= 0 ? forceS7
                    : area == mlir::scl::DirectArea::Input    ? 0x81
                    : area == mlir::scl::DirectArea::Output   ? 0x82
                    : area == mlir::scl::DirectArea::Marker   ? 0x83
                    : area == mlir::scl::DirectArea::DataBlock ? 0x84
                    : area == mlir::scl::DirectArea::Local    ? 0x86
                    : area == mlir::scl::DirectArea::VLocal   ? 0x87
                                                              : -1;
        if (s7 > 0) {
          auto i32 = builder.getI32Type();
          mlir::Value v = castTo(ldBase, i32);
          v = builder.create<awl::AndOp>(loc(), i32, v,
                                         constInt(i32, 0x00FFFFFF));
          v = builder.create<awl::OrOp>(loc(), i32, v,
                                        constInt(i32, s7 << 24));
          ldBase = zextTo64(v);
        }
      }
      if (offBits)
        ldBase = builder.create<awl::AddOp>(
            loc(), builder.getI64Type(), ldBase,
            constInt(builder.getI64Type(), offBits));
      return ldBase;
    }
    return builder.create<awl::IndirectOp>(
        loc(), builder.getI64Type(), builder.getI32IntegerAttr(ar),
        builder.getI32IntegerAttr(offBits));
  }

  // DB number for a DB operand without an explicit number: inside a
  // DATA_BLOCK init section that is the DATA_BLOCK's own number.
  int32_t effDb(int32_t db) const {
    return db == 0 && dbOverride >= 0 ? dbOverride : db;
  }

  // Consume "[i, j, ...]" array index suffixes and return the offset in
  // BITS relative to the member base (per-dimension lower bounds applied;
  // BOOL arrays use one bit per element).
  int32_t consumeArrayIndices(const MemberInfo *mi) {
    int32_t off = 0;
    size_t dim = 0;
    while (lexer.getCurToken() == Tok::lbracket) {
      lexer.next();
      while (lexer.getCurToken() != Tok::rbracket &&
             lexer.getCurToken() != Tok::eof) {
        if (lexer.getCurToken() == Tok::integer && mi) {
          int64_t idx = lexer.getInteger();
          lexer.next();
          int32_t lo = mi->lo;
          int64_t mult = (mi->elemSize ? mi->elemSize * 8 : 1);
          if (dim < mi->dims.size()) {
            lo = mi->dims[dim].first;
            for (size_t j = dim + 1; j < mi->dims.size(); ++j)
              mult *= mi->dims[j].second - mi->dims[j].first + 1;
          } else {
            mult = mi->stride ? mi->stride * 8 : 8;
          }
          off += (int32_t)((idx - lo) * mult);
          ++dim;
        } else {
          lexer.next(); // comma etc.
        }
      }
      if (lexer.getCurToken() == Tok::rbracket)
        lexer.next();
    }
    return off;
  }

  // apply an index-bit offset to a byte/bit pair
  static void addBits(int32_t &byte, int32_t &bit, int32_t bits) {
    int32_t total = bit + bits;
    byte += total / 8;
    bit = total % 8;
  }

  // awlsim statusword bit names: NER=0 VKE/RLO=1 STA=2 OR=3 OS=4 OV=5
  // A0/CC0=6 A1/CC1=7 BIE/BR=8. Returns -1 for unknown names.
  static int stwBitNumber(const std::string &name) {
    if (name == "NER") return 0;
    if (name == "VKE" || name == "RLO") return 1;
    if (name == "STA") return 2;
    if (name == "OR") return 3;
    if (name == "OS") return 4;
    if (name == "OV") return 5;
    if (name == "A0" || name == "CC0") return 6;
    if (name == "A1" || name == "CC1") return 7;
    if (name == "BIE" || name == "BR") return 8;
    return -1;
  }

  // SPEC[AWL] IEC 61131-3 STL / Siemens AWL manual "Addressing modes": absolute (M 10.0, DB1.DBW 4), indirect ([AR1, P#x.y], [LD n], [#var]), symbolic, P# pointer literals
  mlir::Value parseOperand(bool bitContext = false) {
    ptrParamSlot = -1;
    lastOperandIndirect = false;
    // operand starting with '[': fully indirect U [AR1, P#0.0]
    if (lexer.getCurToken() == Tok::lbracket)
      return parseIndirectBracket(bitContext ? 1 : 16,
                                  mlir::scl::DirectArea::Marker);
    // #name: local variable of the block; with a known interface layout it
    // resolves to the member's offset (instance DB for FB, L area for FC),
    // otherwise a marker bit by name hash
    if (lexer.getCurToken() == Tok::localvar) {
      if (lexer.curPtrLocal())
        return parsePtrLocal();
      std::string name = upperStr(lexer.getIdent());
      lexer.next();
      const MemberInfo *mi = nullptr;
      bool isParamMember = false; // interface member (vs. TEMP var)
      mlir::scl::DirectArea area = mlir::scl::DirectArea::Marker;
      int32_t byte = 0;
      if (curLayout) {
        auto it = curLayout->members.find(name);
        if (it != curLayout->members.end()) {
          mi = &it->second;
          isParamMember = true;
          // FB interface lives in the instance DB; FC interface lives in
          // the caller's L frame at the parameter area offset, i.e. the
          // VL area from inside the callee (awlsim MEM_VL)
          area = curBlockFb ? mlir::scl::DirectArea::DataBlock
                            : mlir::scl::DirectArea::VLocal;
          byte = curBlockFb ? mi->byte : kLocalBase + mi->byte;
        }
      }
      if (!mi && curTempLayout) {
        // TEMP variable: L area of the block's own stack frame
        auto it = curTempLayout->members.find(name);
        if (it != curTempLayout->members.end()) {
          mi = &it->second;
          area = mlir::scl::DirectArea::Local;
          byte = mi->byte;
        }
      }
      if (mi) {
        // TIMER/COUNTER typed variables store the timer/counter NUMBER in
        // the member's storage; accesses indirect through the T/C area at
        // that number: load the number and pack a dynamic T/C pointer (i64)
        if (mi->type == "TIMER" || mi->type == "COUNTER") {
          mlir::scl::DirectArea storeArea = area;
          int32_t storeDb =
              storeArea == mlir::scl::DirectArea::DataBlock ? -1 : 0;
          auto numAddr = mkDirect(storeArea, storeDb, byte, 0, 16);
          mlir::Value num = builder.create<awl::LoadOp>(loc(), numAddr);
          lastOperandType = mi->type;
          return builder.create<awl::PackAreaOp>(
              loc(), builder.getI64Type(),
              builder.getI32IntegerAttr(mi->type == "TIMER" ? 'T' : 'C'),
              castTo(num, builder.getI32Type()),
              builder.getI32IntegerAttr(0));
        }
        if (mi->type == "TIMER")
          area = mlir::scl::DirectArea::Timer;
        else if (mi->type == "COUNTER")
          area = mlir::scl::DirectArea::Counter;
        // array index / UDT sub-member suffixes: #ARR[1].SUB.X
        // (the deref below must use the ROOT param's slot: after descending
        // into .SUB members `mi` points at the leaf, not the by-ref param)
        bool rootByRefParam = isParamMember && !curBlockFb &&
                              (isByRefType(mi->type) || mi->type == "ARRAY");
        int32_t rootParamByte = byte;
        std::string rootParamType = mi->type;
        int32_t bit = mi->bit;
        int32_t width = mi->widthBits ? mi->widthBits : 16;
        std::string type = mi->type;
        int32_t stride = mi->stride;
        while (lexer.getCurToken() == Tok::lbracket ||
               lexer.getCurToken() == Tok::dot) {
          if (lexer.getCurToken() == Tok::lbracket) {
            if (mi->type.rfind("STRING", 0) == 0) {
              // STRING[i]: char access — chars start at byte 2, one byte
              // each, 1-based index (awlsim AwlDataType string indexing)
              lexer.next();
              int32_t idx = 1;
              if (lexer.getCurToken() == Tok::integer) {
                idx = (int32_t)lexer.getInteger();
                lexer.next();
              }
              while (lexer.getCurToken() != Tok::rbracket &&
                     lexer.getCurToken() != Tok::eof)
                lexer.next();
              if (lexer.getCurToken() == Tok::rbracket)
                lexer.next();
              byte += 2 + (idx - 1); // chars start at byte 2, 1-based
              bit = 0;
              width = 8;
              type = "CHAR";
            } else {
              addBits(byte, bit, consumeArrayIndices(mi));
              if (!mi->elemType.empty())
                type = mi->elemType; // array element type for .sub chains
            }
          } else {
            lexer.next(); // '.'
            if (lexer.getCurToken() == Tok::ident ||
                lexer.getCurToken() == Tok::localvar) {
              std::string sub = upperStr(lexer.getIdent());
              lexer.next();
              const Layout *next = nullptr;
              if (mi && mi->nested)
                next = mi->nested.get();
              else if (mi && mi->type == "ARRAY" && mi->elemNested)
                next = mi->elemNested.get();
              else {
                auto uit = udtLayouts.find(type);
                if (uit != udtLayouts.end())
                  next = &uit->second;
              }
              if (next) {
                auto sit = next->members.find(sub);
                if (sit != next->members.end()) {
                  byte += sit->second.byte;
                  bit = sit->second.bit;
                  width = sit->second.widthBits ? sit->second.widthBits : 16;
                  type = sit->second.type;
                  stride = sit->second.stride;
                  mi = &sit->second;
                }
              }
            }
          }
        }
        lastOperandType = type;
        if (isParamMember && !curBlockFb && mi->type == "POINTER")
          ptrParamSlot = byte; // store/load writes/reads the 6-byte header
        if (rootByRefParam) {
          // FC by-ref compound param (STRING/ANY/DATE_AND_TIME/ARRAY): the
          // VL slot holds a 6-byte pointer header { db-nr word @+0, S7 ptr
          // dword @+2 with db field 0 }; dereference it — combine the
          // header's DB number into the raw pointer (awlsim resolves the
          // compound via the header's dbNumber, not the DB register), add
          // the member access offset in bits. The runtime's rawToPacked
          // decodes the raw pointer in the indirect load/store.
          int32_t slotBase = rootParamByte;
          int64_t relBits = (int64_t)(byte - slotBase) * 8 + bit;
          auto hdr = mkDirect(mlir::scl::DirectArea::VLocal, 0, slotBase + 2, 0, 32);
          mlir::Value raw =
              castTo(builder.create<awl::LoadOp>(loc(), hdr), builder.getI32Type());
          if (rootParamType != "ANY") {
            // Combine the header's DB number into the raw pointer, but only
            // for DB/DI-area headers: the staged dword has db field 0 there,
            // while for VL (0x87) bits 16-23 are part of the 21-bit byte
            // field (kLocalBase = 8192 needs them) and must stay untouched.
            auto dbwAddr =
                mkDirect(mlir::scl::DirectArea::VLocal, 0, slotBase, 0, 16);
            mlir::Value dbw = castTo(
                builder.create<awl::LoadOp>(loc(), dbwAddr), builder.getI32Type());
            dbw = builder.create<awl::AndOp>(loc(), builder.getI32Type(), dbw,
                                             constInt(builder.getI32Type(), 0xFFFF));
            auto i32 = builder.getI32Type();
            mlir::Value areaBits = builder.create<awl::AndOp>(
                loc(), i32, raw, constInt(i32, (int32_t)0xFF000000));
            mlir::Value isDb = builder.create<awl::CmpEqOp>(
                loc(), builder.getI1Type(), areaBits,
                constInt(i32, (int32_t)0x84000000));
            mlir::Value isDi = builder.create<awl::CmpEqOp>(
                loc(), builder.getI1Type(), areaBits,
                constInt(i32, (int32_t)0x85000000));
            auto zextI1 = [&](mlir::Value b) {
              auto op = builder.create<awl::CastOp>(loc(), i32, b);
              op->setAttr("zext", builder.getUnitAttr());
              return mlir::Value(op);
            };
            mlir::Value f =
                builder.create<awl::OrOp>(loc(), i32, zextI1(isDb), zextI1(isDi));
            mlir::Value fmask = builder.create<awl::MulOp>(loc(), i32, f,
                                                           constInt(i32, 0xFFFF));
            mlir::Value dbwEff =
                builder.create<awl::AndOp>(loc(), i32, dbw, fmask);
            mlir::Value dbwSh = builder.create<awl::MulOp>(loc(), i32, dbwEff,
                                                           constInt(i32, 0x10000));
            raw = builder.create<awl::OrOp>(loc(), i32, raw, dbwSh);
          }
          mlir::Value p = zextTo64(raw);
          if (relBits)
            p = builder.create<awl::AddOp>(
                loc(), builder.getI64Type(), p,
                constInt(builder.getI64Type(), relBits));
          lastIndirectWidth = width != 1 && width != 8 && width != 16 && width != 32
                                  ? 32
                                  : width;
          return p;
        }
        // FB locals live in the instance DB addressed via the DI register
        // (db -1); DI-area accesses add the AR2 multi-instance base offset
        return mkDirect(area, area == mlir::scl::DirectArea::DataBlock ? -1 : 0,
                        byte, bit, width);
      }
      lastOperandType = "";
      uint32_t h = 5381;
      for (char c : name)
        h = h * 33 + (unsigned char)c;
      return builder.create<awl::DirectOp>(
          loc(), mlir::scl::DirectArea::Marker, 0, (int32_t)(h % 4096),
          (int32_t)((h >> 12) % 8), 1);
    }
    if (lexer.getCurToken() != Tok::ident)
      return (error("expected operand"), nullptr);
    std::string raw = lexer.getIdent().str();
    std::string w = upperIdent();
    bool quoted = lexer.curQuoted();
    char quoteChar = lexer.curQuoteChar();
    lexer.next();
    if (quoted) {
      if (quoteChar == '\'' && raw.size() <= 4) {
        // char literal: 'A' -> 0x41, 'AB' -> 0x4142, 'ABCD' -> 0x41424344
        // (first char in the most significant byte, latin-1)
        uint32_t v = 0;
        for (char c : raw)
          v = (v << 8) | (unsigned char)c;
        return constInt(builder.getI32Type(), v);
      }
      // quoted DATA_BLOCK name with a member: "Data_1".VAR0 [i] .sub
      auto dbIt = namedDbs.find(w);
      if (dbIt != namedDbs.end() && lexer.getCurToken() == Tok::dot) {
        lexer.next(); // dot
        if (lexer.getCurToken() == Tok::ident ||
            lexer.getCurToken() == Tok::localvar) {
          std::string member = upperStr(lexer.getIdent());
          lexer.next();
          return parseDbMemberSymbolic(dbIt->second, member, bitContext);
        }
      }
      return symbolicOperand(w); // "symbol" / 'literal' in operand position
    }
    // pseudo operands DBLG/DBNO/DILG/DINO (DB length/number): runtime info;
    // db = 0 refers to the current DB (AUF/OPN)
    if (w == "DBLG" || w == "DBNO" || w == "DILG" || w == "DINO") {
      int kind = w == "DBLG" ? 0 : w == "DBNO" ? 1 : w == "DILG" ? 2 : 3;
      return builder.create<awl::DbInfoOp>(loc(), builder.getI64Type(),
                                           builder.getI32IntegerAttr(kind),
                                           builder.getI32IntegerAttr(0));
    }

    // __STW <bitname|n>: individual status-word bit -> i1
    if (w == "__STW") {
      int bitNo = -1;
      if (lexer.getCurToken() == Tok::ident) { // w's lexer.next() already done
        bitNo = stwBitNumber(upperStr(lexer.getIdent()));
        lexer.next();
      } else if (lexer.getCurToken() == Tok::integer) {
        bitNo = (int)lexer.getInteger();
        lexer.next();
      }
      if (bitNo >= 0)
        return builder.create<awl::StwBitOp>(loc(), builder.getI1Type(),
                                             builder.getI32IntegerAttr(bitNo));
      return builder.create<awl::StwGetOp>(loc(), builder.getI32Type());
    }
    // STW: status word (L STW / T STW)
    if (w == "STW")
      return builder.create<awl::StwGetOp>(loc(), builder.getI32Type());
    // individual STW bits (U OV / U BIE ...): NER=0 VKE=1 STA=2 OR=3 OS=4
    // OV=5 CC0=6 CC1=7 BIE/BR=8
    {
      // A0/A1 are deliberately absent here: they collide with output-area
      // operands (A1.0); the CC0/CC1 spellings and __STW A0 / __STW A1
      // cover the STW bits
      static const char *stwBitNames[] = {"NER", "VKE", "RLO", "STA", "OR",
                                          "OS",  "OV",  "CC0", "CC1", "BIE",
                                          "BR"};
      for (auto name : stwBitNames)
        if (w == name)
          return builder.create<awl::StwBitOp>(
              loc(), builder.getI1Type(),
              builder.getI32IntegerAttr(stwBitNumber(name)));
      if (w == "UO") // unordered condition: A0 && A1
        return builder.create<awl::StwCondOp>(loc(), builder.getI1Type(),
                                              builder.getI32IntegerAttr(6));
    }
    // DI-area operands (DIX/DIB/DIW/DID): the instance DB in the DI
    // register — db = -1 selects it at runtime
    if (w.rfind("DI", 0) == 0 && w.size() >= 3 &&
        (w[2] == 'X' || w[2] == 'B' || w[2] == 'W' || w[2] == 'D')) {
      int32_t width = w[2]=='X'?1:w[2]=='B'?8:w[2]=='W'?16:32;
      if (lexer.getCurToken() == Tok::lbracket)
        return parseIndirectBracket(width, mlir::scl::DirectArea::DataBlock,
                                    0x85); // DIX [LD 0]: DI area
      int32_t byte = 0, bit = 0;
      bool hasBit = false;
      double realBit = -1;
      if (lexer.getCurToken() == Tok::integer) {
        byte = (int32_t)lexer.getInteger();
        lexer.next();
      } else if (lexer.getCurToken() == Tok::real) {
        double r = lexer.getReal();
        byte = (int32_t)r;
        realBit = r - byte;
        lexer.next();
      }
      if (realBit >= 0) {
        bit = (int32_t)(realBit * 10 + 0.5);
        hasBit = true;
      } else if (lexer.getCurToken() == Tok::dot) {
        lexer.next();
        bit = (int32_t)lexer.getInteger();
        lexer.next();
        hasBit = true;
      }
      if (width == 1 && !hasBit)
        return (error("DIX bit access requires a bit number"), nullptr);
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::DataBlock,
                                            -1, byte, bit, width);
    }
    // width letter first = indirect access: L B [AR1, P#0.0] — the address
    // register contents are not modelled; use the P# byte offset from the
    // brackets in the marker area
    if (w == "B" || w == "W" || w == "D") {
      int32_t width = w[0] == 'B' ? 8 : w[0] == 'W' ? 16 : 32;
      if (lexer.getCurToken() == Tok::lbracket)
        return parseIndirectBracket(width, mlir::scl::DirectArea::Local,
                                    0x86); // B/W/D [AR1, P#x.y]: L area
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
        return parseIndirectBracket(width, mlir::scl::DirectArea::DataBlock,
                                    0x84); // DBW [AR1, P#x.y]: DB area
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
        // indices): resolve through the DATA_BLOCK's layout
        return parseDbMemberSymbolic(db, dw, bitContext);
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
      return mkDirect(mlir::scl::DirectArea::DataBlock, effDb(db), byte, bit,
                      width);
    }

    // E/A/M/T/Z [B|W|D] [byte] [.bit]; P[E|A][B|W|D] = peripheral I/Q
    bool isPeriph = w.size() >= 2 && w[0] == 'P' &&
                    (w[1] == 'E' || w[1] == 'A' || w[1] == 'I' || w[1] == 'Q');
    if (w.size() < 1 || !(isAreaLetter(w[0]) || isPeriph))
      return symbolicOperand(w); // awlsim symbol/string in operand position
    auto area = isPeriph ? (w[1] == 'E' || w[1] == 'I' ? mlir::scl::DirectArea::Input
                                                      : mlir::scl::DirectArea::Output)
                         : areaOf(w[0]); // 'L' -> Local area (L-stack frame)
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
      return parseIndirectBracket(width, area);
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
    // i64: indirect bit operand [AR1, P#x.y] (packed pointer); T/C-area
    // directs are S/R on timers/counters (not bit operands, but valid here)
    if (v.getType().isInteger(1))
      return v; // STW bit/condition (stw_bit / stw_cond) is already i1
    if (!isBit(v) && !v.getType().isInteger(64)) {
      if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(v.getType()))
        if (dt.getArea() == mlir::scl::DirectArea::Timer ||
            dt.getArea() == mlir::scl::DirectArea::Counter)
          return v;
      return (error("expected a bit operand (e.g. M 0.0)"), nullptr);
    }
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
      bool pl = lexer.curPtrLit();
      int lw = lexer.getLitWidth();
      // P#... literals: resolve symbolic members (P#DB42.DTVAR1) to their
      // byte/bit offset via consumePtrLiteral
      int64_t v = pl ? consumePtrLiteral() : lexer.getInteger();
      if (!pl)
        lexer.next();
      if (neg)
        v = -v;
      if (pl) {
        // a P#DBn.../P#DIn... literal loaded as a 32-bit value carries the
        // DB number only in the 48-bit DBPointer, not in the pointer dword
        // (awlsim: setDWord(AREA_DB<<24 | offset), dbNr separate)
        uint64_t a = ((uint64_t)v >> 24) & 0xFF;
        if (a == 0x84 || a == 0x85)
          v = (int64_t)((uint64_t)v & 0xFF00FFFFll);
      }
      mlir::Type ty;
      int64_t val;
      if (lw == 8) { // B#.. : unsigned byte, zero-extended into ACCU
        ty = builder.getI32Type();
        val = (v % 256 + 256) % 256;
      } else if (lw == 16) { // W#../DATE : unsigned word, zero-extended
        ty = builder.getI32Type();
        val = (v % 65536 + 65536) % 65536;
      } else if (lw == 32) { // L#../TIME/TOD : 32-bit pattern
        ty = builder.getI32Type();
        val = (int32_t)(uint32_t)v;
      } else if (v >= -32768 && v <= 32767) {
        ty = builder.getI32Type();
        val = (v % 65536 + 65536) % 65536; // 16-bit value, zero-extended
      } else if (v >= -2147483648LL && v <= 4294967295LL) {
        ty = builder.getI32Type();
        val = (int32_t)(uint32_t)v; // 32-bit bit pattern
      } else {
        ty = builder.getI64Type();
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

  // AUF/OPN with a runtime DB number: `AUF #INPUT_DB` (a BLOCK_DB FC
  // parameter) or `AUF DB [LW 0]` (DB number in the L area)
  bool parseSetDbValue(bool di = false) {
    mlir::Value addr;
    if (lexer.getCurToken() == Tok::localvar) {
      addr = parseOperand(false);
      if (!addr)
        return false;
    } else if (lexer.getCurToken() == Tok::lbracket) {
      lexer.next();
      if (lexer.getCurToken() == Tok::localvar) {
        addr = parseOperand(false); // AUF DB [#DBNR]
        if (!addr)
          return false;
      } else if (lexer.getCurToken() == Tok::ident) {
        std::string w = upperIdent();
        lexer.next();
        int32_t width = 16;
        if (w.size() >= 2 && w[0] == 'L') {
          char c = w[1];
          width = c == 'X' ? 1 : c == 'B' ? 8 : c == 'W' ? 16 : 32;
        }
        int32_t byte = 0;
        if (lexer.getCurToken() == Tok::integer) {
          byte = (int32_t)lexer.getInteger();
          lexer.next();
        }
        addr = builder.create<awl::DirectOp>(
            loc(), w[0] == 'L' ? mlir::scl::DirectArea::Local
                               : mlir::scl::DirectArea::Marker,
            0, byte, 0, width);
      }
      while (lexer.getCurToken() != Tok::rbracket &&
             lexer.getCurToken() != Tok::eof)
        lexer.next();
      if (lexer.getCurToken() == Tok::rbracket)
        lexer.next();
    } else {
      return false;
    }
    if (!addr)
      return false;
    mlir::Value v = mlir::isa<mlir::scl::DirectType>(addr.getType())
                        ? (mlir::Value)builder.create<awl::LoadOp>(loc(), addr)
                        : addr;
    if (di)
      builder.create<awl::SetDivOp>(loc(), castTo(v, builder.getI64Type()));
    else
      builder.create<awl::SetDbvOp>(loc(), castTo(v, builder.getI64Type()));
    return true;
  }

  bool parseLoad() {
    // L <constant> | L <operand>
    mlir::Value v;
    if (lexer.getCurToken() == Tok::integer ||
        lexer.getCurToken() == Tok::real || lexer.getCurToken() == Tok::minus) {
      v = parseConst();
    } else if (lexer.getCurToken() == Tok::localvar && lexer.curPtrLocal()) {
      v = parsePtrLocal(); // L P##name: the pointer value itself
    } else {
      mlir::Value addr = parseOperand(false);
      if (!addr)
        return false;
      if (isPseudoValue(addr)) {
        // pseudo operands (DBLG/DBNO/DILG/DINO) already yield a value
        v = addr;
      } else if (addr.getType().isInteger(64)) {
        // indirect load L/LC [AR1, P#x.y] through the packed pointer (also
        // dynamic T/C pointers from TIMER/COUNTER vars: L #counter_var)
        int32_t w = lastIndirectWidth;
        if (w != 1 && w != 8 && w != 16 && w != 32)
          w = 16; // not from a [ARn, ...] operand: default word load
        if (w == 1) {
          v = builder.create<awl::LoadBitIndirectOp>(
              loc(), builder.getI1Type(), addr);
        } else {
          mlir::Value loaded = builder.create<awl::LoadIndirectOp>(
              loc(), builder.getI64Type(), addr, builder.getI32IntegerAttr(w));
          v = zextAcc(castTo(loaded, builder.getIntegerType(w)));
        }
      } else {
        // pseudo operands (DBLG/DBNO/DILG) already yield a constant
        if (mlir::isa<mlir::scl::DirectType>(addr.getType())) {
          auto dt = mlir::cast<mlir::scl::DirectType>(addr.getType());
          v = (lastOperandType == "REAL" && dt.getWidth() == 32)
                  ? (mlir::Value)builder.create<awl::LoadOp>(
                        loc(), builder.getF32Type(), addr)
                  : zextAcc(builder.create<awl::LoadOp>(loc(), addr));
        } else {
          v = addr;
        }
      }
    }
    if (!v)
      return false;
    loadAcc(v); // awlsim insn_l: ACCU2 := ACCU1, ACCU1 := value
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
    // awlsim insn_T: inside an open MCR area the store is skipped unless
    // mcrIsOn() — emulated by storing old + (v-old)*gate
    mlir::Value mcrG = mcrActive ? mcrOn() : mlir::Value();
    if (isPseudoValue(addr)) {
      popAcc(); // store to a pseudo operand: value dropped
    } else if (addr.getType().isInteger(64)) {
      // indirect store T [AR1, P#x.y]
      mlir::Value sv = castTo(v, builder.getI64Type());
      if (mcrG) {
        mlir::Value old = builder.create<awl::LoadIndirectOp>(
            loc(), builder.getI64Type(), addr);
        auto gzOp = builder.create<awl::CastOp>(loc(), builder.getI64Type(), mcrG);
        gzOp.setZextAttr(builder.getUnitAttr()); // i1 gate: 0/1, not 0/-1
        mlir::Value gz = gzOp.getResult();
        sv = builder.create<awl::AddOp>(
            loc(), builder.getI64Type(), old,
            builder.create<awl::MulOp>(
                loc(), builder.getI64Type(),
                builder.create<awl::SubOp>(loc(), builder.getI64Type(), sv, old).getResult(),
                gz).getResult()).getResult();
      }
      builder.create<awl::StoreIndirectOp>(
          loc(), sv, addr,
          builder.getI32IntegerAttr(lastIndirectWidth));
    } else if (mlir::isa_and_nonnull<awl::StwGetOp>(addr.getDefiningOp())) {
      // T STW: write the status word
      auto stwVal = castTo(v, builder.getI32Type());
      builder.create<awl::StwSetOp>(loc(), stwVal);
      // with a constant value, fold the written bits into the parse-time
      // logic state (NER=bit0, VKE=bit1, STA=bit2, OR=bit3)
      if (auto c = mlir::dyn_cast_or_null<awl::ConstOp>(v.getDefiningOp())) {
        int64_t w = c.getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();
        stNer = i1Const(w & 1);
        rlo = i1Const((w >> 1) & 1);
        stSta = i1Const((w >> 2) & 1);
        stOr = i1Const((w >> 3) & 1);
        rloActive = true;
      }
    } else if (mlir::isa<mlir::scl::DirectType>(addr.getType())) {
      if (ptrParamSlot >= 0) {
        // T #ptrParam: write the 6-byte DB-pointer header into the VL slot
        // { db-nr word @+0, ptr dword @+2 with db field 0 }
        int64_t raw = 0;
        bool isConst = false;
        if (auto c = mlir::dyn_cast_or_null<awl::ConstOp>(v.getDefiningOp())) {
          raw = c.getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();
          isConst = true;
        }
        if (!isConst) {
          // non-constant pointer value: plain store fallback (rare)
          builder.create<awl::StoreOp>(loc(), v, addr);
          rloActive = false;
          return true;
        }
        auto slot = mlir::cast<mlir::scl::DirectType>(addr.getType());
        uint32_t areaS7 = (uint32_t)((raw >> 24) & 0xFF);
        uint32_t dbw = (areaS7 == 0x84 || areaS7 == 0x85)
                           ? (uint32_t)((raw >> 16) & 0xFF) : 0;
        uint32_t ptr = (uint32_t)(raw & 0xFF00FFFFll);
        builder.create<awl::StoreOp>(
            loc(), constInt(builder.getI32Type(), (int32_t)dbw),
            mkDirect(slot.getArea(), 0, ptrParamSlot, 0, 16));
        builder.create<awl::StoreOp>(
            loc(), constInt(builder.getI32Type(), (int32_t)ptr),
            mkDirect(slot.getArea(), 0, ptrParamSlot + 2, 0, 32));
      } else {
        mlir::Value sv = v;
        if (mcrG) {
          mlir::Value old = builder.create<awl::LoadOp>(loc(), addr);
          auto vt = old.getType();
          sv = castTo(sv, vt);
          old = castTo(old, vt);
          auto gzOp = builder.create<awl::CastOp>(loc(), vt, mcrG);
          gzOp.setZextAttr(builder.getUnitAttr()); // i1 gate: 0/1, not 0/-1
          mlir::Value gz = gzOp.getResult();
          sv = builder.create<awl::AddOp>(
              loc(), vt, old,
              builder.create<awl::MulOp>(
                  loc(), vt,
                  builder.create<awl::SubOp>(loc(), vt, sv, old).getResult(),
                  gz).getResult()).getResult();
        }
        builder.create<awl::StoreOp>(loc(), sv, addr);
      }
    } else {
      popAcc(); // store to a pseudo operand: value dropped
    }
    rloActive = false;
    return true;
  }

  // Combine the operand bit x into the RLO with awlsim's exact status-word
  // formulas (insn_U/insn_O/insn_X): U: VKE = ((VKE | !NER) & x) | (OR & NER),
  // O: VKE = (VKE & NER) | x, X: VKE = (VKE & NER) ^ x; then OR/STA/NER are
  // updated as in awlsim.
  void rloCombine(bool isOr, bool isXor, mlir::Value raw, bool neg = false) {
    auto i1 = builder.getI1Type();
    mlir::Value vke = vkeD(), orB = orD(), ner = nerD();
    mlir::Value x = neg ? (mlir::Value)builder.create<awl::NotOp>(loc(), i1, raw) : raw;
    mlir::Value newOr;
    if (isOr || isXor) {
      newOr = i1Const(false);
      rlo = isOr ? (mlir::Value)builder.create<awl::OrOp>(
                       loc(), i1, builder.create<awl::AndOp>(loc(), i1, vke, ner).getResult(), x)
                 : (mlir::Value)builder.create<awl::XOrOp>(
                       loc(), i1, builder.create<awl::AndOp>(loc(), i1, vke, ner).getResult(), x);
    } else {
      newOr = builder.create<awl::AndOp>(loc(), i1, orB, ner).getResult();
      rlo = builder.create<awl::OrOp>(
          loc(), i1,
          builder.create<awl::AndOp>(
              loc(), i1,
              builder.create<awl::OrOp>(loc(), i1, vke,
                                        builder.create<awl::NotOp>(loc(), i1, ner).getResult()).getResult(),
              x).getResult(),
          newOr).getResult();
    }
    stOr = newOr;
    stSta = raw; // awlsim: STA is the scanned operand, before negation
    stNer = i1Const(true);
    rloActive = true;
  }

  bool parseRloLogic(const std::string &m) {
    stmtLogic = true;
    bool isOr = (m == "O" || m == "ON");
    bool isXor = (m == "X" || m == "XN");
    bool neg = (m == "UN" || m == "ON" || m == "AN" || m == "XN");
    auto i1 = builder.getI1Type();
    if (!ensureRlo())
      return false;
    // U/O TRUE/FALSE: constant RLO operand (awlsim extension)
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "TRUE" || w == "FALSE") {
        lexer.next();
        rloCombine(isOr, isXor, i1Const(w == "TRUE"), neg);
        return true;
      }
    }
    mlir::Value addr = parseBitOperand();
    if (!addr)
      return false;
    mlir::Value bit;
    if (addr.getType().isInteger(1))
      bit = addr; // STW bit/condition operand (U __STW OV, U UO)
    else if (addr.getType().isInteger(64))
      bit = builder.create<awl::LoadBitIndirectOp>(loc(), i1, addr); // U/O [AR1, ...]
    else
      bit = builder.create<awl::LoadOp>(loc(), addr); // bit load -> i1
    rloCombine(isOr, isXor, bit, neg);
    return true;
  }

  bool parseCloseBracket() {
    stmtLogic = true;
    if (brackets.empty())
      return true; // stray ) — ignore
    Bracket b = brackets.pop_back_val();
    if (!rloActive) // e.g. MCR) without a preceding MCR(
      return true;
    auto i1 = builder.getI1Type();
    if (b.isMcr) {
      // awlsim insn_bmcr: pop the mcrStack, reset OR/STA/NER, VKE unchanged
      stOr = i1Const(false);
      stSta = i1Const(true);
      stNer = i1Const(false);
      rloActive = true;
      return true;
    }
    // awlsim insn_bend __run_UB/OB/...: combine the saved (VKE,NER,OR) with
    // the inner group result per the bracket type
    mlir::Value inner = vkeD();
    if (b.neg)
      inner = builder.create<awl::NotOp>(loc(), i1, inner);
    mlir::Value pv = b.savedRlo, pn = b.savedNer, po = b.savedOr;
    mlir::Value newOr = (b.isOr || b.isXor)
        ? (mlir::Value)i1Const(false)
        : (mlir::Value)builder.create<awl::AndOp>(loc(), i1, po, pn);
    if (b.isXor)
      rlo = builder.create<awl::XOrOp>(
          loc(), i1, builder.create<awl::AndOp>(loc(), i1, pv, pn).getResult(), inner);
    else if (b.isOr)
      rlo = builder.create<awl::OrOp>(
          loc(), i1, builder.create<awl::AndOp>(loc(), i1, pv, pn).getResult(), inner);
    else
      rlo = builder.create<awl::OrOp>(
          loc(), i1,
          builder.create<awl::AndOp>(
              loc(), i1,
              builder.create<awl::OrOp>(loc(), i1, pv,
                                        builder.create<awl::NotOp>(loc(), i1, pn).getResult()).getResult(),
              inner).getResult(),
          newOr).getResult();
    stOr = newOr;
    stSta = i1Const(true);
    stNer = i1Const(true);
    rloActive = true;
    return true;
  }

  bool parseAssign() {
    mlir::Value addr = parseBitOperand();
    if (!addr)
      return false;
    // awlsim: `=` stores VKE, or 0 when inside an inactive MCR area —
    // i.e. the stored value is the RLO ANDed with the MCR gate
    mlir::Value rv = gatedRlo();
    if (!rv)
      return false;
    if (addr.getType().isInteger(64))
      builder.create<awl::WriteBitIndirectOp>(loc(), rv, addr);
    else
      builder.create<awl::AssignOp>(loc(), rv, addr);
    // awlsim insn_assign: OR=0, STA=VKE, NER=0 (all zero under inactive MCR,
    // which is what the gated rv encodes) — mirror into the runtime STW and
    // the parse-time logic state, since `=` terminates the logic chain and
    // the statement-end mirror does not fire for it
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(3),
                                      i1Const(false)); // OR := 0
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(2),
                                      rv);         // STA := VKE
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(0),
                                      i1Const(false)); // NER := 0
    stOr = i1Const(false);
    stSta = rv;
    stNer = i1Const(false);
    rloActive = false;
    return true;
  }

  bool parseSetReset(bool set) {
    mlir::Value v = parseOperand(false);
    if (!v)
      return false;
    auto dt = v.getType().dyn_cast<mlir::scl::DirectType>();
    if (!dt && v.getType().isInteger(64)) {
      if (lastOperandType == "COUNTER") {
        // S/R #counter_var: counter op through the dynamic pointer
        mlir::Value val;
        int op;
        if (set) {
          val = acc.empty() ? constInt(builder.getI32Type(), 0) : acc.back();
          op = 2; // SCL_RT_COUNTER_SET
        } else {
          auto i16 = builder.getI16Type();
          val = builder.create<awl::ConstOp>(
              loc(), i16, builder.getIntegerAttr(i16, 0));
          op = 3; // SCL_RT_COUNTER_RESET
        }
        builder.create<awl::CounterOp>(
            loc(), mlir::TypeRange{}, mlir::ValueRange{val, v, rloInput()},
            builder.getNamedAttr("op", builder.getI32IntegerAttr(op)));
        return true;
      }
      if (lastOperandType == "TIMER" && !set) {
        builder.create<awl::TimerResetOp>(loc(), v);
        return true;
      }
      // S/R [AR1, P#x.y]: indirect bit store (gated like the direct bit path)
      auto i1 = builder.getI1Type();
      mlir::Value gate = gatedRlo();
      if (!gate)
        return false;
      mlir::Value old = builder.create<awl::LoadBitIndirectOp>(
          loc(), builder.getI1Type(), v);
      mlir::Value newVal = set
          ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, old, gate)
          : (mlir::Value)builder.create<awl::AndOp>(
                loc(), i1, old,
                builder.create<awl::NotOp>(loc(), i1, gate).getResult());
      builder.create<awl::WriteBitIndirectOp>(loc(), newVal, v);
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
    // awlsim insn_S/insn_R: the bit is stored only if VKE && mcrIsOn; emulate
    // the conditional store by read-modify-write: S: new = old | gate,
    // R: new = old & !gate (gate = RLO ANDed with the open MCR brackets)
    mlir::Value gate = gatedRlo();
    if (!gate)
      return false;
    mlir::Value old = builder.create<awl::LoadOp>(loc(), v);
    mlir::Value newVal = set
        ? (mlir::Value)builder.create<awl::OrOp>(loc(), i1, old, gate)
        : (mlir::Value)builder.create<awl::AndOp>(
              loc(), i1, old,
              builder.create<awl::NotOp>(loc(), i1, gate).getResult());
    builder.create<awl::AssignOp>(loc(), newVal, v);
    rlo = builder.create<awl::ConstOp>(
        loc(), i1, builder.getIntegerAttr(i1, set ? 1 : 0));
    rloActive = true;
    return true;
  }

  // SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA [n]: shift/rotate ACCU1 by n
  // (immediate, or ACCU1 if no immediate was given: `L val`, `L n`, `SLD`)
  bool parseShift(const std::string &m) {
    padAcc();
    mlir::Value v = popAcc();
    bool throughCarry = m == "RLDA" || m == "RRDA";
    auto i32 = builder.getI32Type();
    auto i1 = builder.getI1Type();
    mlir::Value amount;
    if (lexer.getCurToken() == Tok::integer) {
      amount = constInt(i32, (int64_t)lexer.getInteger());
      lexer.next();
    } else if (throughCarry) {
      amount = constInt(i32, 1); // rotate through carry by one
    } else {
      amount = popAcc(); // no immediate: shift count is ACCU2's low byte
      amount = builder.create<awl::AndOp>(loc(), i32, castTo(amount, i32),
                                          constInt(i32, 0xFF)).getResult();
    }
    v = castTo(v, i32);
    std::string kind = m;
    for (auto &c : kind)
      c = (char)tolower((unsigned char)c);
    mlir::Value shifted;
    mlir::Value pos = builder.create<awl::CmpGtOp>(loc(), i1, amount, constInt(i32, 0));
    if (throughCarry) {
      // awlsim insn_rlda/insn_rrda: rotate ACCU1 by exactly 1 through the
      // A1 status bit (STW bit 7): RLDA fills the LSB with old A1, RRDA
      // fills the MSB with old A1. (The awl::ShiftOp rlda/rrda lowering only
      // approximates this with carry 0.)
      auto oldA1op = builder.create<awl::StwBitOp>(loc(), i1,
                                                   builder.getI32IntegerAttr(7));
      auto carryOp = builder.create<awl::CastOp>(loc(), i32, mlir::ValueRange{oldA1op});
      carryOp->setAttr("zext", builder.getUnitAttr());
      mlir::Value carry = carryOp.getResult();
      if (m == "RLDA")
        shifted = builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::ShiftOp>(
                loc(), i32, builder.getStringAttr("sld"),
                builder.create<awl::AndOp>(loc(), i32, v, constInt(i32, 0x7FFFFFFF))
                    .getResult(),
                constInt(i32, 1)).getResult(),
            carry).getResult();
      else
        shifted = builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("srd"),
                                         v, constInt(i32, 1)).getResult(),
            builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("sld"),
                                         carry, constInt(i32, 31)).getResult()).getResult();
    } else {
      shifted = builder.create<awl::ShiftOp>(
          loc(), i32, builder.getStringAttr(kind), v, amount);
    }
    // awlsim: count <= 0 leaves ACCU1 and STW unchanged
    mlir::Value r = throughCarry ? shifted : mux(pos, shifted, v);
    pushAcc(r);
    // awlsim: A1 = the bit shifted out of the ORIGINAL value — bit
    // (width-count) for left shifts/rotates (SLW/SLD/RLD/RLDA), bit
    // (count-1) for right shifts/rotates (SRW/SSI/SRD/SSD/RRD/RRDA);
    // A0=0, OV=0. count is clamped to the operand width.
    bool word = kind == "slw" || kind == "srw" || kind == "ssi";
    int width = word ? 16 : 32;
    bool left = kind == "slw" || kind == "sld" || kind == "rld" || kind == "rlda";
    mlir::Value clamped = mux(
        builder.create<awl::CmpGtOp>(loc(), i1, amount, constInt(i32, width)).getResult(),
        constInt(i32, width), amount);
    mlir::Value idx = left
        ? builder.create<awl::SubOp>(loc(), i32, constInt(i32, width), clamped).getResult()
        : builder.create<awl::SubOp>(loc(), i32, clamped, constInt(i32, 1)).getResult();
    mlir::Value bit = builder.create<awl::AndOp>(
        loc(), i32,
        builder.create<awl::ShiftOp>(loc(), i32, builder.getStringAttr("srd"),
                                     v, idx).getResult(),
        constInt(i32, 1)).getResult();
    mlir::Value a1 = builder.create<awl::CmpNeOp>(loc(), i1, bit, constInt(i32, 0));
    mlir::Value oldA1 = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(7));
    mlir::Value oldA0 = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(6));
    mlir::Value oldOV = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(5));
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(7),
                                     muxI1(pos, a1, oldA1));
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(6),
                                     muxI1(pos, i1Const(false), oldA0));
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(5),
                                     muxI1(pos, i1Const(false), oldOV));
    rloActive = false;
    return true;
  }

  // ACCU value -> f32 for real ops: an integer accumulator is reinterpreted
  // as raw bits (awlsim getPyFloat reads the dword), not value-converted
  mlir::Value accToReal(mlir::Value v) {
    if (mlir::isa<mlir::FloatType>(v.getType()))
      return castTo(v, builder.getF32Type());
    return bitcastTo(castTo(v, builder.getI32Type()), builder.getF32Type());
  }

  // ABS/NEGR/SIN/COS/TAN/ASIN/ACOS/ATAN/SQRT/EXP/LN: unary real math on ACCU1
  bool parseRealMath(const std::string &m) {
    mlir::Value a = popAcc();
    if (!a)
      return false;
    a = accToReal(a);
    std::string kind = m;
    for (auto &c : kind)
      c = (char)tolower((unsigned char)c);
    pushAcc(builder.create<awl::RealMathOp>(
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
    // symbolic/indirect operands: coerce to a valid timer index (a dynamic
    // i64 T pointer from a TIMER-typed variable is passed through)
    if (!t.getType().isInteger(64)) {
      if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(t.getType())) {
        if (dt.getArea() != mlir::scl::DirectArea::Timer)
          t = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Timer,
                                            0, 0, 0, 16);
      } else {
        t = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Timer,
                                          0, 0, 0, 16);
      }
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
    // symbolic/indirect operands: coerce to a valid counter index (a dynamic
    // i64 T/C pointer from a COUNTER-typed variable is passed through)
    if (!c.getType().isInteger(64)) {
      if (auto dt = mlir::dyn_cast<mlir::scl::DirectType>(c.getType())) {
        if (dt.getArea() != mlir::scl::DirectArea::Counter)
          c = builder.create<awl::DirectOp>(
              loc(), mlir::scl::DirectArea::Counter, 0, 0, 0, 16);
      } else {
        c = builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Counter,
                                          0, 0, 0, 16);
      }
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
    mlir::Value b = popAcc(); // ACCU1 (top)
    mlir::Value a = popAcc(); // ACCU2
    if (!a || !b)
      return false;
    if (w == 'R') {
      a = accToReal(a);
      b = accToReal(b);
      mlir::Value r;
      switch (k) {
      case ArithKind::Add: r = builder.create<awl::AddOp>(loc(), ty, a, b); break;
      case ArithKind::Sub: r = builder.create<awl::SubOp>(loc(), ty, a, b); break;
      case ArithKind::Mul: r = builder.create<awl::MulOp>(loc(), ty, a, b); break;
      case ArithKind::Div: r = builder.create<awl::DivOp>(loc(), ty, a, b); break;
      }
      pushAcc(r);
      // the float arith lowering routes through scl_rt_arith_f, which sets
      // the STW itself (awlsim setForFloatingPoint + *R OV/OS override)
      rloActive = false;
      return true;
    }
    // integer arith: awlsim computes word ops on the sign-extended words and
    // stores back with setWord (+I/-I /I: low word, high word of ACCU1 kept)
    // or setDWord (*I: full 32-bit product replaces ACCU1). STW A0/A1 come
    // from the UNtruncated result (computed at i64), OV/OS when it leaves
    // the 16/32-bit range.
    auto i16 = builder.getI16Type();
    auto i32 = builder.getI32Type();
    auto i64 = builder.getI64Type();
    mlir::Value as = w == 'D' ? castTo(a, i32) : castTo(castTo(a, i16), i32);
    mlir::Value bs = w == 'D' ? castTo(b, i32) : castTo(castTo(b, i16), i32);
    mlir::Value b32 = castTo(b, i32);
    mlir::Value as64 = castTo(as, i64), bs64 = castTo(bs, i64);
    mlir::Value wide; // untruncated result at i64 (also avoids divsi UB)
    switch (k) {
    case ArithKind::Add: wide = builder.create<awl::AddOp>(loc(), i64, as64, bs64); break;
    case ArithKind::Sub: wide = builder.create<awl::SubOp>(loc(), i64, as64, bs64); break;
    case ArithKind::Mul: wide = builder.create<awl::MulOp>(loc(), i64, as64, bs64); break;
    case ArithKind::Div: wide = builder.create<awl::DivOp>(loc(), i64, as64, bs64); break;
    }
    mlir::Value r32 = castTo(wide, i32); // truncated to 32 bits for ACCU1
    mlir::Value r; // value pushed back to ACCU1
    if (k == ArithKind::Div) {
      // awlsim insn_di_i/insn_di_d: division by zero leaves ACCU1 unchanged
      // (only STW is set). The quotient is computed at i64, so INT_MIN / -1
      // yields +2^31 there (stored wrapped as 0x80000000, flagged via OV).
      auto i1 = builder.getI1Type();
      mlir::Value isZero = builder.create<awl::CmpEqOp>(loc(), i1, bs, constInt(i32, 0));
      if (w == 'D') {
        r = mux(isZero, b32, r32); // setDWord(quo)
      } else {
        // /I: setDWord with the quotient in the low word and the remainder
        // in the high word (C truncation: rem = dividend - quo * divisor)
        mlir::Value rem = builder.create<awl::SubOp>(
            loc(), i32, as,
            builder.create<awl::MulOp>(loc(), i32, r32, bs).getResult()).getResult();
        mlir::Value packed = builder.create<awl::OrOp>(
            loc(), i32,
            builder.create<awl::ShiftOp>(
                loc(), i32, builder.getStringAttr("sld"),
                builder.create<awl::AndOp>(loc(), i32, rem,
                                           constInt(i32, 0xFFFF)).getResult(),
                constInt(i32, 16)).getResult(),
            builder.create<awl::AndOp>(loc(), i32, r32,
                                       constInt(i32, 0xFFFF)).getResult()).getResult();
        r = mux(isZero, b32, packed);
      }
    } else if (w == 'D') {
      r = r32; // setDWord
    } else if (k == ArithKind::Mul) {
      r = r32; // *I: setDWord (full product)
    } else {
      // setWord: low word of the result, high word of ACCU1 preserved
      r = builder.create<awl::OrOp>(
          loc(), i32,
          builder.create<awl::AndOp>(loc(), i32, b32,
                                     constInt(i32, (int64_t)(int32_t)0xFFFF0000)).getResult(),
          builder.create<awl::AndOp>(loc(), i32, r32,
                                     constInt(i32, 0xFFFF)).getResult()).getResult();
    }
    pushAcc(r);
    // STW: A0/A1 from the stored result (16-bit truncated for word ops,
    // full product for *I), OV/OS when the untruncated result leaves the
    // 16-bit (word) / 32-bit (dword) range; div-by-zero -> A0=A1=OV=OS=1
    if (k == ArithKind::Div) {
      auto div = builder.create<awl::StwDivOp>(loc(), bs, wide);
      div.setRangeAttr(builder.getI32IntegerAttr(w == 'D' ? 0x7FFFFFFF : 0x7FFF));
    } else {
      int op = k == ArithKind::Add ? 0 : k == ArithKind::Sub ? 1 : 2;
      int32_t range = w == 'D' ? 0x7FFFFFFF : 0x7FFF;
      // awlsim +D/-D classify the TRUNCATED stored dword for A0/A1, *D/*I
      // the untruncated product; word ops the truncated word. OV/OS use the
      // untruncated wide value (lhs/rhs are sign-extended to i64 in the
      // runtime, which recomputes it).
      mlir::Value stwVal = k == ArithKind::Mul
                               ? wide
                               : w == 'D' ? r32 : castTo(r32, i16);
      builder.create<awl::StwSignOp>(loc(), stwVal, as64, bs64,
                                      builder.getI32IntegerAttr(op),
                                      builder.getI32IntegerAttr(range));
    }
    rloActive = false;
    return true;
  }

  bool parseWordLogic(const std::string &m) {
    // AW/OW/XOW/UW operate on words (i16), AD/OD/XOD/UD on double words
    // (i32), AB/OB/UB on bytes (i8); U (Und) = AND, O (Oder) = OR, X = XOR.
    // Word/byte ops modify only the low word/byte of ACCU1 — the rest of
    // ACCU1 is preserved (awlsim setWord/setByte) — and set STW A1 = result
    // != 0, A0 = 0, OV = 0.
    char w = m.back();
    auto ty = w == 'D' ? builder.getI32Type()
             : w == 'B' ? builder.getI8Type()
                        : builder.getI16Type();
    char op = m[0] == 'U' ? 'A' : m[0];
    mlir::Value A, B; // A = ACCU1 (top of stack), B = operand
    if (lexer.getCurToken() == Tok::integer || lexer.getCurToken() == Tok::real) {
      // immediate operand: ACCU1 op imm
      B = parseConst();
      if (acc.empty())
        pushAcc(constInt(builder.getI32Type(), 0));
      A = popAcc();
    } else {
      padAcc();
      A = popAcc();
      B = popAcc();
    }
    if (!A || !B)
      return false;
    auto i32 = builder.getI32Type();
    auto logicOp = [&](mlir::Type t, mlir::Value x, mlir::Value y) {
      if (op == 'A')
        return (mlir::Value)builder.create<awl::AndOp>(loc(), t, x, y);
      if (op == 'O')
        return (mlir::Value)builder.create<awl::OrOp>(loc(), t, x, y);
      return (mlir::Value)builder.create<awl::XOrOp>(loc(), t, x, y);
    };
    mlir::Value r;
    if (w == 'D') {
      r = logicOp(i32, castTo(A, i32), castTo(B, i32));
    } else {
      int64_t mask = w == 'B' ? 0xFF : 0xFFFF;
      mlir::Value A32 = castTo(A, i32), B32 = castTo(B, i32);
      mlir::Value alo = builder.create<awl::AndOp>(loc(), i32, A32,
                                                   constInt(i32, mask)).getResult();
      mlir::Value blo = builder.create<awl::AndOp>(loc(), i32, B32,
                                                   constInt(i32, mask)).getResult();
      mlir::Value rlo = logicOp(ty, castTo(alo, ty), castTo(blo, ty));
      mlir::Value rhi = builder.create<awl::AndOp>(
          loc(), i32, A32,
          constInt(i32, (int64_t)(int32_t)~(uint32_t)mask)).getResult();
      auto z = builder.create<awl::CastOp>(loc(), i32, rlo);
      z.setZextAttr(builder.getUnitAttr());
      r = builder.create<awl::OrOp>(loc(), i32, rhi, z.getResult()).getResult();
    }
    pushAcc(r);
    // STW: A1 (bit 7) = result != 0, A0 (bit 6) = 0, OV (bit 5) = 0
    auto i1 = builder.getI1Type();
    auto cmpTy = w == 'D' ? i32 : ty;
    mlir::Value nz = builder.create<awl::CmpNeOp>(
        loc(), i1, castTo(r, cmpTy), castTo(constInt(ty, 0), cmpTy));
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(7), nz);
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(6),
                                     builder.create<awl::ConstOp>(
                                         loc(), i1, builder.getIntegerAttr(i1, 0)));
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(5),
                                     builder.create<awl::ConstOp>(
                                         loc(), i1, builder.getIntegerAttr(i1, 0)));
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
        pushAcc(constInt(builder.getI32Type(), 0));
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
    // record the comparison result into the STW A0/A1 bits (and VKE)
    int pred = cmp == Tok::eq ? 0
               : cmp == Tok::ne ? 1
               : cmp == Tok::lt ? 2
               : cmp == Tok::le ? 3
               : cmp == Tok::gt ? 4
                               : 5;
    if (w == 'R') {
      // awlsim: NaN operand -> unordered: VKE=0, STA=0, A0=A1=OV=OS=1
      // (the STW bits are set by stw_cmp_f); the comparison result only
      // counts when both operands are ordered
      mlir::Value nanA = builder.create<awl::CmpNeOp>(loc(), i1, a, a); // NaN != NaN
      mlir::Value nanB = builder.create<awl::CmpNeOp>(loc(), i1, b, b);
      mlir::Value ordered = builder.create<awl::NotOp>(
          loc(), i1,
          builder.create<awl::OrOp>(loc(), i1, nanA, nanB).getResult()).getResult();
      r = builder.create<awl::AndOp>(loc(), i1, r, ordered).getResult();
    }
    builder.create<awl::StwCmpOp>(loc(), builder.getI32IntegerAttr(pred), a, b);
    rlo = r;
    stmtLogic = true;
    rloActive = true;
    return true;
  }

  bool parseJump(const std::string &m) {
    if (lexer.getCurToken() != Tok::ident)
      return error("expected label after " + m);
    std::string label = upperIdent();
    lexer.next();
    auto i1 = builder.getI1Type();
    syncAcc(); // the jump target reloads the ACCUs from the runtime: sync first
    if (m == "SPA" || m == "JU") {
      builder.create<awl::JumpOp>(loc(), builder.getStringAttr(label));
      rloActive = false;
      return true;
    }
    if (!ensureRlo())
      return false;
    // SPBB/SPBNB save the RLO into the BR/BIE bit before the jump test
    if (m == "SPBB" || m == "SPBNB")
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(8), rlo);
    mlir::Value cond = rlo;
    bool spnTest = m == "SPN"; // awlsim insn_spn: jump if A1^A0, no state change
    bool brTest = m == "SPBI" || m == "SPBIN" || m == "JBI" || m == "JNBI";
    if (brTest) // SPBI/SPBIN test the BR bit, not the RLO
      cond = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(8));
    if (spnTest)
      cond = builder.create<awl::XOrOp>(
          loc(), i1,
          builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(7)).getResult(),
          builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(6)).getResult()).getResult();
    bool stwTest = m == "SPO" || m == "SOS" || m == "SPU" || spnTest;
    if (m == "SPO" || m == "SOS" || m == "SPU") {
      if (m == "SPO")
        cond = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(5)); // OV
      else if (m == "SOS")
        cond = builder.create<awl::StwBitOp>(loc(), i1, builder.getI32IntegerAttr(4)); // OS
      else
        cond = builder.create<awl::StwCondOp>(loc(), i1, builder.getI32IntegerAttr(6)); // UO = A0&A1
    }
    bool jumpIfNot = m == "SPBN" || m == "SPBNB" || m == "SPBIN" ||
                     m == "JNBI" || m == "JNB" || m == "JCN";
    if (!stwTest) {
      // awlsim: RLO-tested jumps (SPB/SPN/SPBB/SPBNB...) end with OR=0,
      // STA=1, VKE=1, NER=0; BR-tested jumps (SPBI/SPBIN...) keep VKE,
      // set OR=0, STA=1, NER=0. Emit the STW mirror BEFORE the jump op so
      // the target label observes the post-jump state (program order).
      stOr = i1Const(false);
      stSta = i1Const(true);
      stNer = i1Const(false);
      if (!brTest)
        rlo = i1Const(true);
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(1), rlo);
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(0), stNer);
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(2), stSta);
      builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(3), stOr);
    }
    if (jumpIfNot)
      builder.create<awl::JumpIfNotOp>(loc(), cond, builder.getStringAttr(label));
    else
      builder.create<awl::JumpIfOp>(loc(), cond, builder.getStringAttr(label));
    if (stwTest) {
      // awlsim insn_spo/insn_sos/insn_spu: STW (incl. RLO) untouched
      rloActive = false;
      return true;
    }
    stmtLogic = true;
    rloActive = true;
    return true;
  }

  // __ASSERT<cmp>[R] <lhs>, <rhs> — awlsim test oracle. lhs is __ACCU n, a
  // memory operand (loaded), or a status-word bit (__STW <bit|n>, UO, or a
  // bare bit name), compared as 0/1. Lowered to awl.assert / awl.assert_r.
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
    if (lexer.getCurToken() == Tok::eq || lexer.getCurToken() == Tok::ne ||
        lexer.getCurToken() == Tok::lt || lexer.getCurToken() == Tok::le ||
        lexer.getCurToken() == Tok::gt || lexer.getCurToken() == Tok::ge) {
      // lhs is a status condition: __ASSERT== ==0, 1 / __ASSERT== >0, 0 ...
      Tok ct = lexer.getCurToken();
      lexer.next();
      if (lexer.getCurToken() == Tok::integer)
        lexer.next();
      int cond = ct == Tok::eq ? 0
                 : ct == Tok::ne ? 1
                 : ct == Tok::gt ? 2
                 : ct == Tok::lt ? 3
                 : ct == Tok::ge ? 4
                                 : 5;
      auto i1 = builder.getI1Type();
      actual = builder.create<awl::StwCondOp>(loc(), i1,
                                              builder.getI32IntegerAttr(cond));
    } else if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperIdent();
      if (w == "__DBR") {
        lexer.nextLine(); // DB-register value not tracked: skip
        return true;
      }
      // status-word operand as assert lhs: __STW <bit|n>, UO, bare bit name
      {
        int stwBit = -1;
        bool isUo = false;
        if (w == "__STW") {
          lexer.next(); // consume __STW itself (upperIdent does not)
          if (lexer.getCurToken() == Tok::ident) {
            stwBit = stwBitNumber(upperStr(lexer.getIdent()));
            lexer.next();
          } else if (lexer.getCurToken() == Tok::integer) {
            stwBit = (int)lexer.getInteger();
            lexer.next();
          }
        } else if (w == "UO") {
          isUo = true;
        } else {
          stwBit = stwBitNumber(w);
        }
        if (stwBit >= 0 || isUo) {
          auto i1 = builder.getI1Type();
          actual = isUo
                       ? (mlir::Value)builder.create<awl::StwCondOp>(
                             loc(), i1, builder.getI32IntegerAttr(6))
                       : (mlir::Value)builder.create<awl::StwBitOp>(
                             loc(), i1, builder.getI32IntegerAttr(stwBit));
          goto haveLhs;
        }
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
                     ? zextAcc(builder.create<awl::LoadOp>(loc(), addr))
                     : addr;
      }
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
    haveLhs:;
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
      std::string rw = upperStr(lexer.getIdent());
      if (rw == "__DBR" || rw == "__AR") {
        lexer.nextLine(); // register values not tracked: skip the assert
        return true;
      }
      // status-word operand as assert rhs: __STW <bit|n>, UO, bare bit name
      int stwBit = -1;
      bool isUo = false;
      if (rw == "__STW") {
        lexer.next(); // consume __STW itself
        if (lexer.getCurToken() == Tok::ident) {
          stwBit = stwBitNumber(upperStr(lexer.getIdent()));
          lexer.next();
        } else if (lexer.getCurToken() == Tok::integer) {
          stwBit = (int)lexer.getInteger();
          lexer.next();
        }
      } else if (rw == "UO") {
        isUo = true;
      } else {
        stwBit = stwBitNumber(rw);
      }
      if (stwBit >= 0 || isUo) {
        auto i1 = builder.getI1Type();
        expected = isUo
                       ? (mlir::Value)builder.create<awl::StwCondOp>(
                             loc(), i1, builder.getI32IntegerAttr(6))
                       : (mlir::Value)builder.create<awl::StwBitOp>(
                             loc(), i1, builder.getI32IntegerAttr(stwBit));
      } else {
        // memory operand rhs, e.g. __ASSERT== __ACCU 1, MD 8
        mlir::Value addr = parseOperand(false);
        if (!addr) {
          lexer.nextLine();
          return true;
        }
        expected = mlir::isa<mlir::scl::DirectType>(addr.getType())
                       ? zextAcc(builder.create<awl::LoadOp>(loc(), addr))
                       : addr;
      }
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
    // awlsim __ASSERT==R compares as real. Plain __ASSERT== compares the
    // raw 32-bit patterns (AwlMemoryObject_asScalar of a real operand is
    // its float bit pattern), so a real accumulator against DW#16#40000000
    // must compare 0x40000000, not 2.0f vs 1073741824.0.
    bool actualF = mlir::isa<mlir::FloatType>(actual.getType());
    bool expectedF = mlir::isa<mlir::FloatType>(expected.getType());
    if (isReal) {
      actual = castTo(actual, builder.getF64Type());
      expected = castTo(expected, builder.getF64Type());
      builder.create<awl::AssertROp>(loc(), idAttr, opAttr, actual, expected);
    } else if (actualF || expectedF) {
      if (actualF)
        actual = bitcastTo(castTo(actual, builder.getF32Type()),
                           builder.getI32Type());
      if (expectedF)
        expected = bitcastTo(castTo(expected, builder.getF32Type()),
                             builder.getI32Type());
      actual = castTo(actual, builder.getI64Type());
      expected = castTo(expected, builder.getI64Type());
      builder.create<awl::AssertOp>(loc(), idAttr, opAttr, actual, expected);
    } else {
      actual = castTo(actual, builder.getI64Type());
      expected = castTo(expected, builder.getI64Type());
      builder.create<awl::AssertOp>(loc(), idAttr, opAttr, actual, expected);
    }
    // awlsim insn_assert_*: after the comparison NER := 0 (only NER — the
    // RLO/VKE/OR/STA are untouched), which the next logic statement sees
    builder.create<awl::StwSetBitOp>(loc(), builder.getI32IntegerAttr(0),
                                      i1Const(false)); // NER := 0
    stNer = i1Const(false);
    return true;
  }

  // RHS of a CALL parameter assignment: constant, block reference
  // (FC 10 as a BLOCK_FC value), boolean, or a plain operand (LW 16, ...).
  // For an operand RHS, *targetOut (if given) receives the operand address:
  // parameters are passed by reference, so an operand RHS is also written
  // back after the call.
// Resolve a symbolic DB member access (SYMBOL[.SUB][[i]]..., the lexer is
  // positioned at the first member name `dw`, already consumed) through the
  // DATA_BLOCK's layout. Fills byte/bit/width; false on error.
  // Consume a P#... literal token (Tok::integer holding the packed S7
  // pointer value) and return it, resolving a symbolic member part
  // (P#DB1.STR1) to its byte/bit offset via the DB layout.
  int64_t consumePtrLiteral() {
    int64_t v = lexer.getInteger();
    llvm::StringRef sym = lexer.curPtrSym();
    lexer.next();
    if (sym.empty())
      return v;
    int64_t area = (v >> 24) & 0xFF;
    int64_t db = (v >> 16) & 0xFF;
    // P#"Data".VAR: quoted DB name — resolve to its number (area DB)
    if (sym.size() >= 2 && sym.front() == '"') {
      size_t close = sym.find('"', 1);
      if (close != llvm::StringRef::npos) {
        std::string name = sym.substr(1, close - 1).str();
        for (auto &c : name)
          c = (char)toupper((unsigned char)c);
        auto cli = gCliNamedDbs.find(name);
        auto nit = cli != gCliNamedDbs.end() ? cli : namedDbs.find(name);
        bool found = cli != gCliNamedDbs.end() || nit != namedDbs.end();
        if (found) {
          area = 0x84;
          db = nit->second;
          sym = sym.substr(close + 1); // ".DTVAR1"
          v = (int64_t)(((uint64_t)area << 24) | ((uint64_t)db << 16) |
                        ((uint64_t)v & 0xFFFF));
        }
      }
    }
    if (area != 0x84 && area != 0x85)
      return v; // only DB-area pointers have symbolic members
    // ".DBX"/".DBB"/... are access-width keywords, not symbolic members:
    // the literal's own byte.bit is the address (P#DB50.DBX 10.0)
    if (sym.size() >= 4 && sym[0] == '.') {
      std::string kw;
      for (char c : sym.substr(1, 3))
        kw += (char)toupper((unsigned char)c);
      if (kw == "DBX" || kw == "DBB" || kw == "DBW" || kw == "DBD" ||
          kw == "DIX" || kw == "DIB" || kw == "DIW" || kw == "DID")
        return v;
    }
    std::string text = "DB" + std::to_string(db) + std::string(sym);
    Lexer sub(text, "<ptrsym>");
    sub.start();
    Lexer savedLexer = lexer;
    lexer = sub;
    std::string w0 = upperIdent(); // DB<n>
    int32_t sdb = w0.size() > 2 ? atoi(w0.c_str() + 2) : 0;
    lexer.next();                  // '.'
    lexer.next();                  // first member ident
    std::string dw = upperStr(lexer.getIdent());
    lexer.next();
    int32_t byte = 0, bit = 0, width = 16;
    std::string mtype;
    bool ok = resolveDbMember(sdb, dw, false, byte, bit, width, &mtype);
    lexer = savedLexer;
    if (!ok)
      return v;
    if (!mtype.empty())
      lastOperandType = mtype; // e.g. INT for the ANY type code
    return (v & ~((int64_t)0x1FFF << 3)) | ((byte & 0x1FFF) << 3) |
           (bit & 7);
  }

  bool resolveDbMember(int32_t db, const std::string &dw, bool bitCtxt,
                       int32_t &byte, int32_t &bit, int32_t &width,
                       std::string *typeOut = nullptr) {
    // SPEC[AWL] Siemens AWL manual symbolic DB members (DBn.SYMBOL...) resolved through the DATA_BLOCK layout (approx: a member missing from a declared DB resolves to DBW 0/DBX 0.0 of that DB)
    byte = 0;
    bit = 0;
    width = bitCtxt ? 1 : 16;
    auto lit = dbLayouts.find(effDb(db));
    const Layout *lay = lit != dbLayouts.end() ? &lit->second : nullptr;
    const MemberInfo *mi = nullptr;
    if (lay) {
      auto it0 = lay->members.find(dw);
      if (it0 != lay->members.end()) {
        byte = it0->second.byte;
        bit = it0->second.bit;
        width = it0->second.widthBits;
        mi = &it0->second;
        if (typeOut) *typeOut = mi->type;
      }
    } else {
      // no DATA_BLOCK declaration: auto-assign the symbolic member in
      // order of first use, sized by the access width
      // SPEC[AWL] Siemens AWL manual symbolic DB members of undeclared DBs, auto-assigned offsets in order of first use (awlsim symbol tables) -> autoDbCursor branch of resolveDbMember (approx: DB contents are runtime memory, so the synthetic offset only fixes the address, not the value)
      int32_t key = effDb(db);
      int32_t w = bitCtxt ? 1 : 16;
      int32_t stride = w >= 16 ? 2 : 1;
      auto &cursor = autoDbCursor[key];
      if (cursor % stride)
        cursor += stride - cursor % stride;
      auto skey = std::make_pair(key, dw);
      auto ait = autoDbSyms.find(skey);
      if (ait == autoDbSyms.end())
        ait = autoDbSyms.emplace(skey, cursor).first;
      byte = ait->second;
      width = w;
      cursor = std::max(cursor, byte + stride);
      auto &hw = dbRefBytes[key];
      if (cursor > hw)
        hw = cursor;
    }
    bool stringIndexed = false; // STRING[i] set an explicit char width
    while (true) {
      // [i] index?
      if (lexer.getCurToken() == Tok::lbracket) {
        if (mi && mi->type.rfind("STRING", 0) == 0) {
          // STRING[i]: char access — chars start at byte 2, one byte each
          lexer.next();
          int32_t idx = 1;
          if (lexer.getCurToken() == Tok::integer) {
            idx = (int32_t)lexer.getInteger();
            lexer.next();
          }
          while (lexer.getCurToken() != Tok::rbracket &&
                 lexer.getCurToken() != Tok::eof)
            lexer.next();
          if (lexer.getCurToken() == Tok::rbracket)
            lexer.next();
          byte += 2 + (idx - 1);
          bit = 0;
          width = 8;
          stringIndexed = true;
        } else {
          addBits(byte, bit, consumeArrayIndices(mi));
        }
      }
      if (lexer.getCurToken() == Tok::dot) {
        lexer.next();
        if (lexer.getCurToken() == Tok::ident ||
            lexer.getCurToken() == Tok::localvar) {
          std::string sub = upperStr(lexer.getIdent());
          lexer.next();
          if (lay) {
            const Layout *next = lay;
            if (mi && mi->nested)
              next = mi->nested.get();
            else if (mi && mi->type == "ARRAY" && mi->elemNested)
              next = mi->elemNested.get();
            else if (mi && mi->type != "ARRAY") {
              auto uit = udtLayouts.find(mi->type);
              if (uit != udtLayouts.end())
                next = &uit->second;
              else
                next = nullptr;
            } else if (mi && mi->type == "ARRAY") {
              auto uit = udtLayouts.find(mi->elemType);
              next = uit != udtLayouts.end() ? &uit->second : nullptr;
            }
            if (next) {
              auto it2 = next->members.find(sub);
              if (it2 != next->members.end()) {
                byte += it2->second.byte;
                bit = it2->second.bit;
                width = it2->second.widthBits;
                mi = &it2->second;
                lay = next;
                if (typeOut) *typeOut = mi->type;
                continue;
              }
            }
          }
          // unknown sub-member: stop resolving, keep current offset
          continue;
        }
        continue;
      }
      break;
    }
    if (mi && !stringIndexed) {
      width = mi->widthBits ? mi->widthBits : width;
      if (width == 1)
        bit = mi->bit;
      if (typeOut && !stringIndexed)
        *typeOut = mi->type;
      if (mi->type == "ARRAY") {
        lastPtrCount = mi->count > 0 ? mi->count : 1;
        lastPtrElemType = mi->elemType;
        lastPtrElemBits = mi->elemSize ? mi->elemSize * 8
                                       : (mi->widthBits ? mi->widthBits : 16);
      }
    }
    if (width != 1 && width != 8 && width != 16 && width != 32 &&
        width != 64)
      width = 32; // UDT/STRING whole-access: dword granularity
    return true;
  }

  // Parse a symbolic DB member operand (DBn.SYMBOL...): the lexer is
  // positioned at the first member name `dw` (already consumed).
  mlir::Value parseDbMemberSymbolic(int32_t db, const std::string &dw,
                                    bool bitCtxt) {
    int32_t byte = 0, bit = 0, width = bitCtxt ? 1 : 16;
    std::string mtype;
    if (!resolveDbMember(db, dw, bitCtxt, byte, bit, width, &mtype))
      return nullptr;
    if (!mtype.empty())
      lastOperandType = mtype;
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      width = 32; // UDT/STRING whole-access: dword granularity
    return builder.create<awl::DirectOp>(
        loc(), mlir::scl::DirectArea::DataBlock, effDb(db), byte, bit, width);
  }

  mlir::Value parseCallParamValue(mlir::Value *targetOut = nullptr) {
    if (lexer.getCurToken() == Tok::integer ||
        lexer.getCurToken() == Tok::real || lexer.getCurToken() == Tok::minus)
      return parseConst();
    if (lexer.getCurToken() == Tok::ident) {
      std::string w = upperStr(lexer.getIdent());
      if (w == "FC" || w == "FB" || w == "SFC" || w == "SFB" || w == "DB" ||
          w == "DI") {
        // block/DB reference value: FC 10, FB 2, DB 1, DI 3 -> the number
        lexer.next();
        if (lexer.getCurToken() == Tok::lbracket) {
          // dynamic: DB [MW 14] / FC [MW 12] — value from a memory cell
          lexer.next();
          mlir::Value addr = parseOperand(false);
          if (!addr)
            return nullptr;
          while (lexer.getCurToken() != Tok::rbracket &&
                 lexer.getCurToken() != Tok::eof)
            lexer.next();
          if (lexer.getCurToken() == Tok::rbracket)
            lexer.next();
          mlir::Value v = mlir::isa<mlir::scl::DirectType>(addr.getType())
                              ? (mlir::Value)builder.create<awl::LoadOp>(loc(), addr)
                              : addr;
          if (w == "DB" || w == "DI")
            // DB pointer parameter: S7 pointer with the dynamic DB number
            return builder.create<awl::PackPtrOp>(
                loc(), builder.getI64Type(), builder.getI32IntegerAttr(0x84),
                builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(0),
                castTo(v, builder.getI32Type()));
          return v; // block number from a variable
        }
        int64_t n = 0;
        if (lexer.getCurToken() == Tok::integer) {
          n = lexer.getInteger();
          lexer.next();
        }
        return constInt(builder.getI32Type(), n);
      }
      if (w == "TRUE" || w == "FALSE") {
        lexer.next();
        return constInt(builder.getI32Type(), w == "TRUE" ? 1 : 0);
      }
      if ((w == "T" || w == "C" || w == "Z") &&
          lexer.peekToken() == Tok::integer) {
        // T 42 / C 7 as a CALL param: the timer/counter NUMBER (block ref)
        lexer.next();
        int64_t n = lexer.getInteger();
        lexer.next();
        return constInt(builder.getI32Type(), n);
      }
    }
    mlir::Value addr = parseOperand(false);
    if (!addr)
      return nullptr;
    if (mlir::isa<mlir::scl::DirectType>(addr.getType())) {
      if (targetOut)
        *targetOut = addr;
      return builder.create<awl::LoadOp>(loc(), addr);
    }
    if (lastOperandIndirect)
      // indirect RHS (M [AR1, P#0.0], MD [#var], ...): load through the
      // address instead of passing the pointer as the value
      return builder.create<awl::LoadIndirectOp>(
          loc(), builder.getI64Type(), addr,
          builder.getI32IntegerAttr(lastIndirectWidth));
    return addr;
  }

  // MARK: runtime-staged calls (SFC n, IEC library FC called by name)
  // Parameters are staged in the marker area at kLocalBase (the same
  // staging the FC param copies use) and read/written by the runtime
  // (scl_rt_sfc / scl_rt_iec). ANY params stage the full 10-byte ANY
  // header inline.
  struct RtParam {
    const char *name;
    int32_t byte, width;
    bool out, any;
  };
  // SFC interfaces: staging offset from kLocalBase, width, direction.
  // ANY params (BVAL/BLK) stage the full 10-byte ANY header inline
  // (BVAL @ +0, BLK @ +10), RET_VAL (i16) @ +20.
  static constexpr RtParam sfc21Tab[] = {{"BVAL", 0, 80, false, true},
                                     {"BLK", 10, 80, false, true},
                                     {"RET_VAL", 20, 16, true, false}};
  static constexpr RtParam sfc47Tab[] = {{"WT", 0, 16, false, false}};
  static constexpr RtParam sfc64Tab[] = {{"RET_VAL", 0, 32, true, false}};
  // IEC library (awlsim library/iec): LEN 21, EQ_DT 9, GE_DT 12,
  // DELETE 4. STRING / DATE_AND_TIME params are passed as ANY pointers.
  static constexpr RtParam iecLenTab[] = {{"S", 0, 80, false, true},
                                      {"RET_VAL", 20, 16, true, false}};
  static constexpr RtParam iecDtTab[] = {{"DT1", 0, 80, false, true},
                                     {"DT2", 10, 80, false, true},
                                     {"RET_VAL", 20, 1, true, false}};
  static constexpr RtParam iecDeleteTab[] = {{"IN", 0, 80, false, true},
                                           {"L", 10, 16, false, false},
                                         {"P", 12, 16, false, false},
                                         // RET_VAL is an OUT STRING passed
                                         // as an ANY pointer that the
                                         // runtime writes through (staged
                                         // like an input)
                                         {"RET_VAL", 20, 80, false, true}};

  // Complex-typed parameters (STRING/ANY/DATE_AND_TIME/block refs) are
  // passed by reference: the caller stages a 10-byte ANY header at the
  // parameter slot and the callee accesses the actual operand through it
  // (P##param, like awlsim).
  static bool isByRefParam(const MemberInfo *m) {
    return isByRefType(m->type);
  }

  // Is the current CALL parameter RHS an addressable operand (memory
  // operand or #localvar)? Literals (numbers, DT#..., FC/DB refs, ...) are
  // passed as values, not pointers.
  bool callRhsAddressable() {
    if (lexer.getCurToken() == Tok::localvar)
      return true;
    if (lexer.getCurToken() != Tok::ident)
      return false;
    std::string w = upperStr(lexer.getIdent());
    static const char *kw[] = {"FC",  "FB",  "SFC", "SFB", "DB", "DI",
                               "TRUE", "FALSE", "D",  "T",   "Z",  "C",
                               "DT",  "DATE", "TIME", "TOD", "S5T",
                               "DATE_AND_TIME", "L",  "B",  "W"};
    for (const char *k : kw)
      if (w == k)
        return false;
    return true;
  }

  // Should this by-reference parameter stage a pointer? P#... literals are
  // wrapped into an ANY header only for ANY-typed params.
  bool byRefRhs(bool isAny) {
    if (lexer.getCurToken() == Tok::integer &&
        (!lexer.curPtrSym().empty() || lexer.curPtrTypeCode() >= 0))
      return isAny;
    return callRhsAddressable();
  }

  // awlsim ANYPointer.makeByAutoType: an ARRAY with a supported element
  // type uses (elem type code, elem count); anything else falls back to
  // makeByTypeWidth(totalBits): DWORD if %32, else WORD if %16, else BYTE
  // if %8, else BOOL.
  static int32_t anyTypeCodeOf(const std::string &ty) {
    static const std::pair<const char *, int32_t> tc[] = {
        {"BOOL", 0x01}, {"BYTE", 0x02}, {"CHAR", 0x03}, {"WORD", 0x04},
        {"INT", 0x05}, {"DWORD", 0x06}, {"DINT", 0x07}, {"REAL", 0x08},
        {"DATE", 0x09}, {"TOD", 0x0A}, {"TIME", 0x0B}, {"TIME_OF_DAY", 0x0A},
        {"S5TIME", 0x0C}, {"S5T", 0x0C}, {"DATE_AND_TIME", 0x0E},
        {"DT", 0x0E}, {"STRING", 0x13}, {"TIMER", 0x1D}, {"COUNTER", 0x1C},
        {"BLOCK_FB", 0x17}, {"BLOCK_FC", 0x18}, {"BLOCK_DB", 0x19},
        {"BLOCK_SDB", 0x1A}, {"POINTER", 0x13}, {"ANY", 0x10}};
    for (auto &t : tc)
      if (ty == t.first)
        return t.second;
    return 0;
  }
  static void anyAutoType(const std::string &ty, bool isArray, int32_t count,
                          const std::string &elemTy, int32_t elemBits,
                          int32_t &typeCode, int32_t &repCount) {
    if (isArray) {
      int32_t ec = anyTypeCodeOf(elemTy);
      if (ec && elemTy != "STRUCT") {
        typeCode = ec;
        repCount = count > 0 ? count : 1;
        return;
      }
      int32_t bits = count * (elemBits > 0 ? elemBits : 16);
      if (bits % 32 == 0) { typeCode = 0x06; repCount = bits / 32; }
      else if (bits % 16 == 0) { typeCode = 0x04; repCount = bits / 16; }
      else if (bits % 8 == 0) { typeCode = 0x02; repCount = bits / 8; }
      else { typeCode = 0x01; repCount = bits; }
      return;
    }
    typeCode = anyTypeCodeOf(ty);
    if (repCount <= 0)
      repCount = 1;
  }

  // Stage a 10-byte ANY header at (area, db, byte) for the operand at the
  // current lexer position. `#anyvar` copies the variable's own header; any
  // other operand is wrapped in a synthesized ANY header (awlsim converts
  // operand parameters to ANY pointers the same way).
  // toVl: staging for a CALL param (the header lands in the caller's L
  // frame, so L-area r-values are seen as VL/0x87 by the callee); false for
  // runtime-staged SFC/IEC calls (read in the CURRENT frame, L stays 0x86)
  bool emitAnyParamHeader(mlir::scl::DirectArea area, int32_t db,
                          int32_t byte, bool toVl = true) {
    // P#M 10.0 BYTE 7 ... literal: build the 10-byte ANY header directly
    // from the pointer value + type code + count (awlsim ANYPointerConst)
    if (lexer.getCurToken() == Tok::integer &&
        (!lexer.curPtrSym().empty() || lexer.curPtrTypeCode() >= 0)) {
      int typeCode = lexer.curPtrTypeCode() >= 0 ? lexer.curPtrTypeCode()
                                                 : 0x06; // DWORD default
      int count = lexer.curPtrCount() > 0 ? lexer.curPtrCount() : 1;
      lastOperandType = "";
      lastPtrCount = 0;
      lastPtrElemType = "";
      lastPtrElemBits = 0;
      int64_t p = consumePtrLiteral(); // resolves P#DBn.SYM byte offset
      if (lexer.curPtrCount() <= 0 && lastPtrCount > 0)
        count = lastPtrCount; // P#DBn.ARRAYVAR: element count from the layout
      // a symbolic member carries its declared type and (for arrays) the
      // element type/count for the ANY header (awlsim makeByAutoType:
      // INT -> 0x05, an ARRAY -> (elem type code, elem count))
      if (lexer.curPtrTypeCode() < 0 && !lastOperandType.empty())
        anyAutoType(lastOperandType, lastOperandType == "ARRAY", lastPtrCount,
                    upperStr(lastPtrElemType), lastPtrElemBits, typeCode,
                    count);
      auto put = [&](int32_t i, int32_t w, int32_t val) {
        builder.create<awl::StoreOp>(
            loc(), constInt(builder.getI32Type(), val),
            mkDirect(area, db, byte + i, 0, w));
      };
      put(0, 8, 0x10);
      put(1, 8, typeCode);
      put(2, 16, count);
      put(4, 16, (int32_t)((p >> 16) & 0xFF)); // DB number
      // S7 pointer dword in P#DBX style: db field 0 (awlsim ANYPointer)
      put(6, 32, (int32_t)(p & 0xFF00FFFFll));
      return true;
    }
    bool isAnyVar = false;
    const MemberInfo *opMember = nullptr; // for ARRAY repCount/elemType
    if (lexer.getCurToken() == Tok::localvar) {
      std::string vname = upperStr(lexer.getIdent());
      if (curLayout) {
        auto it = curLayout->members.find(vname);
        if (it != curLayout->members.end())
          opMember = &it->second;
      }
      if (!opMember && curTempLayout) {
        auto it = curTempLayout->members.find(vname);
        if (it != curTempLayout->members.end())
          opMember = &it->second;
      }
      if (opMember && opMember->type == "ANY")
        isAnyVar = true;
    }
    mlir::Value a = parseOperand(false);
    if (!a)
      return false;
    auto dt = mlir::dyn_cast<mlir::scl::DirectType>(a.getType());
    if (isAnyVar && dt) {
      for (int i = 0; i < 10; ++i) {
        auto src = builder.create<awl::DirectOp>(loc(), dt.getArea(),
                                                 dt.getDb(),
                                                 dt.getByte() + i, 0, 8);
        mlir::Value bv = builder.create<awl::LoadOp>(loc(), src);
        builder.create<awl::StoreOp>(
            loc(), bv, mkDirect(area, db, byte + i, 0, 8));
      }
      return true;
    }
    if (!dt)
      return true; // not an addressable operand: nothing to point at
    int32_t width = dt.getWidth();
    int32_t repCount = 1;
    // ANY type code from the operand's declared TYPE when known
    // (awlsim ANYPointer.makeByAutoType: INT=0x05 differs from WORD=0x04);
    // fall back to the access width otherwise
    int32_t typeCode = anyTypeCodeOf(lastOperandType);
    if (!typeCode)
      typeCode = width == 1   ? 0x01
                 : width == 8 ? 0x02
                 : width == 16 ? 0x04
                               : 0x06;
    if (opMember && opMember->type == "ARRAY") {
      // ANY of an ARRAY: element type code + element count, or the
      // width-derived fallback (awlsim makeByAutoType / makeByTypeWidth)
      int32_t eb = opMember->elemSize ? opMember->elemSize * 8
                                      : (opMember->widthBits ? opMember->widthBits
                                                             : 16);
      anyAutoType("ARRAY", true, opMember->count, upperStr(opMember->elemType),
                  eb, typeCode, repCount);
    }
    // L-stack r-values are forwarded to the callee as VL (0x87, the
    // caller's frame) — awlsim _FC_trans_dbpointerInVL maps L -> VL
    bool isLocal = dt.getArea() == mlir::scl::DirectArea::Local ||
                   dt.getArea() == mlir::scl::DirectArea::VLocal;
    int32_t areaS7 =
        dt.getArea() == mlir::scl::DirectArea::Input    ? 0x81
        : dt.getArea() == mlir::scl::DirectArea::Output ? 0x82
        : dt.getArea() == mlir::scl::DirectArea::Marker ? 0x83
        : (isLocal && toVl)                             ? 0x87
        : isLocal                                       ? 0x86
                                                        : 0x84;
    int32_t pdb = dt.getArea() == mlir::scl::DirectArea::DataBlock
                      ? dt.getDb()
                      : 0;
    bool isInstance = pdb == -1; // FB static: current DI at runtime
    if (isInstance)
      areaS7 = 0x84; // awlsim _trans_anyPointerFromDI: area DB + open DI nr
    // the S7 pointer dword uses P#DBX style for DB areas: db field 0 (the
    // DB number lives in the ANY header word @+4, awlsim ANYPointer);
    // VL (0x87) has no db field, so its byte offset uses bits 3..23
    uint32_t byteMask = (areaS7 == 0x87 && toVl) ? 0x1FFFFF : 0x1FFF;
    uint32_t ptr = ((uint32_t)areaS7 << 24) |
                   ((uint32_t)(areaS7 == 0x84 ? 0 : (pdb & 0xFF)) << 16) |
                   ((uint32_t)(dt.getByte() & byteMask) << 3) |
                   (uint32_t)(dt.getBit() & 7);
    auto put = [&](int32_t i, int32_t w, int32_t val) {
      builder.create<awl::StoreOp>(
          loc(), constInt(builder.getI32Type(), val),
          mkDirect(area, db, byte + i, 0, w));
    };
    auto putV = [&](int32_t i, int32_t w, mlir::Value val) {
      builder.create<awl::StoreOp>(loc(), val,
                                   mkDirect(area, db, byte + i, 0, w));
    };
    put(0, 8, 0x10);          // magic
    put(1, 8, typeCode);      // data type
    put(2, 16, repCount);     // repetition factor
    if (isInstance) {
      // DB number = the currently open instance DB (awlsim: diRegister)
      mlir::Value dino = builder.create<awl::DbInfoOp>(
          loc(), builder.getI64Type(), builder.getI32IntegerAttr(3),
          builder.getI32IntegerAttr(0));
      putV(4, 16, castTo(dino, builder.getI32Type()));
      // bake the AR2 multi-instance base into the pointer (awlsim
      // _trans_anyPointerFromDI adds the AR2 base for instance data)
      mlir::Value ar2 = builder.create<awl::ArGetOp>(
          loc(), builder.getI64Type(), builder.getI32IntegerAttr(2));
      mlir::Value baseBits = builder.create<awl::AndOp>(
          loc(), builder.getI64Type(), ar2,
          constInt(builder.getI64Type(), 0xFFFFFF));
      mlir::Value bits = builder.create<awl::AddOp>(
          loc(), builder.getI64Type(), baseBits,
          constInt(builder.getI64Type(), (int64_t)dt.getByte() * 8 + dt.getBit()));
      mlir::Value ptrV = builder.create<awl::OrOp>(
          loc(), builder.getI64Type(), bits,
          constInt(builder.getI64Type(), (int64_t)0x84 << 24));
      putV(6, 32, castTo(ptrV, builder.getI32Type()));
    } else {
      put(4, 16, pdb);        // DB number
      put(6, 32, (int32_t)ptr); // S7 pointer
    }
    return true;
  }

  // Stage a 6-byte POINTER (awlsim POINTER type: DB-number word @+0, S7
  // pointer dword @+2) at (area, db, byte) for a by-reference parameter of
  // compound type (STRING/DATE_AND_TIME/...). The callee reads it through
  // P##param: `L W [AR1, P#0.0]` = DB number, `L D [AR1, P#2.0]` = pointer.
  bool emitPtrParamHeader(mlir::scl::DirectArea area, int32_t db,
                          int32_t byte) {
    mlir::Value a = parseOperand(false);
    if (!a)
      return false;
    auto dt = mlir::dyn_cast<mlir::scl::DirectType>(a.getType());
    if (!dt)
      return true; // not an addressable operand: nothing to point at
    // awlsim _FC_trans_dbpointerInVL: the 6-byte header in the caller's
    // L-stack (VL from the callee) is { db-number word @+0, S7 pointer
    // dword @+2 }. An L-stack r-value is forwarded as VL (0x87); a DB
    // r-value as DB (0x84) with the DB number only in the header word
    // (db field 0, P#DBX style — awlsim drops the pointer's db field for
    // indirect access); an instance-DB r-value as DB with the current
    // DI number (DINO) in the header word.
    bool isLocal = dt.getArea() == mlir::scl::DirectArea::Local ||
                   dt.getArea() == mlir::scl::DirectArea::VLocal;
    int32_t areaS7 =
        dt.getArea() == mlir::scl::DirectArea::Input    ? 0x81
        : dt.getArea() == mlir::scl::DirectArea::Output ? 0x82
        : dt.getArea() == mlir::scl::DirectArea::Marker ? 0x83
        : isLocal                                       ? 0x87
                                                        : 0x84;
    bool isInstance =
        dt.getArea() == mlir::scl::DirectArea::DataBlock && dt.getDb() < 0;
    int32_t pdb = dt.getArea() == mlir::scl::DirectArea::DataBlock
                      ? (dt.getDb() > 0 ? dt.getDb() : 0)
                      : 0;
    uint32_t byteMask = areaS7 == 0x87 ? 0x1FFFFF : 0x1FFF;
    uint32_t ptr = ((uint32_t)areaS7 << 24) |
                   ((uint32_t)(dt.getByte() & byteMask) << 3) |
                   (uint32_t)(dt.getBit() & 7);
    if (isInstance) {
      // DB number = the currently open instance DB (awlsim: diRegister)
      mlir::Value dino = builder.create<awl::DbInfoOp>(
          loc(), builder.getI64Type(), builder.getI32IntegerAttr(3),
          builder.getI32IntegerAttr(0));
      builder.create<awl::StoreOp>(
          loc(), castTo(dino, builder.getI32Type()),
          mkDirect(area, db, byte, 0, 16));
    } else {
      builder.create<awl::StoreOp>(
          loc(), constInt(builder.getI32Type(), pdb),
          mkDirect(area, db, byte, 0, 16));
    }
    builder.create<awl::StoreOp>(
        loc(), constInt(builder.getI32Type(), (int32_t)ptr),
        mkDirect(area, db, byte + 2, 0, 32));
    return true;
  }

  // Emit a call to a runtime-implemented function: parse the
  // (name :=/=> value, ...) list, stage the inputs at kLocalBase, emit the
  // call, then read the outputs back. rtNum goes into the `sfc` attr
  // (callee SFCn) or the `iec` attr (callee IECn).
  bool emitRtCall(bool cond, mlir::Value rloVal, const std::string &callee,
                  int32_t rtNum, bool isIec, const RtParam *tab, int tabLen) {
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), callee)));
    attrs.push_back(builder.getNamedAttr(isIec ? "iec" : "sfc",
                                         builder.getI32IntegerAttr(rtNum)));
    struct SfcOut {
      int32_t byte, width;
      mlir::Value target;
    };
    llvm::SmallVector<SfcOut, 2> sfcOuts;
    if (lexer.getCurToken() == Tok::lparen) {
      lexer.next();
      while (lexer.getCurToken() != Tok::rparen &&
             lexer.getCurToken() != Tok::eof) {
        if (lexer.getCurToken() == Tok::comma ||
            lexer.getCurToken() == Tok::semi) {
          lexer.next();
          continue;
        }
        if (lexer.getCurToken() != Tok::ident &&
            lexer.getCurToken() != Tok::localvar)
          return error("expected parameter name in CALL");
        std::string pname = upperStr(lexer.getIdent());
        lexer.next();
        bool isOutput = false;
        if (lexer.getCurToken() == Tok::colon) { // :=
          lexer.next();
          if (lexer.getCurToken() != Tok::assign)
            return error("expected ':=' in CALL parameter");
          lexer.next();
        } else if (lexer.getCurToken() == Tok::assign) { // =>
          lexer.next();
          if (lexer.getCurToken() != Tok::gt)
            return error("expected '=>' in CALL parameter");
          lexer.next();
          isOutput = true;
        } else {
          return error("expected ':=' or '=>' in CALL parameter");
        }
        const RtParam *p = nullptr;
        for (int i = 0; i < tabLen; ++i)
          if (pname == tab[i].name)
            p = &tab[i];
        if (!p) {
          // unknown parameter: skip its value/target
          int depth = 0;
          while (lexer.getCurToken() != Tok::eof) {
            Tok t = lexer.getCurToken();
            if (t == Tok::lparen || t == Tok::lbracket)
              ++depth;
            else if (t == Tok::rparen || t == Tok::rbracket) {
              if (depth == 0)
                break;
              --depth;
            } else if (depth == 0 && (t == Tok::comma || t == Tok::rparen))
              break;
            lexer.next();
          }
          continue;
        }
        // direction comes from the interface section, not the syntax
        // (RET_VAL is passed with := but is an output)
        if (p->out)
          isOutput = true;
        if (!isOutput) {
          if (p->any) {
            // ANY param: stage the full 10-byte ANY header at kLocalBase
            if (!emitAnyParamHeader(mlir::scl::DirectArea::Marker, 0,
                                    kLocalBase + p->byte, false))
              return false;
          } else {
            mlir::Value v = parseCallParamValue(nullptr);
            if (!v)
              return false;
            builder.create<awl::StoreOp>(
                loc(), v,
                mkDirect(mlir::scl::DirectArea::Marker, 0,
                         kLocalBase + p->byte, 0, p->width));
          }
        } else {
          mlir::Value target = parseOperand(false);
          if (!target)
            return false;
          // stage the current value as well (awlsim _FC_trans_copyToVL
          // copies unconditionally): if the callee leaves the OUT slot
          // untouched, the copy-back below restores the old value
          if (mlir::isa<mlir::scl::DirectType>(target.getType())) {
            mlir::Value cur = builder.create<awl::LoadOp>(loc(), target);
            builder.create<awl::StoreOp>(
                loc(), cur,
                mkDirect(mlir::scl::DirectArea::Marker, 0,
                         kLocalBase + p->byte, 0, p->width));
          } else if (target.getType().isInteger(64)) {
            mlir::Value cur = builder.create<awl::LoadIndirectOp>(
                loc(), builder.getI64Type(), target,
                builder.getI32IntegerAttr(p->width));
            builder.create<awl::StoreOp>(
                loc(), castTo(cur, builder.getI32Type()),
                mkDirect(mlir::scl::DirectArea::Marker, 0,
                         kLocalBase + p->byte, 0, p->width));
          }
          sfcOuts.push_back({p->byte, p->width, target});
        }
      }
      if (lexer.getCurToken() == Tok::rparen)
        lexer.next();
    }
    builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                cond ? mlir::ValueRange{rloVal}
                                     : mlir::ValueRange{},
                                attrs);
    // BIE/BR is set by the runtime (scl_rt_sfc/scl_rt_sfb: 1 on success,
    // 0 on error paths, awlsim system_sfc_*/system_sfb_*); IEC library
    // FCs set it per their AWL body (see scl_rt_iec)
    // output params: read back from the staging area
    for (auto &o : sfcOuts) {
      auto addr = mkDirect(mlir::scl::DirectArea::Marker, 0,
                           kLocalBase + o.byte, 0, o.width);
      mlir::Value v = builder.create<awl::LoadOp>(loc(), addr);
      if (mlir::isa<mlir::scl::DirectType>(o.target.getType()))
        builder.create<awl::StoreOp>(loc(), v, o.target);
      else if (o.target.getType().isInteger(64))
        builder.create<awl::StoreIndirectOp>(
            loc(), castTo(v, builder.getI64Type()), o.target,
            builder.getI32IntegerAttr(o.width));
    }
    return true;
    return true;
  }

  // SPEC[AWL] Siemens AWL manual "Calling blocks" (CALL/UC/CC with parameter lists := / =>, FB instance DB, multi-instance via AR2)
  bool parseCall(bool cond) {
    // CALL/UC/CC FC n | CALL/UC/CC FB n, DB m ; CC calls only if RLO = 1
    mlir::Value rloVal;
    if (cond) {
      if (!ensureRlo())
        return false;
      rloVal = rlo;
    }
    std::string kind;
    std::string orig;
    std::string namedCallee; // non-empty: CALL "NAME" of a user block
    int64_t suffixed = 0;
    bool hasSuffix = false;
    int32_t multiBase = -1; // >= 0: CALL #var — multi-instance FB/SFB at this offset
    int32_t sfbDbNum = -1;  // direct CALL SFB n, DB m: instance DB number
    if (lexer.getCurToken() == Tok::localvar) {
      // CALL #VAR: call through a block-typed variable — a multi-instance
      // FB/SFB embedded in the caller's instance DB at the member's offset
      std::string vname = upperStr(lexer.getIdent());
      lexer.next();
      const MemberInfo *mi = nullptr;
      if (curLayout) {
        auto it = curLayout->members.find(vname);
        if (it != curLayout->members.end())
          mi = &it->second;
      }
      std::string vty = mi ? mi->type : "";
      // quoted block type names ("Function Block 2") resolve via
      // namedBlockNums (typeNameOf strips the quotes)
      if (mi && !vty.empty()) {
        std::string bare = vty.front() == '"' ? vty.substr(1, vty.size() - 2) : vty;
        auto nit = namedBlockNums.find(bare);
        if (nit != namedBlockNums.end() &&
            (nit->second.first == "FB" || nit->second.first == "SFB"))
          vty = nit->second.first + " " + std::to_string(nit->second.second);
      }
      if (mi && (vty.rfind("FB ", 0) == 0 || vty.rfind("SFB ", 0) == 0)) {
        bool isSfb = vty.rfind("SFB", 0) == 0;
        kind = isSfb ? "SFB" : "FB";
        suffixed = atoll(vty.c_str() + (isSfb ? 4 : 3));
        hasSuffix = true;
        multiBase = mi->byte;
      } else {
      // unknown variable: call a stub so the code still runs
      lexer.next();
      if (lexer.getCurToken() == Tok::comma) {
        lexer.next();
        if (lexer.getCurToken() == Tok::ident ||
            lexer.getCurToken() == Tok::localvar)
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
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), "FC0")));
      builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                  cond ? mlir::ValueRange{rloVal}
                                       : mlir::ValueRange{},
                                  attrs);
      if (!cond)
        rloActive = false;
      return true;
      }
    } else {
    if (lexer.getCurToken() != Tok::ident)
      return error("expected FC/FB/SFC after CALL");
    orig = upperIdent();
    kind = orig;
    lexer.next();
    // awlsim writes the block number attached: CALL SFC47(...)
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
    }
    if (kind != "FC" && kind != "FB" && kind != "SFC" && kind != "SFB") {
      // symbolic block name: CALL "GE_DT" (params) — callee referenced by name
      if (lexer.getCurToken() == Tok::comma) { // CALL "FB", "Instance DB" (..)
        lexer.next();
        if (lexer.getCurToken() == Tok::ident)
          lexer.next();
      }
      // Standard-library functions called by name (awlsim library/iec and
      // system symbols) are implemented in the runtime with staged params.
      static const struct {
        const char *name;
        int32_t num;
        const RtParam *tab;
        int tabLen;
        bool isSfc;
      } rtBuiltins[] = {
          {"LEN", 21, iecLenTab, 2, false},      // FC 21: string length
          {"EQ_DT", 9, iecDtTab, 3, false},      // FC 9:  DATE_AND_TIME ==
          {"GE_DT", 12, iecDtTab, 3, false},     // FC 12: DATE_AND_TIME >=
          {"DELETE", 4, iecDeleteTab, 4, false}, // FC 4: delete substring
          {"TIME_TCK", 64, sfc64Tab, 1, true},   // SFC 64: TIME_TCK
      };
      for (const auto &b : rtBuiltins)
        if (orig == b.name)
          return emitRtCall(cond, rloVal,
                            b.isSfc ? "SFC" + std::to_string(b.num)
                                    : "IEC" + std::to_string(b.num),
                            b.num, !b.isSfc, b.tab, b.tabLen);
      auto nit = namedBlockNums.find(orig);
      if (nit != namedBlockNums.end()) {
        // user block called by name: keep the quoted symbol and route
        // through the common block-call path below (with params staged)
        kind = nit->second.first;
        namedCallee = orig;
        hasSuffix = true;
        suffixed = nit->second.second;
        goto common_call_path;
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
    if (kind == "SFC" && multiBase < 0) {
      // system function call: CALL SFC [-]n [(params)] — params are staged
      // in the marker area at kLocalBase and run by scl_rt_sfc in the runtime
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
          return error("expected SFC number after CALL");
        n = lexer.getInteger();
        lexer.next();
      }
      if (neg)
        n = -n;
      const RtParam *tab = nullptr;
      int tabLen = 0;
      if (n == 21) {
        tab = sfc21Tab;
        tabLen = 3;
      } else if (n == 47) {
        tab = sfc47Tab;
        tabLen = 1;
      } else if (n == 64) {
        tab = sfc64Tab;
        tabLen = 1;
      }
      auto call = emitRtCall(cond, rloVal, "SFC" + std::to_string(n), (int32_t)n,
                             false, tab, tabLen);
      // awlsim system_sfc_46 / system_sfc_m3: STOP/SHUTDOWN raise a
      // maintenance request that aborts the current scan — terminate the
      // block here (like BEA); the rest of the block is dead code
      if (n == 46 || n == -3) {
        builder.create<awl::BeOp>(loc());
        blockEnded = true;
      }
      return call;
    }
    if (kind == "SFB" && multiBase < 0) {
      // direct SFB call: CALL SFB n, DB m [(params)] — the instance DB and
      // parameters are handled by the common path below
      bool negSfb = false;
      if (lexer.getCurToken() == Tok::minus) { // CALL SFB -1 (awlsim basic.awl)
        lexer.next();
        negSfb = true;
      }
      if (!hasSuffix) {
        if (lexer.getCurToken() == Tok::integer) {
          suffixed = lexer.getInteger();
          if (negSfb)
            suffixed = -suffixed;
          hasSuffix = true;
          lexer.next();
        }
      }
      if (lexer.getCurToken() == Tok::comma) {
        lexer.next();
        if (lexer.getCurToken() == Tok::ident && upperIdent() == "DB")
          lexer.next();
        if (lexer.getCurToken() == Tok::integer) {
          sfbDbNum = (int32_t)lexer.getInteger();
          lexer.next();
        }
      }
    }
  common_call_path:
    int64_t num;
    mlir::Value indirectNum;
    if (hasSuffix) {
      num = suffixed;
    } else if (lexer.getCurToken() == Tok::lbracket) {
      // indirect block call: UC/CC FC [MW 50] — the block number is runtime
      // data; keep the operand value and dispatch over the blocks known in
      // this file (jump table, like SPL)
      lexer.next();
      indirectNum = parseOperand(false);
      if (!indirectNum)
        return false;
      // parseOperand yields the operand address for memory operands; the
      // block number is the loaded value
      if (mlir::isa<mlir::scl::DirectType>(indirectNum.getType()))
        indirectNum = zextAcc(builder.create<awl::LoadOp>(loc(), indirectNum));
      while (lexer.getCurToken() != Tok::rbracket &&
             lexer.getCurToken() != Tok::eof)
        lexer.next();
      if (lexer.getCurToken() == Tok::rbracket)
        lexer.next();
      num = 0;
    } else {
      bool neg = false;
      if (lexer.getCurToken() == Tok::minus) {
        lexer.next();
        neg = true;
      }
      if (lexer.getCurToken() != Tok::integer)
        return error("expected block number after " + kind);
      num = lexer.getInteger();
      if (neg)
        num = -num;
      lexer.next();
    }
    std::string callee =
        !namedCallee.empty() ? namedCallee : kind + std::to_string(num);
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr(
        "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(), callee)));
    int64_t dbNum = 0;
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
      dbNum = db;
      attrs.push_back(
          builder.getNamedAttr("db", builder.getI32IntegerAttr((int32_t)db)));
      // explicit ", DB m": the callee runs with this instance DB (AR2 is
      // set to P#DBX 0.0 for the duration of the call)
      attrs.push_back(builder.getNamedAttr("instdb", builder.getUnitAttr()));
    }
    if (kind == "FB" && multiBase < 0 && !attrs.empty() && attrs.size() == 1)
      // CALL FB n without ", DB m": pass DB 0 (the current DB register) so
      // the call matches the FB function's i32 argument
      attrs.push_back(
          builder.getNamedAttr("db", builder.getI32IntegerAttr(0)));
    if (kind == "SFB" && multiBase < 0) {
      attrs.push_back(builder.getNamedAttr(
          "sfc", builder.getI32IntegerAttr((int32_t)num)));
      if (sfbDbNum >= 0) {
        attrs.push_back(builder.getNamedAttr(
            "db", builder.getI32IntegerAttr(sfbDbNum)));
        attrs.push_back(builder.getNamedAttr("instdb", builder.getUnitAttr()));
      }
    }
    if (multiBase >= 0) {
      // multi-instance call: instance DB = DI register (db -1), AR2 is set
      // to P#DBX + multiBase for the duration of the call
      attrs.push_back(
          builder.getNamedAttr("db", builder.getI32IntegerAttr(-1)));
      attrs.push_back(builder.getNamedAttr("instdb", builder.getUnitAttr()));
      attrs.push_back(builder.getNamedAttr(
          "ar2off", builder.getI32IntegerAttr(multiBase)));
      if (kind == "SFB")
        attrs.push_back(builder.getNamedAttr(
            "sfc", builder.getI32IntegerAttr((int32_t)num)));
    }
    // Optional parameter list: CALL FB 1, DB 2 (P := v, Q => addr).
    // Input params (:=) are stored into the callee's interface area before
    // the call (instance DB m for FB, L area for FC); output params (=>)
    // are read back after the call and stored to the target operand.
    const Layout *callLayout = nullptr;
    int32_t paramDb = 0;   // DB number for DB-area params (FB/SFB)
    int32_t paramBase = 0; // multi-instance offset within the instance DB
    if (multiBase >= 0) {
      paramDb = -1; // DI register
      paramBase = multiBase;
      if (kind == "FB") {
        auto it = fbLayouts.find((int32_t)num);
        if (it != fbLayouts.end())
          callLayout = &it->second;
      } else {
        auto sit = sfbCache.find((int32_t)num);
        if (sit == sfbCache.end())
          sit = sfbCache.emplace((int32_t)num, sfbLayoutOf((int32_t)num)).first;
        callLayout = &sit->second;
      }
    } else if (kind == "FB") {
      paramDb = (int32_t)dbNum;
      auto it = fbLayouts.find((int32_t)num);
      if (it != fbLayouts.end())
        callLayout = &it->second;
    } else if (kind == "SFB") {
      if (sfbDbNum >= 0)
        paramDb = sfbDbNum;
      auto sit = sfbCache.find((int32_t)num);
      if (sit == sfbCache.end())
        sit = sfbCache.emplace((int32_t)num, sfbLayoutOf((int32_t)num)).first;
      callLayout = &sit->second;
    } else {
      auto it = fcLayouts.find((int32_t)num);
      if (it != fcLayouts.end())
        callLayout = &it->second;
    }
    if (!callLayout && !namedCallee.empty()) {
      auto nl = namedLayouts.find(namedCallee);
      if (nl != namedLayouts.end())
        callLayout = &nl->second;
    }
    struct OutParam {
      const MemberInfo *m;
      mlir::Value target;
      bool array = false;
    };
    // Whole-array parameter copy: `P := DB2.ARR` copies all elements
    // between the source operand and the callee's interface area.
    auto copyArrayParam = [&](const MemberInfo *m, int32_t baseByte,
                              mlir::Value srcDirect,
                              mlir::scl::DirectArea dstArea, int32_t dstDb,
                              bool toCallee) {
      auto sdt = mlir::cast<mlir::scl::DirectType>(srcDirect.getType());
      int32_t elemBits = m->elemSize ? m->elemSize * 8 : 1;
      int32_t ew = m->widthBits ? m->widthBits : 16;
      for (int32_t k = 0; k < m->count; ++k) {
        int32_t sbits = sdt.getBit() + k * elemBits;
        auto saddr = builder.create<awl::DirectOp>(
            loc(), sdt.getArea(), sdt.getDb(), sdt.getByte() + sbits / 8,
            sbits % 8, ew);
        int32_t dbits = m->bit + k * elemBits;
        auto daddr =
            mkDirect(dstArea, dstDb, baseByte + dbits / 8, dbits % 8, ew);
        if (toCallee) {
          mlir::Value v = builder.create<awl::LoadOp>(loc(), saddr);
          builder.create<awl::StoreOp>(loc(), v, daddr);
        } else {
          mlir::Value v = builder.create<awl::LoadOp>(loc(), daddr);
          builder.create<awl::StoreOp>(loc(), v, saddr);
        }
      }
    };
    // Whole-array copy from an indirect source (a forwarded by-ref FC
    // param, e.g. `VAR0 := #VAR00`): element k is read via load_indirect
    // at src + k*elemBits and stored into the callee's interface area.
    auto copyArrayParamInd = [&](const MemberInfo *m, int32_t baseByte,
                                 mlir::Value src, // i64 raw ptr to element lo
                                 mlir::scl::DirectArea dstArea, int32_t dstDb) {
      int32_t elemBits = m->elemSize ? m->elemSize * 8
                                     : (m->widthBits ? m->widthBits : 16);
      int32_t ew = m->widthBits ? m->widthBits : 16;
      for (int32_t k = 0; k < m->count; ++k) {
        mlir::Value a = src;
        if (k * elemBits)
          a = builder.create<awl::AddOp>(
              loc(), builder.getI64Type(), src,
              constInt(builder.getI64Type(), (int64_t)k * elemBits));
        mlir::Value v = builder.create<awl::LoadIndirectOp>(
            loc(), builder.getI64Type(), a, builder.getI32IntegerAttr(ew));
        int32_t dbits = m->bit + k * elemBits;
        builder.create<awl::StoreOp>(loc(), v,
                                     mkDirect(dstArea, dstDb, baseByte + dbits / 8,
                                              dbits % 8, ew));
      }
    };
    // Whole-STRUCT/UDT parameter copy: `P := DB10.STRUCTVAR` copies all of
    // the source struct's bytes between the source operand and the callee's
    // interface area (like copyArrayParam, but for a single struct member).
    auto copyBlockParam = [&](const MemberInfo *m, int32_t baseByte,
                              mlir::Value srcDirect,
                              mlir::scl::DirectArea dstArea, int32_t dstDb,
                              bool toCallee) {
      auto sdt = mlir::cast<mlir::scl::DirectType>(srcDirect.getType());
      const Layout *sub = m->nested ? m->nested.get() : nullptr;
      if (!sub) {
        auto uit = udtLayouts.find(m->type);
        if (uit != udtLayouts.end())
          sub = &uit->second;
      }
      int32_t total = sub ? sub->size : (m->widthBits + 7) / 8;
      for (int32_t k = 0; k < total; ++k) {
        auto saddr = builder.create<awl::DirectOp>(loc(), sdt.getArea(),
                                                   sdt.getDb(),
                                                   sdt.getByte() + k, 0, 8);
        auto daddr = mkDirect(dstArea, dstDb, baseByte + k, 0, 8);
        if (toCallee) {
          mlir::Value v = builder.create<awl::LoadOp>(loc(), saddr);
          builder.create<awl::StoreOp>(loc(), v, daddr);
        } else {
          mlir::Value v = builder.create<awl::LoadOp>(loc(), daddr);
          builder.create<awl::StoreOp>(loc(), v, saddr);
        }
      }
    };
    // a param is copied as a block when it is a STRUCT or UDT member
    auto isBlockParam = [&](const MemberInfo *m) {
      return m->type == "STRUCT" || isByRefType(m->type) ||
             (m->type != "ARRAY" && !m->type.empty() &&
              udtLayouts.count(m->type) > 0);
    };
    llvm::SmallVector<OutParam, 4> outParams;
    if (lexer.getCurToken() == Tok::lparen) {
      lexer.next();
      while (lexer.getCurToken() != Tok::rparen &&
             lexer.getCurToken() != Tok::eof) {
        if (lexer.getCurToken() == Tok::comma ||
            lexer.getCurToken() == Tok::semi) {
          lexer.next();
          continue;
        }
        if (lexer.getCurToken() != Tok::ident &&
            lexer.getCurToken() != Tok::localvar)
          return error("expected parameter name in CALL");
        std::string pname = upperStr(lexer.getIdent());
        lexer.next();
        bool isOutput = false;
        if (lexer.getCurToken() == Tok::colon) { // :=
          lexer.next();
          if (lexer.getCurToken() != Tok::assign)
            return error("expected ':=' in CALL parameter");
          lexer.next();
        } else if (lexer.getCurToken() == Tok::assign) { // =>
          lexer.next();
          if (lexer.getCurToken() != Tok::gt)
            return error("expected '=>' in CALL parameter");
          lexer.next();
          isOutput = true;
        } else {
          return error("expected ':=' or '=>' in CALL parameter");
        }
        const MemberInfo *m = nullptr;
        if (callLayout) {
          auto it = callLayout->members.find(pname);
          if (it != callLayout->members.end())
            m = &it->second;
        }
        // The parameter direction comes from the interface section, not the
        // := / => syntax (awlsim uses the field type): VAR_OUTPUT fields are
        // copied back after the call and are NOT stored into before it
        if (m && m->section == 1)
          isOutput = true;
        if (!m) {
          // unknown parameter (no interface layout): skip its value/target
          int depth = 0;
          while (lexer.getCurToken() != Tok::eof) {
            Tok t = lexer.getCurToken();
            if (t == Tok::lparen || t == Tok::lbracket)
              ++depth;
            else if (t == Tok::rparen || t == Tok::rbracket)
              { if (depth == 0) break; --depth; }
            else if (depth == 0 && (t == Tok::comma || t == Tok::rparen))
              break;
            lexer.next();
          }
          continue;
        }
        // FC params at the call site use the absolute marker staging area
        // (kLocalBase); scl_rt_param_in/out copies it to/from the callee's
        // own L frame, so the caller's frame params are never clobbered.
        // FB/SFB params go to the instance DB (DI register for multi-instance
        // calls), at the multi-instance base offset if applicable.
        bool isDbParam = kind == "FB" || kind == "SFB";
        // by-reference (pointer header in the slot): FC params always;
        // FB params only when IN_OUT (awlsim overrides the instance-DB
        // field with a DB-pointer then). FB IN/OUT compound params are
        // copied by value into the instance DB.
        bool byRef = (isByRefParam(m) || m->type == "ARRAY") &&
                     (!isDbParam || m->section == 2);
        auto area = isDbParam ? mlir::scl::DirectArea::DataBlock
                              : mlir::scl::DirectArea::Marker;
        int32_t byte = isDbParam ? paramBase + m->byte : kLocalBase + m->byte;
        int32_t width = m->widthBits ? m->widthBits : 16;
        bool isArray = m->type == "ARRAY" && m->count > 1;
        bool strLit = lexer.getCurToken() == Tok::ident && lexer.curQuoted() &&
                      lexer.curQuoteChar() == '\'' &&
                      m->type.rfind("STRING", 0) == 0;
        bool dtLit = m->type == "DATE_AND_TIME" &&
                     lexer.getCurToken() == Tok::integer;
        if (!isOutput && byRef && !isDbParam && (strLit || dtLit)) {
          // literal (string / DT#) for a by-ref compound FC param: copy the
          // value into the CALLER's L frame past the callee's parameter area
          // and stage a {0, VL-pointer} header at the param slot (awlsim
          // _FC_trans_copyToVLWithDBPtr: L -> VL when forwarded)
          std::string raw;
          int64_t dtVal = 0;
          int32_t maxlen = 254;
          if (strLit) {
            raw = lexer.getIdent().str();
            lexer.next();
            maxlen = m->type.size() > 6 ? atoi(m->type.c_str() + 6) : 254;
          } else {
            dtVal = lexer.getInteger(); // DT#...: 8 BCD bytes, byte0=year
            lexer.next();
          }
          int32_t litBytes = strLit ? maxlen + 2 : 8;
          if (litBytes & 1)
            ++litBytes;
          // past the callee's whole parameter area within this frame
          int32_t base = kLocalBase +
                         (callLayout ? callLayout->size : 0) + blockVlLitOff;
          blockVlLitOff += litBytes;
          if (base + litBytes > blockVlLitEnd)
            blockVlLitEnd = base + litBytes;
          auto putB = [&](int32_t off, int32_t v) {
            builder.create<awl::StoreOp>(
                loc(), constInt(builder.getI32Type(), v),
                mkDirect(mlir::scl::DirectArea::Local, 0, base + off, 0, 8));
          };
          if (strLit) {
            putB(0, maxlen);
            putB(1, (int32_t)raw.size());
            for (size_t i = 0; i < raw.size(); ++i)
              putB(2 + (int32_t)i, (unsigned char)raw[i]);
            for (int32_t i = (int32_t)raw.size(); i < maxlen; ++i)
              putB(2 + i, 0);
          } else {
            // DT# packs byte 0 (year) in the HIGH byte of the integer
            for (int32_t k = 0; k < 8; ++k)
              putB(k, (int32_t)((dtVal >> (8 * (7 - k))) & 0xFF));
          }
          // stage the 6-byte header { db-nr 0, VL pointer } in the marker
          // staging area; param_in copies it to the caller frame's VL area
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), 0),
              mkDirect(area, isDbParam ? paramDb : 0, byte, 0, 16));
          uint32_t ptr = ((uint32_t)0x87 << 24) |
                         ((uint32_t)(base & 0x1FFFFF) << 3);
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), (int32_t)ptr),
              mkDirect(area, isDbParam ? paramDb : 0, byte + 2, 0, 32));
          continue;
        }
        if (!isOutput && m->type == "ANY" && isDbParam && m->section != 2) {
          // FB/SFB VAR_IN ANY param: the whole 10-byte ANY header is copied
          // into the instance-DB slot (awlsim FB param translation copies
          // ANY fields verbatim; the header keeps its original area bytes,
          // so no L->VL translation: toVl=false)
          bool ok = emitAnyParamHeader(area, paramDb, byte, /*toVl=*/false);
          if (!ok)
            return false;
          continue;
        }
        if (!isOutput && byRef && byRefRhs(m->type == "ANY")) {
          // by-reference param: ANY params stage a 10-byte ANY header,
          // compound types a 6-byte POINTER; no value copy either way
          bool ok = m->type == "ANY"
              ? emitAnyParamHeader(area, isDbParam ? paramDb : 0, byte)
              : emitPtrParamHeader(area, isDbParam ? paramDb : 0, byte);
          if (!ok)
            return false;
          continue;
        }
        if (!isOutput && m->type == "POINTER" &&
            lexer.getCurToken() == Tok::integer && lexer.curPtrLit()) {
          // POINTER param, P#... literal RHS: stage the 6-byte DB-pointer
          // { db-nr word @+0, S7 ptr dword @+2 with db field 0 } in the
          // slot (awlsim DBPointer memory layout)
          int64_t v = consumePtrLiteral(); // resolves symbolic members
          uint32_t areaS7 = (uint32_t)((v >> 24) & 0xFF);
          uint32_t dbw = (areaS7 == 0x84 || areaS7 == 0x85)
                             ? (uint32_t)((v >> 16) & 0xFF)
                             : 0;
          uint32_t ptr = (uint32_t)(v & 0xFF00FFFFll); // db field 0
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), (int32_t)dbw),
              mkDirect(area, isDbParam ? paramDb : 0, byte, 0, 16));
          builder.create<awl::StoreOp>(
              loc(), constInt(builder.getI32Type(), (int32_t)ptr),
              mkDirect(area, isDbParam ? paramDb : 0, byte + 2, 0, 32));
          continue;
        }
        if (!isOutput && m->type == "POINTER" && callRhsAddressable()) {
          // POINTER param, addressable operand RHS (DB50.DBX 10.1, M 99.7,
          // #var): stage the pointer header like a by-ref param
          bool ok = emitPtrParamHeader(area, isDbParam ? paramDb : 0, byte);
          if (!ok)
            return false;
          continue;
        }
        if (!isOutput && isDbParam && isByRefType(m->type) &&
            m->section != 2) {
          // FB IN/OUT compound param (e.g. STRING): copied by value into
          // the instance DB slot (awlsim FB param translation copies the
          // value; only IN_OUT compound fields are DB-pointers)
          if (lexer.getCurToken() == Tok::ident && lexer.curQuoted() &&
              lexer.curQuoteChar() == '\'' && m->type.rfind("STRING", 0) == 0) {
            std::string raw = lexer.getIdent().str();
            lexer.next();
            int32_t maxlen =
                m->type.size() > 6 ? atoi(m->type.c_str() + 6) : 254;
            auto putB = [&](int32_t off, int32_t v) {
              builder.create<awl::StoreOp>(
                  loc(), constInt(builder.getI32Type(), v),
                  mkDirect(area, paramDb, byte + off, 0, 8));
            };
            putB(0, maxlen);
            putB(1, (int32_t)raw.size());
            for (size_t i = 0; i < raw.size(); ++i)
              putB(2 + (int32_t)i, (unsigned char)raw[i]);
            for (int32_t i = (int32_t)raw.size(); i < maxlen; ++i)
              putB(2 + i, 0);
            continue;
          }
          mlir::Value target;
          mlir::Value v = parseCallParamValue(&target);
          if (!v)
            return false;
          if (target && mlir::isa<mlir::scl::DirectType>(target.getType()))
            copyBlockParam(m, byte, target, area, paramDb, true);
          else
            builder.create<awl::StoreOp>(
                loc(), v, mkDirect(area, paramDb, byte, m->bit, width));
          continue;
        }
        if (!isOutput) {
          mlir::Value target;
          mlir::Value v = parseCallParamValue(&target);
          if (!v)
            return false;
          if (isArray && target &&
              mlir::isa<mlir::scl::DirectType>(target.getType())) {
            copyArrayParam(m, byte, target, area,
                           isDbParam ? paramDb : 0, true);
          } else if (isArray && v.getType().isInteger(64)) {
            // array RHS resolved to an indirect address (forwarded by-ref
            // FC param): copy the elements through the pointer
            copyArrayParamInd(m, byte, v, area, isDbParam ? paramDb : 0);
          } else if (isBlockParam(m) && target &&
                     mlir::isa<mlir::scl::DirectType>(target.getType())) {
            copyBlockParam(m, byte, target, area, isDbParam ? paramDb : 0,
                           true);
          } else {
            auto addr = mkDirect(area, isDbParam ? paramDb : 0,
                                 byte, m->bit, width);
            builder.create<awl::StoreOp>(loc(), v, addr);
          }
          // by-reference parameter: an operand RHS is written back after
          // the call (awlsim passes FC/FB params by reference)
          if (target)
            outParams.push_back(
                {m, target, isArray && mlir::isa<mlir::scl::DirectType>(
                                          target.getType())});
        } else if ((byRef || m->type == "POINTER") && callRhsAddressable()) {
          // by-reference / POINTER OUT param: stage the pointer header
          // before the call so the callee can write through it; nothing to
          // read back as a value
          bool ok = m->type == "ANY"
              ? emitAnyParamHeader(area, isDbParam ? paramDb : 0, byte)
              : emitPtrParamHeader(area, isDbParam ? paramDb : 0, byte);
          if (!ok)
            return false;
        } else {
          mlir::Value target = parseOperand(false);
          if (!target)
            return false;
          outParams.push_back(
              {m, target, isArray && mlir::isa<mlir::scl::DirectType>(
                                        target.getType())});
        }
      }
      if (lexer.getCurToken() == Tok::rparen)
        lexer.next();
    }
    if (indirectNum) {
      // jump-table dispatch over the FC/FB blocks declared in this file
      auto i32 = builder.getI32Type();
      mlir::Value n = castTo(indirectNum, i32);
      std::string base =
          "__ind" + std::to_string(indirectSeq++) + "_";
      std::vector<int32_t> nums;
      const auto &layouts = kind == "FB" ? fbLayouts : fcLayouts;
      for (auto &kv : layouts)
        nums.push_back(kv.first);
      std::sort(nums.begin(), nums.end());
      if (cond)
        builder.create<awl::JumpIfNotOp>(loc(), rloVal,
                                         builder.getStringAttr(base + "end"));
      for (int32_t num : nums) {
        auto c = builder.create<awl::CmpEqOp>(loc(), builder.getI1Type(), n,
                                              constInt(i32, num));
        builder.create<awl::JumpIfOp>(
            loc(), c,
            builder.getStringAttr(base + "n" + std::to_string(num)));
      }
      builder.create<awl::JumpOp>(loc(), builder.getStringAttr(base + "end"));
      for (int32_t num : nums) {
        builder.create<awl::LabelOp>(
            loc(), builder.getStringAttr(base + "n" + std::to_string(num)));
        llvm::SmallVector<mlir::NamedAttribute, 2> cattrs;
        cattrs.push_back(builder.getNamedAttr(
            "callee", mlir::FlatSymbolRefAttr::get(
                          builder.getContext(), kind + std::to_string(num))));
        if (kind == "FB")
          cattrs.push_back(builder.getNamedAttr(
              "db", builder.getI32IntegerAttr((int32_t)dbNum)));
        builder.create<awl::CallOp>(loc(), mlir::TypeRange{}, mlir::ValueRange{},
                                    cattrs);
        builder.create<awl::JumpOp>(loc(), builder.getStringAttr(base + "end"));
      }
      builder.create<awl::LabelOp>(loc(), builder.getStringAttr(base + "end"));
    } else {
      builder.create<awl::CallOp>(loc(), mlir::TypeRange{},
                                  cond ? mlir::ValueRange{rloVal}
                                       : mlir::ValueRange{},
                                  attrs);
      // CALL SFB n: scl_rt_sfb sets BIE/BR (1 on success, 0 on error)
    }
    // output params: read back from the callee's interface area
    for (auto &op : outParams) {
      bool isDbParam = kind == "FB" || kind == "SFB";
      auto area = isDbParam ? mlir::scl::DirectArea::DataBlock
                            : mlir::scl::DirectArea::Marker;
      int32_t byte = isDbParam ? paramBase + op.m->byte
                               : kLocalBase + op.m->byte;
      int32_t width = op.m->widthBits ? op.m->widthBits : 16;
      if (op.array && mlir::isa<mlir::scl::DirectType>(op.target.getType())) {
        copyArrayParam(op.m, byte, op.target, area,
                       isDbParam ? paramDb : 0, false);
        continue;
      }
      if (isBlockParam(op.m) &&
          mlir::isa<mlir::scl::DirectType>(op.target.getType())) {
        copyBlockParam(op.m, byte, op.target, area, isDbParam ? paramDb : 0,
                       false);
        continue;
      }
      auto addr = mkDirect(area, isDbParam ? paramDb : 0, byte,
                           op.m->bit, width);
      mlir::Value v = builder.create<awl::LoadOp>(loc(), addr);
      if (mlir::isa<mlir::scl::DirectType>(op.target.getType()))
        builder.create<awl::StoreOp>(loc(), v, op.target);
      else if (op.target.getType().isInteger(64))
        builder.create<awl::StoreIndirectOp>(
            loc(), castTo(v, builder.getI64Type()), op.target,
            builder.getI32IntegerAttr(width));
    }
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
