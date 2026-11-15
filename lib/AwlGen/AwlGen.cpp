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
  int getLitWidth() const { return cur_.litWidth; }
  double getReal() const { return cur_.realVal; }
  int line() const { return lineNo; }
  llvm::StringRef raw() const { return cur; } // unconsumed input
  int col() const { return colNo; }
  // line of the current token (lineNo itself is ahead due to lookahead)
  int curLine() const { return cur_.line; }
  bool curQuoted() const { return cur_.quoted; }
  bool curBcdLit() const { return cur_.bcdLit; }
  bool curPtrLocal() const { return cur_.ptrLocal; }

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
    bool ptrLocal = false; // P##name: pointer to a block-local variable
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
            toupper((unsigned char)cur[1]) == 'B') {
          cur = cur.drop_front(2);
          area = 0x84;
          while (!cur.empty() && isdigit((unsigned char)cur.front())) {
            db = db * 10 + (cur.front() - '0');
            cur = cur.drop_front();
          }
          // DBX = the instance DB (DI area, code 0x85); other remaining
          // letters of the area word (DBB/DBW/DBD) are skipped
          if (!cur.empty() && toupper((unsigned char)cur.front()) == 'X') {
            area = 0x85;
            cur = cur.drop_front();
          } else {
            while (!cur.empty() && isalpha((unsigned char)cur.front()))
              cur = cur.drop_front();
          }
        } else if (!cur.empty() && isalpha((unsigned char)cur.front())) {
          char a = (char)toupper((unsigned char)cur.front());
          cur = cur.drop_front();
          switch (a) {
          case 'P': area = 0x80; break;
          case 'E': case 'I': area = 0x81; break;
          case 'A': case 'Q': area = 0x82; break;
          case 'M': area = 0x83; break;
          case 'L': area = 0x86; break;
          default: break; // not an area letter: symbolic name, skip below
          }
          if (area == 0)
            while (!cur.empty() &&
                   (isalnum((unsigned char)cur.front()) || cur.front() == '_'))
              cur = cur.drop_front();
        }
        // symbolic member parts: P#DB42.DTVAR1, P#"DB".VAR, [i]
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
                                          "STRING", "TIMER", "COUNTER"};
        for (const char *tw : typeWords) {
          size_t tl = strlen(tw);
          if (cur.size() >= tl && cur.substr(0, tl).equals_insensitive(tw)) {
            cur = cur.drop_front(tl);
            skipWs();
            while (!cur.empty() && isdigit((unsigned char)cur.front()))
              cur = cur.drop_front();
            break;
          }
        }
        t.intVal = (area << 24) | (db << 16) | (byte * 8 + bit);
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
  if (ty.rfind("STRING", 0) == 0) { // STRING[n]: n+2 bytes
    int n = 254;
    if (ty.size() > 6)
      n = atoi(ty.c_str() + 6);
    size = n + 2; align = 2; widthBits = size * 8; return;
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
static void parseStructBody(Lexer &lx, Layout &layout,
                            const std::map<std::string, Layout> &udts,
                            std::initializer_list<const char *> endKws,
                            int section = 0) {
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
  std::string forcedKind; // set by parseModule for FUNCTION(_BLOCK) quoted
  const Layout *curLayout = nullptr;         // interface of the block being parsed
  const Layout *curTempLayout = nullptr;     // VAR_TEMP of the block being parsed
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
              parseStructBody(lx, dst, udtLayouts, {"END_VAR"}, sec);
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
              // total member extent in bits (arrays: count * elem stride)
              int32_t totalBits =
                  m.type == "ARRAY"
                      ? (m.isBit ? m.count : m.count * m.stride * 8)
                      : (m.widthBits ? m.widthBits : 1);
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
            L.bitCursor = bit + sz * 8;
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
        }
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
          mlir::IntegerAttr());
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
            mlir::IntegerAttr(), mlir::IntegerAttr());
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
          mlir::IntegerAttr(), mlir::IntegerAttr());
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
      if (w == "FUNCTION" || w == "FUNCTION_BLOCK") {
        // FB/FC declaration with interface + body: parse as a block so
        // CALL can invoke it (layouts were collected by the pre-scan)
        forcedKind = w == "FUNCTION" ? "FC" : "FB";
        lexer.next();
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
          loc(), builder.getStringAttr("__scl_db_init"), mlir::IntegerAttr(),
          mlir::IntegerAttr());
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
    curBlockNum = (int32_t)num;
    curBlockFb = kind == "FB";
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
    auto blockOp =
        builder.create<awl::BlockOp>(blockLoc, builder.getStringAttr(name),
                                     mlir::IntegerAttr(), mlir::IntegerAttr());
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
    acc.clear();
    rlo = nullptr;
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
    if (m == "BE" || m == "BEA" || m == "BEU") {
      builder.create<awl::BeOp>(loc());
      blockEnded = true;
      return true;
    }
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
      if (lexer.getCurToken() == Tok::semi || lexer.getCurToken() == Tok::rparen)
        return ensureRlo(); // bare `O ;`
      if (lexer.getCurToken() == Tok::eq || lexer.getCurToken() == Tok::ne ||
          lexer.getCurToken() == Tok::lt || lexer.getCurToken() == Tok::le ||
          lexer.getCurToken() == Tok::gt || lexer.getCurToken() == Tok::ge) {
        lexer.next(); // `U ==0` status compare: approximate as bare
        if (lexer.getCurToken() == Tok::integer)
          lexer.next();
        return ensureRlo();
      }
      if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted() &&
          lexer.curLine() != insnLine)
        return ensureRlo(); // next line starts a new instruction: bare op
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
          return ensureRlo();
        }
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
        if (!parseSetDbValue())
          return false;
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
      if (lexer.getCurToken() == Tok::localvar && lexer.curPtrLocal()) {
        ptr = parsePtrLocal(); // LAR1 P##name
      } else if (lexer.getCurToken() == Tok::integer ||
          lexer.getCurToken() == Tok::real) {
        int64_t off = lexer.getInteger(); // P#x.y lexes as byte*8+bit
        lexer.next();
        ptr = constInt(builder.getI64Type(), off);
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
        else if (w.size() <= 2 && w[0] == 'L' &&
                 lexer.peekToken() == Tok::integer) { // LD/LW/LB operand
          lexer.next();
          int32_t byte = 0;
          if (lexer.getCurToken() == Tok::integer) {
            byte = (int32_t)lexer.getInteger();
            lexer.next();
          }
          ptr = builder.create<awl::DirectOp>(
              loc(), mlir::scl::DirectArea::Local, 0, byte, 0, 16);
        } else {
          ptr = parseOperand(false); // DB1.DBW 4, M 10.0, ...
          if (!ptr)
            return true;
        }
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
      if (lexer.getCurToken() == Tok::localvar && lexer.curPtrLocal()) {
        ptr = parsePtrLocal(); // TAR1 P##name
      } else if (lexer.getCurToken() == Tok::integer ||
                 lexer.getCurToken() == Tok::real) {
        int64_t off = lexer.getInteger(); // P#x.y
        lexer.next();
        ptr = constInt(builder.getI64Type(), off);
      } else if (lexer.getCurToken() == Tok::ident && !lexer.curQuoted()) {
        std::string w = upperIdent();
        if (w == "AR1" || w == "AR2") { // TARn ARm: copy the other AR
          lexer.next();
          ptr = builder.create<awl::ArGetOp>(
              loc(), builder.getI64Type(),
              builder.getI32IntegerAttr(w[2] - '0'));
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
            ptr = builder.create<awl::DirectOp>(
                loc(), mlir::scl::DirectArea::Local, 0, byte, 0, 16);
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
        m == "XOD" || m == "UD" || m == "UW" || m == "UB" || m == "OB")
      return parseWordLogic(m);
    if (m == "SP" || m == "SE" || m == "SS" || m == "SV" || m == "SI" ||
        m == "SA" || m == "SD") // SD = English off-delay (German SA)
      return parseTimerStart(m == "SD" ? "SA" : m);
    if (m == "ZV" || m == "ZR" || m == "CU" || m == "CD")
      return parseCounterZW(m == "ZV" || m == "CU");
    if (m == "SPA" || m == "SPB" || m == "SPN" || m == "SPBN" || m == "JU")
      return parseJump(m == "SPBN" || m == "JU" ? (m == "JU" ? "SPA" : "SPN") : m);
    if (m == "SAVE") { // SAVE: RLO -> BR (STW bit 8 = marker byte 1 bit 0)
      if (!ensureRlo())
        return false;
      auto br = builder.create<awl::DirectOp>(
          loc(), mlir::scl::DirectArea::Marker, 0, 1, 0, 1);
      builder.create<awl::AssignOp>(loc(), rlo, br);
      if (lexer.getCurToken() == Tok::semi)
        lexer.next();
      rloActive = false;
      return true;
    }
    if (m == "SF") { // S5-era: consume a timer/counter operand, not modelled
      lexer.next();
      if (lexer.getCurToken() == Tok::integer)
        lexer.next();
      if (lexer.getCurToken() == Tok::semi)
        lexer.next();
      return true;
    }
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
        if (!parseSetDbValue())
          return false;
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
    if (m == "SPBNB" || m == "SPBI" || m == "SPBIN" || m == "JNB" ||
        m == "JNBI" || m == "SPO" || m == "SOS" || m == "SPU" || m == "JCN")
      return parseJump("SPN");
    if (m == "JB" || m == "JBI" || m == "JC")
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
    // FB interface -> instance DB (DI, 0x85); FC interface -> the caller's
    // (parent) L frame (VL, 0x87) at the parameter area offset; TEMP -> the
    // block's own L frame (0x86)
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
           c == 'I' || c == 'Q' || c == 'C' || c == 'L';
  }
  static mlir::scl::DirectArea areaOf(char c) {
    switch (c) {
    case 'E': case 'I': return mlir::scl::DirectArea::Input;
    case 'A': case 'Q': return mlir::scl::DirectArea::Output;
    case 'M': return mlir::scl::DirectArea::Marker;
    case 'L': return mlir::scl::DirectArea::Local;
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

  mlir::Value parseOperand(bool bitContext = false) {
    // operand starting with '[': fully indirect U [AR1, P#0.0]
    if (lexer.getCurToken() == Tok::lbracket)
      return parseIndirectBracket(bitContext ? 1 : 16);
    // #name: local variable of the block; with a known interface layout it
    // resolves to the member's offset (instance DB for FB, L area for FC),
    // otherwise a marker bit by name hash
    if (lexer.getCurToken() == Tok::localvar) {
      if (lexer.curPtrLocal())
        return parsePtrLocal();
      std::string name = upperStr(lexer.getIdent());
      lexer.next();
      const MemberInfo *mi = nullptr;
      mlir::scl::DirectArea area = mlir::scl::DirectArea::Marker;
      int32_t byte = 0;
      if (curLayout) {
        auto it = curLayout->members.find(name);
        if (it != curLayout->members.end()) {
          mi = &it->second;
          area = curBlockFb ? mlir::scl::DirectArea::DataBlock
                            : mlir::scl::DirectArea::Local;
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
        // TIMER/COUNTER typed variables live in the T/C areas (the member
        // offset is the timer/counter number)
        if (mi->type == "TIMER")
          area = mlir::scl::DirectArea::Timer;
        else if (mi->type == "COUNTER")
          area = mlir::scl::DirectArea::Counter;
        // array index / UDT sub-member suffixes: #ARR[1].SUB.X
        int32_t bit = mi->bit;
        int32_t width = mi->widthBits ? mi->widthBits : 16;
        std::string type = mi->type;
        int32_t stride = mi->stride;
        while (lexer.getCurToken() == Tok::lbracket ||
               lexer.getCurToken() == Tok::dot) {
          if (lexer.getCurToken() == Tok::lbracket) {
            addBits(byte, bit, consumeArrayIndices(mi));
            if (!mi->elemType.empty())
              type = mi->elemType; // array element type for .sub chains
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
    lexer.next();
    if (quoted) {
      if (raw.size() == 1) // char literal 'A' / latin-1 byte: raw byte value
        return constInt(builder.getI32Type(), (unsigned char)raw[0]);
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

    // __STW <bitname>: individual status-word bit (A0/A1 = CC0/CC1 = 6/7)
    if (w == "__STW") {
      if (lexer.getCurToken() == Tok::ident) { // w's lexer.next() already done
        std::string bitName = upperStr(lexer.getIdent());
        lexer.next();
        int bitNo = -1;
        if (bitName == "A0" || bitName == "CC0") bitNo = 6;
        else if (bitName == "A1" || bitName == "CC1") bitNo = 7;
        else if (bitName == "BIE" || bitName == "BR") bitNo = 8;
        else if (bitName == "NER") bitNo = 0;
        else if (bitName == "VKE") bitNo = 1;
        else if (bitName == "STA") bitNo = 2;
        else if (bitName == "OR") bitNo = 3;
        else if (bitName == "OS") bitNo = 4;
        else if (bitName == "OV") bitNo = 5;
        if (bitNo >= 0)
          return builder.create<awl::DirectOp>(
              loc(), mlir::scl::DirectArea::Marker, 0, bitNo / 8,
              bitNo % 8, 1);
      }
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Marker,
                                            0, 0, 0, 16);
    }
    // STW: status word — approximated by marker word 0
    if (w == "STW")
      return builder.create<awl::DirectOp>(loc(), mlir::scl::DirectArea::Marker,
                                            0, 0, 0, 16);
    // individual STW bits (U OV / = OS ...): NER=0 VKE=1 STA=2 OR=3 OS=4
    // OV=5 A0/CC0=6 A1/CC1=7 BIE/BR=8, kept in marker word 0
    {
      // A0/A1 are deliberately absent: they collide with output-area
      // operands (A1.0); the CC0/CC1 spellings cover the STW bits
      static const std::pair<const char *, int> stwBits[] = {
          {"NER", 0}, {"VKE", 1}, {"RLO", 1}, {"STA", 2}, {"OR", 3},
          {"OS", 4},  {"OV", 5},  {"CC0", 6}, {"CC1", 7}, {"BIE", 8},
          {"BR", 8}};
      for (auto &sb : stwBits)
        if (w == sb.first) {
          // STW bit 8 (BIE/BR) does not fit in byte 0: use byte 1 bit 0
          int32_t byte = sb.second / 8, bit = sb.second % 8;
          return builder.create<awl::DirectOp>(
              loc(), mlir::scl::DirectArea::Marker, 0, byte, bit, 1);
        }
    }
    // DI-area operands (DIX/DIB/DIW/DID): the instance DB in the DI
    // register — db = -1 selects it at runtime
    if (w.rfind("DI", 0) == 0 && w.size() >= 3 &&
        (w[2] == 'X' || w[2] == 'B' || w[2] == 'W' || w[2] == 'D')) {
      int32_t width = w[2]=='X'?1:w[2]=='B'?8:w[2]=='W'?16:32;
      if (lexer.getCurToken() == Tok::lbracket)
        return parseIndirectBracket(width); // DIX [LD 0]
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
        // indices): resolve through the DATA_BLOCK's layout
        int32_t byte = 0, bit = 0, width = bitContext ? 1 : 16;
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
          }
        } else {
          // no DATA_BLOCK declaration: auto-assign the symbolic member in
          // order of first use, sized by the access width
          int32_t key = effDb(db);
          int32_t w = bitContext ? 1 : 16;
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
        while (true) {
          // [i] index?
          if (lexer.getCurToken() == Tok::lbracket)
            addBits(byte, bit, consumeArrayIndices(mi));
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
        if (mi) {
          width = mi->widthBits ? mi->widthBits : width;
          if (width == 1)
            bit = mi->bit;
        }
        if (width != 1 && width != 8 && width != 16 && width != 32 &&
            width != 64)
          width = 32; // UDT/STRING whole-access: dword granularity
        return builder.create<awl::DirectOp>(
            loc(), mlir::scl::DirectArea::DataBlock, effDb(db), byte, bit,
            width);
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
    // i64: indirect bit operand [AR1, P#x.y] (packed pointer); T/C-area
    // directs are S/R on timers/counters (not bit operands, but valid here)
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
      int64_t v = lexer.getInteger();
      int lw = lexer.getLitWidth();
      lexer.next();
      if (neg)
        v = -v;
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
  bool parseSetDbValue() {
    mlir::Value addr;
    if (lexer.getCurToken() == Tok::localvar) {
      addr = parseOperand(false);
      if (!addr)
        return false;
    } else if (lexer.getCurToken() == Tok::lbracket) {
      lexer.next();
      if (lexer.getCurToken() == Tok::ident) {
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
            loc(), mlir::scl::DirectArea::Marker, 0, byte, 0, width);
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
        // indirect load L/LC [AR1, P#x.y] through the packed pointer
        int32_t w = lastIndirectWidth;
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
    if (isPseudoValue(addr)) {
      popAcc(); // store to a pseudo operand: value dropped
    } else if (addr.getType().isInteger(64)) {
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
      return true; // stray ) — ignore
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
    // AW/OW/XOW/UW operate on words (i16), AD/OD/XOD/UD on double words
    // (i32), AB/OB/UB on bytes (i8); U (Und) = AND, O (Oder) = OR, X = XOR
    char w = m.back();
    auto ty = w == 'D' ? builder.getI32Type()
             : w == 'B' ? builder.getI8Type()
                        : builder.getI16Type();
    char op = m[0] == 'U' ? 'A' : m[0];
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
    if (op == 'A')
      r = builder.create<awl::AndOp>(loc(), ty, a, b);
    else if (op == 'O')
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
                     ? zextAcc(builder.create<awl::LoadOp>(loc(), addr))
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
      // status-word/register rhs is not tracked: skip the assert line
      std::string rw = upperStr(lexer.getIdent());
      if (rw == "__STW" || rw == "__DBR" || rw == "__AR") {
        lexer.nextLine();
        return true;
      }
      // memory operand rhs, e.g. __ASSERT== __ACCU 1, MD 8
      mlir::Value addr = parseOperand(false);
      if (!addr) {
        lexer.nextLine();
        return true;
      }
      expected = mlir::isa<mlir::scl::DirectType>(addr.getType())
                     ? zextAcc(builder.create<awl::LoadOp>(loc(), addr))
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

  // RHS of a CALL parameter assignment: constant, block reference
  // (FC 10 as a BLOCK_FC value), boolean, or a plain operand (LW 16, ...).
  // For an operand RHS, *targetOut (if given) receives the operand address:
  // parameters are passed by reference, so an operand RHS is also written
  // back after the call.
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
    return addr;
  }

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
      // system function call: CALL SFC [-]n [(params)]. Parameters are
      // staged in the marker area at kLocalBase (the same staging the FC
      // param copies use) and read/written by scl_rt_sfc in the runtime.
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
      // SFC interfaces: staging offset from kLocalBase, width, direction.
      // ANY params (BVAL/BLK) pass an i64 S7 pointer to the operand.
      struct SfcParam {
        const char *name;
        int32_t byte, width;
        bool out, any;
      };
      // ANY params stage the full 10-byte ANY header inline (BVAL @ +0,
      // BLK @ +10), RET_VAL (i16) @ +20
      static const SfcParam sfc21[] = {{"BVAL", 0, 80, false, true},
                                       {"BLK", 10, 80, false, true},
                                       {"RET_VAL", 20, 16, true, false}};
      static const SfcParam sfc47[] = {{"WT", 0, 16, false, false}};
      static const SfcParam sfc64[] = {{"RET_VAL", 0, 32, true, false}};
      const SfcParam *tab = nullptr;
      int tabLen = 0;
      if (n == 21) {
        tab = sfc21;
        tabLen = 3;
      } else if (n == 47) {
        tab = sfc47;
        tabLen = 1;
      } else if (n == 64) {
        tab = sfc64;
        tabLen = 1;
      }
      llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
      attrs.push_back(builder.getNamedAttr(
          "callee", mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  "SFC" + std::to_string(n))));
      attrs.push_back(
          builder.getNamedAttr("sfc", builder.getI32IntegerAttr((int32_t)n)));
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
          const SfcParam *p = nullptr;
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
              // ANY param: stage the full 10-byte ANY header at kLocalBase.
              // `#anyvar` copies the variable's header; any other operand is
              // wrapped in a synthesized ANY header (awlsim converts operand
              // parameters to ANY pointers the same way).
              bool isAnyVar = false;
              if (lexer.getCurToken() == Tok::localvar) {
                std::string vname = upperStr(lexer.getIdent());
                const MemberInfo *mi = nullptr;
                if (curLayout) {
                  auto it = curLayout->members.find(vname);
                  if (it != curLayout->members.end())
                    mi = &it->second;
                }
                if (!mi && curTempLayout) {
                  auto it = curTempLayout->members.find(vname);
                  if (it != curTempLayout->members.end())
                    mi = &it->second;
                }
                if (mi && mi->type == "ANY")
                  isAnyVar = true;
              }
              mlir::Value a = parseOperand(false);
              if (!a)
                return false;
              auto dt = mlir::dyn_cast<mlir::scl::DirectType>(a.getType());
              if (isAnyVar && dt) {
                for (int i = 0; i < 10; ++i) {
                  auto src = builder.create<awl::DirectOp>(
                      loc(), dt.getArea(), dt.getDb(), dt.getByte() + i, 0, 8);
                  mlir::Value bv = builder.create<awl::LoadOp>(loc(), src);
                  builder.create<awl::StoreOp>(
                      loc(), bv,
                      mkDirect(mlir::scl::DirectArea::Marker, 0,
                               kLocalBase + p->byte + i, 0, 8));
                }
              } else if (dt) {
                int32_t width = dt.getWidth();
                int32_t typeCode = width == 1   ? 0x01
                                   : width == 8 ? 0x02
                                   : width == 16 ? 0x04
                                                 : 0x06;
                int32_t areaS7 =
                    dt.getArea() == mlir::scl::DirectArea::Input    ? 0x81
                    : dt.getArea() == mlir::scl::DirectArea::Output ? 0x82
                    : dt.getArea() == mlir::scl::DirectArea::Marker ? 0x83
                    : dt.getArea() == mlir::scl::DirectArea::Local  ? 0x86
                                                                    : 0x84;
                int32_t db = dt.getArea() == mlir::scl::DirectArea::DataBlock
                                 ? dt.getDb()
                                 : 0;
                if (db == 0 && dt.getArea() == mlir::scl::DirectArea::DataBlock)
                  areaS7 = 0x85; // instance DB
                uint32_t ptr = ((uint32_t)areaS7 << 24) |
                               ((uint32_t)(db & 0xFF) << 16) |
                               ((uint32_t)(dt.getByte() & 0x1FFF) << 3) |
                               (uint32_t)(dt.getBit() & 7);
                auto put = [&](int32_t i, int32_t w, int32_t val) {
                  builder.create<awl::StoreOp>(
                      loc(), constInt(builder.getI32Type(), val),
                      mkDirect(mlir::scl::DirectArea::Marker, 0,
                               kLocalBase + p->byte + i, 0, w));
                };
                put(0, 8, 0x10);     // magic
                put(1, 8, typeCode); // data type
                put(2, 16, 1);       // repetition
                put(4, 16, db);      // DB number
                put(6, 32, (int32_t)ptr); // S7 pointer
              }
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
    std::string callee = kind + std::to_string(num);
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
      return m->type == "STRUCT" ||
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
        auto area = isDbParam ? mlir::scl::DirectArea::DataBlock
                              : mlir::scl::DirectArea::Marker;
        int32_t byte = isDbParam ? paramBase + m->byte : kLocalBase + m->byte;
        int32_t width = m->widthBits ? m->widthBits : 16;
        bool isArray = m->type == "ARRAY" && m->count > 1;
        if (!isOutput) {
          mlir::Value target;
          mlir::Value v = parseCallParamValue(&target);
          if (!v)
            return false;
          if (isArray && target &&
              mlir::isa<mlir::scl::DirectType>(target.getType())) {
            copyArrayParam(m, byte, target, area,
                           isDbParam ? paramDb : 0, true);
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
