//===- Lexer.h - Lexer for the SCL language -------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a simple Lexer for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_LEXER_H_
#define SCLANG_LEXER_H_

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace sclang {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_none = 0,
  tok_eof = -1,

  // reserved words
  tok_and = -2,
  tok_any = -3,
  tok_array = -4,
  tok_begin = -5,
  tok_block_db = -6,
  tok_block_fb = -7,
  tok_block_fc = -8,
  tok_block_sdb = -9,
  tok_bool = -10,
  tok_by = -11,
  tok_byte = -12,
  tok_case = -13,
  tok_char = -14,
  tok_const = -15,
  tok_continue = -16,
  tok_counter = -17,
  tok_data_block = -18,
  tok_date = -19,
  tok_date_and_time = -20,
  tok_dint = -21,
  tok_div = -22,
  tok_do = -23,
  tok_dword = -25,
  tok_else = -26,
  tok_elsif = -27,
  tok_en = -28,
  tok_eno = -29,
  tok_end_case = -30,
  tok_end_const = -31,
  tok_end_data_block = -32,
  tok_end_for = -33,
  tok_end_function = -34,
  tok_end_function_block = -35,
  tok_end_if = -36,
  tok_end_label = -37,
  tok_end_type = -38,
  tok_end_organization_block = -39,
  tok_end_repeat = -40,
  tok_end_struct = -41,
  tok_end_var = -42,
  tok_end_while = -43,
  tok_exit = -44,
  tok_false = -45,
  tok_for = -46,
  tok_function = -47,
  tok_function_block = -48,
  tok_goto = -49,
  tok_if = -50,
  tok_int = -51,
  tok_label = -52,
  tok_mod = -53,
  tok_nil = -54,
  tok_not = -55,
  tok_of = -56,
  tok_ok = -57,
  tok_or = -58,
  tok_organization_block = -59,
  tok_pointer = -60,
  tok_real = -61,
  tok_repeat = -62,
  tok_return = -63,
  tok_s5time = -64,
  tok_string = -65,
  tok_struct = -66,
  tok_then = -67,
  tok_time = -68,
  tok_timer = -69,
  tok_time_of_day = -70,
  tok_to = -71,
  tok_true = -73,
  tok_type = -74,
  tok_until = -75,
  tok_var = -76,
  tok_var_input = -77,
  tok_var_in_out = -78,
  tok_var_output = -79,
  tok_var_temp = -80,
  tok_while = -81,
  tok_word = -82,
  tok_void = -83,
  tok_xor = -84,

  // single character tokens
  tok_colon = ':',
  tok_semicolon = ';',
  tok_comma = ',',
  tok_dot = '.',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',
  tok_ampersand = '&',
  tok_percent = '%',
  tok_plus = '+',
  tok_minus = '-',
  tok_times = '*',
  tok_divide = '/',
  tok_cmp_lt = '<',
  tok_cmp_gt = '>',
  tok_cmp_eq = '=',

  // comments
  tok_blockcomment_open = -85,  // (*
  tok_blockcomment_close = -86, // *)
  tok_linecomment = -87,        // //
  tok_debugprint = -88,         // // DEBUG PRINT

  // multi character operators
  tok_assignment = -90,    // :=
  tok_exponent = -91,      // **
  tok_range = -92,         // ..
  tok_cmp_le = -93,        // <=
  tok_cmp_ge = -94,        // >=
  tok_cmp_ne = -95,        // <>
  tok_assign_output = -96, // =>

  // names and literals
  tok_identifier = -100,
  tok_symbol = -101,
  tok_integer_literal = -102,
  tok_real_number_literal = -103,
  tok_string_literal = -104,
  tok_time_literal = -105,

  // errors
  tok_error_symbol = -901,
  tok_error_integer = -902,
  tok_error_real = -903,
  tok_error_string = -904,
  tok_error_date = -905,
  tok_error_time = -906,
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &s, Token token);

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purpose.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purpose (attaching a location to a Token).
  Lexer(std::string filename)
      : lastLocation(
            {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  /// Skips over comments.
  Token getNextToken() {
    do {
      curTok = getTok();
    } while (curTok == tok_linecomment);
    return curTok;
  }

  /// Move to the next token in the stream and return it.
  /// Even returns comment tokens
  Token getNextTokenWithComments() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Split a negative value token into a minus and a number, then consume the
  /// minus.
  void consumeMinus() {
    assert(isNegativeValue());
    if (curTok == tok_integer_literal)
      intVal = -intVal;
    if (curTok == tok_real_number_literal)
      realVal = -realVal;
    negative = false;
  }

  /// Return `true` if the current token starts with a minus.
  /// The token can either be used as negative value or the minus can be
  /// consumed by `consumeMinus()`.
  bool isNegativeValue() {
    assert(curTok == tok_integer_literal || curTok == tok_real_number_literal);
    return negative;
  }

  /// Return the type of the literal expression, or `tok_none` if there is none.
  /// The type can be specified as prefix, e.g. as `INT#23`.
  Token getLiteralType() {
    assert(curTok == tok_integer_literal || curTok == tok_real_number_literal ||
           curTok == tok_string_literal || curTok == tok_time_literal);
    return literalType;
  }

  /// Return the current identifier, when the current token is an identifier..
  llvm::StringRef getIdentifier() {
    assert(curTok == tok_identifier);
    return stringValue;
  }

  /// Like `getIdentifier()`, but returns current identifier as lower-case
  /// string.
  llvm::StringRef getIdentifierLower() {
    assert(curTok == tok_identifier);
    return stringLowerValue;
  }

  /// Return the name of a symbol, when the current token is a symbol
  llvm::StringRef getSymbol() {
    assert(curTok == tok_symbol);
    return stringValue;
  }

  llvm::StringRef getComment() {
    assert(curTok == tok_linecomment || curTok == tok_debugprint);
    return stringValue;
  }

  /// Return the value of an integer literal.
  int64_t getIntegerValue() {
    assert(curTok == tok_integer_literal);
    return intVal;
  }

  /// Return the value of a real number literal.
  float getRealNumberValue() {
    assert(curTok == tok_real_number_literal);
    return realVal;
  }

  /// Return the value of a number literal, as raw string value.
  llvm::StringRef getStringValue() {
    assert(curTok == tok_string_literal || curTok == tok_real_number_literal);
    return stringValue;
  }

  /// Get the value of a time literal, separated into time components.
  void getTimeValue(int &year, int &mon, int &day, int &hour, int &min,
                    int &sec, int &msec) {
    assert(curTok == tok_time_literal);
    year = yearVal;
    mon = monVal;
    day = dayVal;
    hour = hourVal;
    min = minVal;
    sec = secVal;
    msec = msecVal;
  }
  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n"
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    auto nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  // MARK: reserved words

  /// Return token for reserved word.
  /// If the current token is not a reserved word, then interpret it as an
  /// identifier.
  Token getReservedWordTok() {
    stringValue = (char)lastChar;
    std::string lower{(char)tolower(lastChar)};
    while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_') {
      stringValue += (char)lastChar;
      lower += (char)tolower(lastChar);
    }
    stringLowerValue = lower;

    if (lower == "and")
      return tok_and;
    if (lower == "any")
      return tok_any;
    if (lower == "array")
      return tok_array;
    if (lower == "begin")
      return tok_begin;
    if (lower == "block_db")
      return tok_block_db;
    if (lower == "block_fb")
      return tok_block_fb;
    if (lower == "block_fc")
      return tok_block_fc;
    if (lower == "block_sdb")
      return tok_block_sdb;
    if (lower == "bool")
      return tok_bool;
    if (lower == "by")
      return tok_by;
    if (lower == "byte")
      return tok_byte;
    if (lower == "case")
      return tok_case;
    if (lower == "char")
      return tok_char;
    if (lower == "const")
      return tok_const;
    if (lower == "continue")
      return tok_continue;
    if (lower == "counter")
      return tok_counter;
    if (lower == "data_block")
      return tok_data_block;
    if (lower == "date")
      return tok_date;
    if (lower == "date_and_time")
      return tok_date_and_time;
    if (lower == "dint")
      return tok_dint;
    if (lower == "div")
      return tok_div;
    if (lower == "do")
      return tok_do;
    if (lower == "dword")
      return tok_dword;
    if (lower == "else")
      return tok_else;
    if (lower == "elsif")
      return tok_elsif;
    if (lower == "en")
      return tok_en;
    if (lower == "eno")
      return tok_eno;
    if (lower == "end_case")
      return tok_end_case;
    if (lower == "end_const")
      return tok_end_const;
    if (lower == "end_data_block")
      return tok_end_data_block;
    if (lower == "end_for")
      return tok_end_for;
    if (lower == "end_function")
      return tok_end_function;
    if (lower == "end_function_block")
      return tok_end_function_block;
    if (lower == "end_if")
      return tok_end_if;
    if (lower == "end_label")
      return tok_end_label;
    if (lower == "end_type")
      return tok_end_type;
    if (lower == "end_organization_block")
      return tok_end_organization_block;
    if (lower == "end_repeat")
      return tok_end_repeat;
    if (lower == "end_struct")
      return tok_end_struct;
    if (lower == "end_var")
      return tok_end_var;
    if (lower == "end_while")
      return tok_end_while;
    if (lower == "exit")
      return tok_exit;
    if (lower == "false")
      return tok_false;
    if (lower == "for")
      return tok_for;
    if (lower == "function")
      return tok_function;
    if (lower == "function_block")
      return tok_function_block;
    if (lower == "goto")
      return tok_goto;
    if (lower == "if")
      return tok_if;
    if (lower == "int")
      return tok_int;
    if (lower == "label")
      return tok_label;
    if (lower == "mod")
      return tok_mod;
    if (lower == "nil")
      return tok_nil;
    if (lower == "not")
      return tok_not;
    if (lower == "of")
      return tok_of;
    if (lower == "ok")
      return tok_ok;
    if (lower == "or")
      return tok_or;
    if (lower == "organization_block")
      return tok_organization_block;
    if (lower == "pointer")
      return tok_pointer;
    if (lower == "real")
      return tok_real;
    if (lower == "repeat")
      return tok_repeat;
    if (lower == "return")
      return tok_return;
    if (lower == "s5time")
      return tok_s5time;
    if (lower == "string")
      return tok_string;
    if (lower == "struct")
      return tok_struct;
    if (lower == "then")
      return tok_then;
    if (lower == "time")
      return tok_time;
    if (lower == "timer")
      return tok_timer;
    if (lower == "time_of_day")
      return tok_time_of_day;
    if (lower == "to")
      return tok_to;
    if (lower == "true")
      return tok_true;
    if (lower == "type")
      return tok_type;
    if (lower == "until")
      return tok_until;
    if (lower == "var")
      return tok_var;
    if (lower == "var_input")
      return tok_var_input;
    if (lower == "var_in_out")
      return tok_var_in_out;
    if (lower == "var_output")
      return tok_var_output;
    if (lower == "var_temp")
      return tok_var_temp;
    if (lower == "while")
      return tok_while;
    if (lower == "word")
      return tok_word;
    if (lower == "void")
      return tok_void;
    if (lower == "xor")
      return tok_xor;

    // expand shortcuts for typed literals
    if (lastChar == '#') {
      if (lower == "b")
        return tok_byte;
      if (lower == "i")
        return tok_int;
      if (lower == "d")
        return tok_date;
      if (lower == "l")
        return tok_dint;
      if (lower == "s5t")
        return tok_s5time;
      if (lower == "dt")
        return tok_date_and_time;
      if (lower == "dw")
        return tok_dword;
      if (lower == "t")
        return tok_time;
      if (lower == "tod")
        return tok_time_of_day;
      if (lower == "w")
        return tok_word;
    }

    return tok_identifier;
  }

  /// Return a literal constant based on the given type.
  /// Assumes that we stopped parsing at a `#` character which separates
  /// the type of a literal constant from the value and consumes the entire
  /// literal.
  Token getTypedLiteralTok(Token type) {
    assert(lastChar == Token('#'));
    literalType = type;
    switch (type) {
    case tok_byte:
    case tok_word:
    case tok_dword:
    case tok_int:
    case tok_dint:
      lastChar = Token(getNextChar());
      if (getNumberLiteralTok() != tok_integer_literal)
        return tok_error_integer;
      return tok_integer_literal;
    case tok_real:
      lastChar = Token(getNextChar());
      if (getNumberLiteralTok() != tok_real_number_literal)
        return tok_error_real;
      return tok_real_number_literal;
    case tok_date:
      lastChar = Token(getNextChar());
      return getDateLiteralTok();
    case tok_date_and_time:
      lastChar = Token(getNextChar());
      return getDateAndTimeLiteralTok();
    case tok_s5time:
    case tok_time:
      lastChar = Token(getNextChar());
      return getTimeLiteralTok();
    case tok_time_of_day:
      lastChar = Token(getNextChar());
      return getTimeOfDayLiteralTok();
    default:
      return type;
    }
  }

  // MARK: number literals

  /// Return a number literal, consumes all numeric characters.
  /// The value of the literal can be obtained with `getIntegerValue()`,
  /// `getRealNumberValue()`, or `getStringValue()`.
  Token getNumberLiteralTok() {
    stringValue = "";
    negative = false;

    if (lastChar == '-') {
      stringValue = "-";
      negative = true;
      lastChar = Token(getNextChar());
      if (!isdigit(lastChar))
        return Token('-');
    }

    enum State { Decimal, Binary, Octal, Hex, Real, Exponent } state = Decimal;
    while (true) {
      if (lastChar == '_')
        lastChar = Token(getNextChar());
      if (isdigit(lastChar)) {
        stringValue += lastChar;
      } else if (isxdigit(lastChar) && state == Hex) {
        stringValue += lastChar;
      } else if (lastChar == '#' && state == Decimal) {
        if (stringValue == "2")
          state = Binary;
        else if (stringValue == "8")
          state = Octal;
        else if (stringValue == "16")
          state = Hex;
        else
          return tok_error_integer;
        stringValue = "";
      } else if (lastChar == '.' && state == Decimal) {
        lastChar = Token(getNextChar());
        if (lastChar == '.') {
          // not a single dot
          lastChar = tok_range;
          break;
        }
        stringValue += '.';
        state = Real;
        continue; // we already got the next char
      } else if ((lastChar == 'e' || lastChar == 'E') &&
                 (state == Decimal || state == Real)) {
        stringValue += 'e';
        state = Exponent;
        // the exponent is the only part where a sign is valid within the number
        lastChar = Token(getNextChar());
        if (lastChar == '-' || lastChar == '+') {
          stringValue += lastChar;
          lastChar = Token(getNextChar());
        }
        continue;
      } else {
        // end of number
        break;
      }
      lastChar = Token(getNextChar());
    }

    switch (state) {
    case Decimal:
      if (!llvm::to_integer(stringValue, intVal))
        return tok_error_integer;
      return tok_integer_literal;

    case Binary:
      if (!llvm::to_integer(stringValue, intVal, 2))
        return tok_error_integer;
      return tok_integer_literal;
    case Octal:
      if (!llvm::to_integer(stringValue, intVal, 8))
        return tok_error_integer;
      return tok_integer_literal;
    case Hex:
      if (!llvm::to_integer(stringValue, intVal, 16))
        return tok_error_integer;
      return tok_integer_literal;
    case Real:
    case Exponent:
      if (!llvm::to_float(stringValue, realVal))
        return tok_error_real;
      return tok_real_number_literal;
    }
  }

  // MARK: strings and symbols

  /// Return a string literal.
  /// Consumes all characters belonging to the string. The literal value can be
  /// obtained using `getStringValue`.
  Token getStringLiteralTok() {
    assert(lastChar == '\'');

    stringValue = "";
    char c;
    while (isprint(c = getNextChar()) && c != '\'') {
      if (c == '$') {
        c = getNextChar();
        switch (c) {
        default:
          return tok_error_string;
        case Token('$'):
        case Token('\''):
          stringValue += c;
          break;
        case Token('l'):
        case Token('L'):
        case Token('p'):
        case Token('P'):
          stringValue += '\n'; // TODO: check
          break;
        case Token('r'):
        case Token('R'):
          stringValue += '\r';
          break;
        case Token('t'):
        case Token('T'):
          stringValue += '\t';
          break;
        case Token('>'): // TODO: check
          while (getNextChar() != '$')
            ;
          if (getNextChar() != '<')
            return tok_error_string;
        }
      } else {
        stringValue += c;
      }
    }
    lastChar = Token(getNextChar());
    if (c != '\'')
      return tok_error_string;
    literalType = tok_string;
    return tok_string_literal;
  }

  /// Return a symbol token.
  /// Consumes all characters belonging to the symbol;
  /// the symbol name can be retrieved by `getSymbolValue()`.
  Token getSymbolTok() {
    assert(lastChar == '"');

    stringValue = "";
    while (isprint(lastChar = Token(getNextChar())) && lastChar != '"')
      stringValue += (char)lastChar;
    if (lastChar != '"' || stringValue.empty())
      return tok_error_symbol;
    lastChar = Token(getNextChar());
    return tok_symbol;
  }

  // MARK: date and time literals

  /// Consume a number and write it to `val`.
  bool getNumber(int &val) {
    std::string number;
    while (isdigit(lastChar) || lastChar == Token('_')) {
      if (isdigit(lastChar))
        number += lastChar;
      lastChar = Token(getNextChar());
    }
    return llvm::to_integer(number, val);
  }

  /// Return a date literal token (YYYY-MM-DD).
  Token getDateLiteralTok() {
    if (!getNumber(yearVal) || lastChar != Token('-'))
      return tok_error_date;
    lastChar = Token(getNextChar());
    if (!getNumber(monVal) || lastChar != Token('-'))
      return tok_error_date;
    lastChar = Token(getNextChar());
    if (!getNumber(dayVal))
      return tok_error_date;

    return tok_time_literal;
  }

  /// Return a time literal token (HH:MM:SS[.msec]).
  Token getTimeOfDayLiteralTok() {
    if (!getNumber(hourVal) || lastChar != Token(':'))
      return tok_error_date;
    lastChar = Token(getNextChar());
    if (!getNumber(minVal) || lastChar != Token(':'))
      return tok_error_date;
    lastChar = Token(getNextChar());
    if (!getNumber(secVal))
      return tok_error_date;
    if (lastChar == Token('.')) {
      lastChar = Token(getNextChar());
      getNumber(msecVal);
    }

    return tok_time_literal;
  }

  /// Return a combined date and time literal (YYYY-MM-DD-HH:MM:SS[.msec]).
  Token getDateAndTimeLiteralTok() {
    if (getDateLiteralTok() != tok_time_literal)
      return tok_error_date;
    if (lastChar != Token('-'))
      return tok_error_time;
    lastChar = Token(getNextChar());
    if (getTimeOfDayLiteralTok() != tok_time_literal)
      return tok_error_time;

    return tok_time_literal;
  }

  /// Return a time duration literal ([NNd] [NNh] [NNm] [NNs]).
  Token getTimeLiteralTok() {
    enum { None, Day, Hour, Min, Sec, MSec } pos, lastpos = None;
    while (isdigit(lastChar) || lastChar == Token('_')) {
      int num;
      if (!getNumber(num))
        return tok_error_time;
      int kind = tolower(lastChar);
      lastChar = Token(getNextChar());
      switch (kind) {
      case 'd':
        pos = Day;
        break;
      case 'h':
        pos = Hour;
        break;
      case 'm':
        if (tolower(lastChar) == Token('s')) {
          lastChar = Token(getNextChar());
          pos = MSec;
        } else {
          pos = Min;
        }
        break;
      case 's':
        pos = Sec;
        break;
      default:
        return tok_error_time;
      }
      if (pos < lastpos)
        return tok_error_time;
      switch (pos) {
      case None:
        assert(false);
      case Day:
        dayVal = num;
        break;
      case Hour:
        hourVal = num;
        break;
      case Min:
        minVal = num;
        break;
      case Sec:
        secVal = num;
        break;
      case MSec:
        msecVal = num;
        break;
      }
    }
    return tok_time_literal;
  }

  // MARK: operators and comments

  Token getOperatorTok() {
    switch (lastChar) {
    default:
      assert(false);
    case '.':
      lastChar = Token(getNextChar());
      if (lastChar == '.') {
        lastChar = Token(getNextChar());
        return tok_range;
      }
      return tok_dot;
    case ':':
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_assignment;
      }
      return tok_colon;
    case '*':
      lastChar = Token(getNextChar());
      if (lastChar == '*') {
        lastChar = Token(getNextChar());
        return tok_exponent;
      }
      return tok_times;
    case '>':
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_cmp_ge;
      }
      return tok_cmp_gt;
    case '<':
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_cmp_le;
      } else if (lastChar == '>') {
        lastChar = Token(getNextChar());
        return tok_cmp_ne;
      }
      return tok_cmp_lt;
    case '=':
      lastChar = Token(getNextChar());
      if (lastChar == '>') {
        lastChar = Token(getNextChar());
        return tok_assign_output;
      }
      return tok_cmp_eq;
    case '/':
      lastChar = Token(getNextChar());
      if (lastChar == '/') { // Comment until end of line.
        stringValue = "";
        do {
          lastChar = Token(getNextChar());
          if (lastChar == EOF)
            return tok_eof;
          stringValue += (char)lastChar;
        } while (lastChar != '\n' && lastChar != '\r');
        std::string debugPrefix(" DEBUG PRINT ");
        if (!stringValue.compare(0, debugPrefix.size(), debugPrefix)) {
          stringValue = stringValue.substr(
              debugPrefix.size(), stringValue.size() - debugPrefix.size() - 1);
          return tok_debugprint;
        }
        return tok_linecomment;
      } else
        return Token('/');
    case '(':
      lastChar = Token(getNextChar());
      if (lastChar == '*') { // comment block
        Token last2;
        do {
          last2 = lastChar;
          lastChar = Token(getNextChar());
          if (lastChar == EOF)
            return tok_eof;
        } while (!(last2 == '*' && lastChar == ')'));
        lastChar = Token(getNextChar());
        // end of comment
        return getTok();
      } else { // no comment block, just an opening paranthesis
        return Token('(');
      }
    }
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    literalType = tok_none;
    yearVal = monVal = dayVal = 0;
    hourVal = minVal = secVal = msecVal = 0;

    // reserved word or identifier or typed literal
    if (isalpha(lastChar) || lastChar == '_') {
      Token tok = getReservedWordTok();
      if (lastChar == Token('#')) {
        return getTypedLiteralTok(tok);
      }
      return tok;
    }

    if (lastChar == '#') { // identifier, escapes reserved words
      stringValue = "";
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_') {
        stringValue += (char)lastChar;
      }
      return tok_identifier;
    }

    if (lastChar == '\'') { // String literal
      return getStringLiteralTok();
    }

    if (lastChar == '"') { // Symbol
      return getSymbolTok();
    }

    if (isdigit(lastChar) || lastChar == '-') { // Number: [0-9.]+
      return getNumberLiteralTok();
    }

    if (lastChar == '.' || lastChar == ':' || lastChar == '*' ||
        lastChar == '>' || lastChar == '<' || lastChar == '=' ||
        lastChar == '/' || lastChar == '(') {
      return getOperatorTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token ThisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return ThisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// if current Token is a literal, this stores the type
  Token literalType;

  /// If the current Token is an identifier, symbol, or string, this string
  /// contains the value.
  std::string stringValue;

  /// If the current Token is an identifier, symbol, or string, this string
  /// contains the value in lower case.
  std::string stringLowerValue;

  /// If the current Token is a number literal, then true if that number is
  /// negative.
  bool negative = false;

  /// If the current Token is a number, this contains the value.
  float realVal = 0;

  int64_t intVal = 0;

  // If the current Token is a date or time, these contain the value
  int yearVal = 0;
  int monVal = 0;
  int dayVal = 0;
  int hourVal = 0;
  int minVal = 0;
  int secVal = 0;
  int msecVal = 0;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n')
      ++current;
    if (current <= end && *current)
      ++current;
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }
  const char *current, *end;
};
} // namespace sclang

#endif // SCLANG_LEXER_H_
