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
  tok_dt = -24,
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
  tok_tod = -72,
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
  tok_blockcomment_open = -85, // (*
  tok_blockcomment_close = -86, // *)
  tok_linecomment = -87, // //

  // multi character operators
  tok_assignment = -88, // :=
  tok_exponent = -89, // **
  tok_range = -90, // ..
  tok_cmp_le = -91, // <=
  tok_cmp_ge = -92, // >=
  tok_cmp_ne = -93, // <>

  // names and literals
  tok_identifier = -100,
  tok_symbol = -101,
  tok_integer_literal = -102,
  tok_real_number_literal = -103,
  tok_string_literal = -104,

  // errors
  tok_error_symbol = -901,
  tok_error_integer = -902,
  tok_error_real = -903,
  tok_error_string = -904,
};
llvm::raw_ostream& operator<< (llvm::raw_ostream& s, Token token);

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
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getIdentifier() {
    assert(curTok == tok_identifier);
    return stringValue;
  }

  llvm::StringRef getSymbol() {
    assert(curTok == tok_symbol);
    return stringValue;
  }

  llvm::StringRef getString() {
    assert(curTok == tok_string);
    return stringValue;
  }

  int64_t getIntegerValue() {
    assert(curTok == tok_integer_literal);
    return intVal;
  }

  float getRealNumberValue() {
    assert(curTok == tok_real_number_literal);
    return realVal;
  }

  llvm::StringRef getStringValue() {
    assert(curTok == tok_string_literal);
    return stringValue;
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

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // reserved word or identifier
    if (isalpha(lastChar) || lastChar == '_') {
      stringValue = (char)lastChar;
      std::string lower { (char)tolower(lastChar) };
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_') {
        stringValue += (char)lastChar;
        lower += (char)tolower(lastChar);
      }

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
      if (lower == "dt")
        return tok_dt;
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
      if (lower == "tod")
        return tok_tod;
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

      return tok_identifier;
    }

    if (lastChar == '#') { // identifier, escapes reserved words
      stringValue = "";
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_') {
        stringValue += (char)lastChar;
      }
      return tok_identifier;
    }

    // MARK: strings and symbols

    if (lastChar == '\'') { // String literal
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
          case Token('l'): case Token('L'):
          case Token('p'): case Token('P'):
            stringValue += '\n'; // TODO: check
            break;
          case Token('r'): case Token('R'):
            stringValue += '\r';
            break;
          case Token('t'): case Token('T'):
            stringValue += '\t';
            break;
          case Token('>'): // TODO: check
            while (getNextChar() != '$');
            if (getNextChar() != '<')
              return tok_error_string;
          }
        } else {
          stringValue += c;
        }
      }
      if (c != '\'')
        return tok_error_string;
      return tok_string;
    }

    if (lastChar == '"') { // Symbol
      stringValue = "";
      while (isprint(lastChar = Token(getNextChar())) && lastChar != '"')
        stringValue += (char)lastChar;
      if (lastChar != '"')
        return tok_error_symbol;
      return tok_symbol;
    }

    // MARK: number literals

    if (isdigit(lastChar) || lastChar == '-') { // Number: [0-9.]+
      std::string numStr;

      if (lastChar == '-') {
        numStr = "-";
        lastChar = Token(getNextChar());
        if (!isdigit(lastChar))
          return Token('-');
      }

      enum State { Decimal, Binary, Octal, Hex, Real, Exponent } state = Decimal;
      do {
        if (lastChar == '_')
          lastChar = Token(getNextChar());
        if (isdigit(lastChar)) {
          numStr += lastChar;
        } else if (isxdigit(lastChar) && state == Hex) {
          numStr += lastChar;
        } else if (lastChar == '#' && state == Decimal) {
          if (numStr == "2")
            state = Binary;
          else if (numStr == "8")
            state = Octal;
          else if (numStr == "16")
            state = Hex;
          else
            return tok_error_integer;
          numStr = "";
        } else if (lastChar == '.' && state == Decimal) {
          lastChar = Token(getNextChar());
          if (lastChar == '.') {
            // not a single dot
            lastChar = tok_range;
            break;
          }
          numStr += '.';
          state = Real;
          continue; // we already got the next char
        } else if ((lastChar == 'e' || lastChar == 'E') && (state == Decimal || state == Real)) {
          numStr += 'e';
          state = Exponent;
        }
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.' || lastChar == '_' || lastChar == 'e' || lastChar == 'E');

      switch (state) {
      case Decimal:
        intVal = stoll(numStr);
        return tok_integer_literal;
      case Binary:
        intVal = stoll(numStr, nullptr, 2);
        return tok_integer_literal;
      case Octal:
        intVal = stoll(numStr, nullptr, 8);
        return tok_integer_literal;
      case Hex:
        intVal = stoll(numStr, nullptr, 16);
        return tok_integer_literal;
      case Real:
      case Exponent:
        realVal = stof(numStr, nullptr);
        return tok_real_number_literal;
      }
    }

    // MARK: multi-character operators

    if (lastChar == '.') {
      lastChar = Token(getNextChar());
      if (lastChar == '.') {
        lastChar = Token(getNextChar());
        return tok_range;
      }
      return tok_dot;
    }
    if (lastChar == ':') {
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_assignment;
      }
      return tok_colon;
    }
    if (lastChar == '*') {
      lastChar = Token(getNextChar());
      if (lastChar == '*') {
        lastChar = Token(getNextChar());
        return tok_exponent;
      }
      return tok_times;
    }
    if (lastChar == '>') {
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_cmp_ge;
      }
      return tok_cmp_gt;
    }
    if (lastChar == '<') {
      lastChar = Token(getNextChar());
      if (lastChar == '=') {
        lastChar = Token(getNextChar());
        return tok_cmp_le;
      }
      if (lastChar == '>') {
        lastChar = Token(getNextChar());
        return tok_cmp_ne;
      }
      return tok_cmp_lt;
    }

    // MARK: comments

    if (lastChar == '/') {
      lastChar = Token(getNextChar());
      if (lastChar == '/') { // Comment until end of line.
        do {
          lastChar = Token(getNextChar());
          if (lastChar == EOF)
            return tok_eof;
        } while (lastChar != '\n' && lastChar != '\r');
        return getTok();
      } else
        return Token('/');
    }
    if (lastChar == '(') {
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

  /// If the current Token is an identifier, symbol, or string, this string contains the value.
  std::string stringValue;

  /// If the current Token is a number, this contains the value.
  float realVal = 0;

  int64_t intVal = 0;

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
