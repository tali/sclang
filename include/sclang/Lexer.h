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

  // primary
  tok_identifier = -100,
  tok_number = -101,
};

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
    return IdentifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() {
    assert(curTok == tok_number);
    return NumVal;
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
    while (isspace(LastChar))
      LastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // reserved word or identifier
    if (isalpha(LastChar) || LastChar == '_') {
      IdentifierStr = (char)LastChar;
      while (isalnum((LastChar = Token(getNextChar()))) || LastChar == '_')
        IdentifierStr += (char)LastChar;

      if (IdentifierStr == "and")
        return tok_and;
      if (IdentifierStr == "any")
        return tok_any;
      if (IdentifierStr == "array")
        return tok_array;
      if (IdentifierStr == "begin")
        return tok_begin;
      if (IdentifierStr == "block_db")
        return tok_block_db;
      if (IdentifierStr == "block_fb")
        return tok_block_fb;
      if (IdentifierStr == "block_fc")
        return tok_block_fc;
      if (IdentifierStr == "block_sdb")
        return tok_block_sdb;
      if (IdentifierStr == "bool")
        return tok_bool;
      if (IdentifierStr == "by")
        return tok_by;
      if (IdentifierStr == "byte")
        return tok_byte;
      if (IdentifierStr == "case")
        return tok_case;
      if (IdentifierStr == "char")
        return tok_char;
      if (IdentifierStr == "const")
        return tok_const;
      if (IdentifierStr == "continue")
        return tok_continue;
      if (IdentifierStr == "counter")
        return tok_counter;
      if (IdentifierStr == "data_block")
        return tok_data_block;
      if (IdentifierStr == "date")
        return tok_date;
      if (IdentifierStr == "date_and_time")
        return tok_date_and_time;
      if (IdentifierStr == "dint")
        return tok_dint;
      if (IdentifierStr == "div")
        return tok_div;
      if (IdentifierStr == "do")
        return tok_do;
      if (IdentifierStr == "dt")
        return tok_dt;
      if (IdentifierStr == "dword")
        return tok_dword;
      if (IdentifierStr == "else")
        return tok_else;
      if (IdentifierStr == "elsif")
        return tok_elsif;
      if (IdentifierStr == "en")
        return tok_en;
      if (IdentifierStr == "eno")
        return tok_eno;
      if (IdentifierStr == "end_case")
        return tok_end_case;
      if (IdentifierStr == "end_const")
        return tok_end_const;
      if (IdentifierStr == "end_data_block")
        return tok_end_data_block;
      if (IdentifierStr == "end_for")
        return tok_end_for;
      if (IdentifierStr == "end_function")
        return tok_end_function;
      if (IdentifierStr == "end_function_block")
        return tok_end_function_block;
      if (IdentifierStr == "end_if")
        return tok_end_if;
      if (IdentifierStr == "end_label")
        return tok_end_label;
      if (IdentifierStr == "end_type")
        return tok_end_type;
      if (IdentifierStr == "end_organization_block")
        return tok_end_organization_block;
      if (IdentifierStr == "end_repeat")
        return tok_end_repeat;
      if (IdentifierStr == "end_struct")
        return tok_end_struct;
      if (IdentifierStr == "end_var")
        return tok_end_var;
      if (IdentifierStr == "end_while")
        return tok_end_while;
      if (IdentifierStr == "exit")
        return tok_exit;
      if (IdentifierStr == "false")
        return tok_false;
      if (IdentifierStr == "for")
        return tok_for;
      if (IdentifierStr == "function")
        return tok_function;
      if (IdentifierStr == "function_block")
        return tok_function_block;
      if (IdentifierStr == "goto")
        return tok_goto;
      if (IdentifierStr == "if")
        return tok_if;
      if (IdentifierStr == "int")
        return tok_int;
      if (IdentifierStr == "label")
        return tok_label;
      if (IdentifierStr == "mod")
        return tok_mod;
      if (IdentifierStr == "nil")
        return tok_nil;
      if (IdentifierStr == "not")
        return tok_not;
      if (IdentifierStr == "of")
        return tok_of;
      if (IdentifierStr == "ok")
        return tok_ok;
      if (IdentifierStr == "or")
        return tok_or;
      if (IdentifierStr == "organization_block")
        return tok_organization_block;
      if (IdentifierStr == "pointer")
        return tok_pointer;
      if (IdentifierStr == "real")
        return tok_real;
      if (IdentifierStr == "repeat")
        return tok_repeat;
      if (IdentifierStr == "return")
        return tok_return;
      if (IdentifierStr == "s5time")
        return tok_s5time;
      if (IdentifierStr == "string")
        return tok_string;
      if (IdentifierStr == "struct")
        return tok_struct;
      if (IdentifierStr == "then")
        return tok_then;
      if (IdentifierStr == "time")
        return tok_time;
      if (IdentifierStr == "timer")
        return tok_timer;
      if (IdentifierStr == "time_of_day")
        return tok_time_of_day;
      if (IdentifierStr == "to")
        return tok_to;
      if (IdentifierStr == "tod")
        return tok_tod;
      if (IdentifierStr == "true")
        return tok_true;
      if (IdentifierStr == "type")
        return tok_type;
      if (IdentifierStr == "until")
        return tok_until;
      if (IdentifierStr == "var")
        return tok_var;
      if (IdentifierStr == "var_input")
        return tok_var_input;
      if (IdentifierStr == "var_in_out")
        return tok_var_in_out;
      if (IdentifierStr == "var_output")
        return tok_var_output;
      if (IdentifierStr == "var_temp")
        return tok_var_temp;
      if (IdentifierStr == "while")
        return tok_while;
      if (IdentifierStr == "word")
        return tok_word;
      if (IdentifierStr == "void")
        return tok_void;
      if (IdentifierStr == "xor")
        return tok_xor;

      return tok_identifier;
    }

    if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
      std::string NumStr;
      do {
        NumStr += LastChar;
        LastChar = Token(getNextChar());
      } while (isdigit(LastChar) || LastChar == '.');

      NumVal = strtod(NumStr.c_str(), nullptr);
      return tok_number;
    }

    if (LastChar == ':') {
      LastChar = Token(getNextChar());
      if (LastChar == '=') {
        LastChar = Token(getNextChar());
        return tok_assignment;
      }
      return tok_colon;
    }

    if (LastChar == '"') {
      // Symbol until closing "

    }
    if (LastChar == '#') {
      // Comment until end of line.
      do
        LastChar = Token(getNextChar());
      while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

      if (LastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (LastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token ThisChar = Token(LastChar);
    LastChar = Token(getNextChar());
    return ThisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string IdentifierStr;

  /// If the current Token is a number, this contains the value.
  double NumVal = 0;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token LastChar = Token(' ');

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
