//===- lexer.cpp - Helper for printing out the SCL tokens -----------------===//
//
// Copyright 2020 The SCLANG Authors.
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
// This file implements the token dump for the SCL language.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclGen/Lexer.h"

#include "llvm/Support/raw_ostream.h"

namespace sclang {

llvm::raw_ostream &operator<<(llvm::raw_ostream &s, Token token) {
  switch (token) {
  case tok_none:
    return s << "<none>";
  case tok_eof:
    return s << "<eof>";

    // reserved words
  case tok_and:
    return s << "AND";
  case tok_any:
    return s << "ANY";
  case tok_array:
    return s << "ARRAY";
  case tok_begin:
    return s << "BEGIN";
  case tok_block_db:
    return s << "BEGIN_DB";
  case tok_block_fb:
    return s << "BEGIN_FB";
  case tok_block_fc:
    return s << "BLOCK_FC";
  case tok_block_sdb:
    return s << "BLOCK_SDB";
  case tok_bool:
    return s << "BOOL";
  case tok_by:
    return s << "BY";
  case tok_byte:
    return s << "BYTE";
  case tok_case:
    return s << "CASE";
  case tok_char:
    return s << "CHAR";
  case tok_const:
    return s << "CONST";
  case tok_continue:
    return s << "CONTINUE";
  case tok_counter:
    return s << "COUNTER";
  case tok_data_block:
    return s << "DATA_BLOCK";
  case tok_date:
    return s << "DATE";
  case tok_date_and_time:
    return s << "DATE_AND_TIME";
  case tok_dint:
    return s << "DINT";
  case tok_div:
    return s << "DIV";
  case tok_do:
    return s << "DO";
  case tok_dword:
    return s << "DWORD";
  case tok_else:
    return s << "ELSE";
  case tok_elsif:
    return s << "ELSIF";
  case tok_en:
    return s << "EN";
  case tok_eno:
    return s << "ENO";
  case tok_end_case:
    return s << "END_CASE";
  case tok_end_const:
    return s << "END_CONST";
  case tok_end_data_block:
    return s << "END_DATA_BLOCK";
  case tok_end_for:
    return s << "END_FOR";
  case tok_end_function:
    return s << "END_FUNCTION";
  case tok_end_function_block:
    return s << "END_FUNCTION_BLOCK";
  case tok_end_if:
    return s << "END_IF";
  case tok_end_label:
    return s << "END_LABEL";
  case tok_end_type:
    return s << "END_TYPE";
  case tok_end_organization_block:
    return s << "END_ORGANIZATION_BLOCK";
  case tok_end_repeat:
    return s << "END_REPEAT";
  case tok_end_struct:
    return s << "END_STRUCT";
  case tok_end_var:
    return s << "END_VAR";
  case tok_end_while:
    return s << "END_WHILE";
  case tok_exit:
    return s << "EXIT";
  case tok_false:
    return s << "FALSE";
  case tok_for:
    return s << "FOR";
  case tok_function:
    return s << "FUNCTION";
  case tok_function_block:
    return s << "FUNCTION_BLOCK";
  case tok_goto:
    return s << "GOTO";
  case tok_if:
    return s << "IF";
  case tok_int:
    return s << "INT";
  case tok_label:
    return s << "LABEL";
  case tok_mod:
    return s << "MOD";
  case tok_nil:
    return s << "NIL";
  case tok_not:
    return s << "NOT";
  case tok_of:
    return s << "OF";
  case tok_ok:
    return s << "OK";
  case tok_or:
    return s << "OR";
  case tok_organization_block:
    return s << "ORGANIZATION_BLOCK";
  case tok_pointer:
    return s << "POINTER";
  case tok_real:
    return s << "REAL";
  case tok_repeat:
    return s << "REPEAT";
  case tok_return:
    return s << "RETURN";
  case tok_s5time:
    return s << "S5TIME";
  case tok_string:
    return s << "STRING";
  case tok_struct:
    return s << "STRUCT";
  case tok_then:
    return s << "THEN";
  case tok_time:
    return s << "TIME";
  case tok_timer:
    return s << "TIMER";
  case tok_time_of_day:
    return s << "TIME_OF_DAY";
  case tok_to:
    return s << "TO";
  case tok_true:
    return s << "TRUE";
  case tok_type:
    return s << "TYPE";
  case tok_until:
    return s << "UNTIL";
  case tok_var:
    return s << "VAR";
  case tok_var_input:
    return s << "VAR_INPUT";
  case tok_var_in_out:
    return s << "VAR_IN_OUT";
  case tok_var_output:
    return s << "VAR_OUTPUT";
  case tok_var_temp:
    return s << "VAR_TEMP";
  case tok_while:
    return s << "WHILE";
  case tok_word:
    return s << "WORD";
  case tok_void:
    return s << "VOID";
  case tok_xor:
    return s << "XOR";

    // single character tokens
  case tok_colon:
    return s << ':';
  case tok_semicolon:
    return s << ';';
  case tok_comma:
    return s << ',';
  case tok_dot:
    return s << '.';
  case tok_parenthese_open:
    return s << '(';
  case tok_parenthese_close:
    return s << ')';
  case tok_bracket_open:
    return s << '{';
  case tok_bracket_close:
    return s << '}';
  case tok_sbracket_open:
    return s << '[';
  case tok_sbracket_close:
    return s << ']';
  case tok_ampersand:
    return s << '&';
  case tok_percent:
    return s << '%';
  case tok_plus:
    return s << '+';
  case tok_minus:
    return s << '-';
  case tok_times:
    return s << '*';
  case tok_divide:
    return s << '/';
  case tok_cmp_lt:
    return s << '<';
  case tok_cmp_gt:
    return s << '>';
  case tok_cmp_eq:
    return s << '=';

    // comments
  case tok_blockcomment_open:
    return s << "(*";
  case tok_blockcomment_close:
    return s << "*)";
  case tok_linecomment:
    return s << "//";
  case tok_debugprint:
    return s << "DEBUG PRINT";

    // multi character operators
  case tok_assignment:
    return s << ":=";
  case tok_exponent:
    return s << "**";
  case tok_range:
    return s << "..";
  case tok_cmp_le:
    return s << "<=";
  case tok_cmp_ge:
    return s << ">=";
  case tok_cmp_ne:
    return s << "<>";
  case tok_assign_output:
    return s << "=>";

    // names and literals
  case tok_identifier:
    return s << "<identifier>";
  case tok_symbol:
    return s << "<symbol>";
  case tok_integer_literal:
    return s << "<integer literal>";
  case tok_real_number_literal:
    return s << "<real number literal>";
  case tok_string_literal:
    return s << "<string literal>";
  case tok_time_literal:
    return s << "<time literal>";

    // errors
  case tok_error_symbol:
    return s << "<error symbol>";
  case tok_error_integer:
    return s << "<error integer>";
  case tok_error_real:
    return s << "<error real>";
  case tok_error_string:
    return s << "<error string>";
  case tok_error_date:
    return s << "<error date>";
  case tok_error_time:
    return s << "<error time>";
  }
  assert(false);
}

} // namespace sclang
