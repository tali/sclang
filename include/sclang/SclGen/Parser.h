//===- Parser.h - SCL Language Parser -------------------------------------===//
//
// Copyright 2019 The Sclang Authors.
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
// This file implements the parser for the SCL language. It processes the Token
// provided by the Lexer and returns an AST.
//
// It is based on the Siemens SIMATIC manual 6ES7811-1CA02-8BA0:
// "Structured Control Language (SCL) for S7-300/S7-400, Programming"
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_PARSER_H
#define SCLANG_PARSER_H

#include "sclang/SclGen/AST.h"
#include "sclang/SclGen/Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

namespace sclang {

/// This is a simple recursive parser for the SCL language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> ParseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in this vector.
    std::vector<std::unique_ptr<UnitAST>> units;
    while (auto unit = ParseSCLProgramUnit()) {
      units.push_back(std::move(unit));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(units));
  }

private:
  Lexer &lexer;

  // Appendix B - Lexical Rules
  // MARK: B.1 Identifiers

  std::string ParseIdentifier() {
    if (lexer.getCurToken() != tok_identifier) {
      llvm::errs() << "expected identifier, got " << lexer.getCurToken()
                   << "\n";
      return "";
    }
    std::string identifier(lexer.getIdentifier());
    lexer.consume(tok_identifier);

    return identifier;
  }

  // Appendix C - Syntac Rules
  // MARK: C.1 Subunits of SCL source files

  /// Parse a SCL Program Unit
  /// SCL Program Unit ::= Organization Block | Function | Function block | Data
  /// block | User-defined data type
  std::unique_ptr<UnitAST> ParseSCLProgramUnit() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_organization_block:
      return ParseOrganizationBlock();
    case tok_function:
      return ParseFunction();
    case tok_function_block:
      return ParseFunctionBlock();
    case tok_data_block:
      return ParseDataBlock();
    case tok_type:
      return ParseUDType();
    }
  }

  /// Parse an Organization Block
  ///
  /// Organization Block ::=
  /// "ORGANIZATION_BLOCK" Identifier
  /// Declaration Section
  /// "BEGIN"
  ///   Code Section
  /// "END_ORGANIZATION_BLOCK"
  std::unique_ptr<UnitAST> ParseOrganizationBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_organization_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;
    auto attrs = ParseBlockAttributes();
    auto decls = ParseDeclarationSection();
    if (!decls)
      return nullptr;

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();
    if (!code)
      return parseError<UnitAST>("code block", "in organization block");

    auto unit = std::make_unique<OrganizationBlockAST>(
        std::move(identifier), std::move(loc), std::move(attrs),
        std::move(decls), std::move(code));
    if (lexer.getCurToken() != tok_end_organization_block)
      return parseError<OrganizationBlockAST>(tok_end_organization_block,
                                              "to end organization block");
    lexer.consume(tok_end_organization_block);

    return unit;
  }

  /// Parse a Function
  ///
  /// Function ::=
  /// "FUNCTION" Identifier ":" Type
  /// Declaration Section
  /// "BEGIN"
  ///   Code Section
  /// "END_FUNCTION"
  std::unique_ptr<UnitAST> ParseFunction() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_function);

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;
    if (lexer.getCurToken() != tok_colon)
      return parseError<FunctionAST>(tok_colon, "for function type");
    lexer.consume(tok_colon);

    auto type = ParseDataTypeSpec();
    if (!type)
      return nullptr;
    auto attrs = ParseBlockAttributes();
    auto decls = ParseDeclarationSection();
    if (!decls)
      return nullptr;

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();
    if (!code)
      return parseError<UnitAST>("code block", "in function");

    auto unit = std::make_unique<FunctionAST>(
        std::move(identifier), std::move(loc), std::move(type),
        std::move(attrs), std::move(decls), std::move(code));

    if (lexer.getCurToken() != tok_end_function)
      return parseError<FunctionAST>(tok_end_function, "to end function");
    lexer.consume(tok_end_function);

    return unit;
  }

  /// Parse a Function Block
  ///
  /// Function Block ::=
  /// "FUNCTION_BLOCK" Identifier
  /// Declaration Section
  /// "BEGIN"
  ///   Code Section
  /// "END_FUNCTION_BLOCK"
  std::unique_ptr<UnitAST> ParseFunctionBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_function_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;
    auto attrs = ParseBlockAttributes();
    auto decls = ParseDeclarationSection();

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();
    if (!code)
      return parseError<UnitAST>("code block", "in function block");

    auto unit = std::make_unique<FunctionBlockAST>(
        std::move(identifier), std::move(loc), std::move(attrs),
        std::move(decls), std::move(code));

    if (lexer.getCurToken() != tok_end_function_block)
      return parseError<FunctionBlockAST>(tok_end_function_block,
                                          "to end function block");
    lexer.consume(tok_end_function_block);

    return unit;
  }

  /// Parse a Data Block
  ///
  /// Data Block ::=
  /// "DATA_BLOCK" Identifier
  /// Declaration Section
  /// "BEGIN"
  /// DB Assignments Section
  /// "END_DATA_BLOCK"
  std::unique_ptr<UnitAST> ParseDataBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_data_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;
    auto attrs = ParseBlockAttributes();
    auto type = ParseDataTypeSpec();
    if (!type)
      return parseError<UnitAST>("data type", "for data block");

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto assignments = ParseCodeSection();

    auto nodecls = NoDeclarationSection(loc);
    auto unit = std::make_unique<DataBlockAST>(
        std::move(identifier), std::move(loc), std::move(attrs),
        std::move(nodecls), std::move(type), std::move(assignments));

    if (lexer.getCurToken() != tok_end_data_block)
      return parseError<DataBlockAST>(tok_end_data_block, "to end data block");
    lexer.consume(tok_end_data_block);

    return unit;
  }

  /// Parse a User-Defined Data Type
  ///
  /// User-Defined Data Block ::=
  /// "TYPE" Identifier
  /// Struct Data Type Specification
  /// "END_TYPE"
  std::unique_ptr<UnitAST> ParseUDType() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_type);

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;
    auto type = ParseDataTypeSpec(); // TODO: STRUCT
    auto attrs = ParseBlockAttributes();

    auto nodecls = NoDeclarationSection(loc);
    auto block = std::make_unique<UserDefinedTypeAST>(
        std::move(identifier), std::move(loc), std::move(attrs),
        std::move(nodecls), std::move(type));

    if (lexer.getCurToken() != tok_end_type)
      return parseError<UserDefinedTypeAST>(tok_end_type,
                                            "to end user defined type");
    lexer.consume(tok_end_type);

    return block;
  }

  std::unique_ptr<AttributeAST> ParseAttributeValue(Token tok) {
    auto loc = lexer.getLastLocation();

    std::string name(lexer.getIdentifierLower());
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != tok)
      return parseError<AttributeAST>(tok, "attribute");
    lexer.consume(tok);

    std::string value;
    switch (lexer.getCurToken()) {
    default:
      return parseError<AttributeAST>(tok_string_literal, "attribute");
    case tok_identifier:
      value = std::string(lexer.getIdentifier());
      lexer.consume(tok_identifier);
      break;
    case tok_string_literal:
      value = std::string(lexer.getStringValue());
      lexer.consume(tok_string_literal);
      break;
    case tok_real_number_literal:
      value = std::string(lexer.getStringValue());
      lexer.consume(tok_real_number_literal);
      break;
    }

    return std::make_unique<AttributeAST>(std::move(loc), std::move(name),
                                          std::move(value));
  }

  /// Block  Attribute ::= ("AUTHOR" | "FAMILY" | "NAME" | "FAMILY") ":" (String
  /// Literal | Identifier) | "KNOW_HOW_PROTECT"
  std::unique_ptr<AttributeAST> ParseBlockAttribute() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_identifier)
      return nullptr;
    std::string name(lexer.getIdentifierLower());
    std::string value;
    if (name == "title") {
      return ParseAttributeValue(tok_cmp_eq);
    } else if (name == "know_how_protect") {
      lexer.consume(tok_identifier);
      return std::make_unique<AttributeAST>(std::move(loc), std::move(name),
                                            "1");
    } else if (name == "author" || name == "family" || name == "name" ||
               name == "version") {
      return ParseAttributeValue(tok_colon);
    }

    return nullptr;
  }

  /// Parse the block attribues
  ///
  /// Block Attributes ::= { Block Attribute }
  std::vector<std::unique_ptr<AttributeAST>> ParseBlockAttributes() {
    std::vector<std::unique_ptr<AttributeAST>> attrs;

    while (auto attr = ParseBlockAttribute())
      attrs.push_back(std::move(attr));

    return attrs;
  }

  // MARK: C.2 Structure of Declaration Sections

  std::unique_ptr<DeclarationSectionAST> NoDeclarationSection(Location loc) {
    std::vector<std::unique_ptr<DeclarationSubsectionAST>> subsections;
    return std::make_unique<DeclarationSectionAST>(loc, std::move(subsections));
  }

  /// Parse a declaration section
  ///
  /// Declaration Sections ::= { Declaration Subsection }
  std::unique_ptr<DeclarationSectionAST> ParseDeclarationSection() {
    auto loc = lexer.getLastLocation();

    // Parse subsections one at a time and accumulate in this vector.
    std::vector<std::unique_ptr<DeclarationSubsectionAST>> subsections;
    while (auto unit = ParseDeclarationSubsection()) {
      subsections.push_back(std::move(unit));
    }

    return std::make_unique<DeclarationSectionAST>(std::move(loc),
                                                   std::move(subsections));
  }

  /// Parse a subsection of the declaration section
  ///
  /// Declaration Subsection ::= Constant Subsection | Jump Label Subsection
  ///  | Temporary Variable Subsection | Static Variable Subsection | Parameters
  ///  Subsection
  std::unique_ptr<DeclarationSubsectionAST> ParseDeclarationSubsection() {
    switch (lexer.getCurToken()) {
    default:
      return nullptr;
    case tok_semicolon:
      lexer.consume(lexer.getCurToken());
      return ParseDeclarationSubsection();
    case tok_const:
      return ParseConstantSubsection();
    case tok_label:
      return ParseJumpLabelSubsection();
    case tok_var:
    case tok_var_temp:
    case tok_var_input:
    case tok_var_output:
    case tok_var_in_out:
      return ParseVariableSubsection();
      break;
    }
  }

  /// Parse a Constant Declaration
  ///
  /// Constant Declaration = Identifier ":=" Simple Expression ";"
  std::unique_ptr<ConstantDeclarationAST> ParseConstantDeclaration() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;

    if (lexer.getCurToken() != tok_assignment)
      return parseError<ConstantDeclarationAST>(tok_assignment,
                                                "for constant assignment");
    lexer.consume(tok_assignment);

    auto value = ParseExpression();
    if (!value)
      return nullptr;

    if (lexer.getCurToken() != tok_semicolon)
      return parseError<ConstantDeclarationAST>(tok_semicolon,
                                                "to end constant declaration");
    lexer.consume(tok_semicolon);

    return std::make_unique<ConstantDeclarationAST>(
        std::move(loc), std::move(identifier), std::move(value));
  }

  /// Parse a Constant Subsection
  ///
  /// Constant Subsection ::=
  /// "CONST"
  /// { Constant Declaration }
  /// "END_CONST"
  std::unique_ptr<ConstantDeclarationSubsectionAST> ParseConstantSubsection() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_const);

    std::vector<std::unique_ptr<ConstantDeclarationAST>> consts;
    while (lexer.getCurToken() != tok_end_const) {
      auto constant = ParseConstantDeclaration();
      if (!constant)
        break;
      consts.push_back(std::move(constant));
    }
    if (lexer.getCurToken() != tok_end_const)
      return parseError<ConstantDeclarationSubsectionAST>(
          tok_end_const, "to end constant subsection");
    lexer.consume(tok_end_const);

    return std::make_unique<ConstantDeclarationSubsectionAST>(
        loc, std::move(consts));
  }

  /// Parse a Jump Label Declaration
  ///
  /// Jump Label Declaration = Identifier ";"
  std::unique_ptr<JumpLabelDeclarationAST> ParseJumpLabelDeclaration() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return nullptr;

    if (lexer.getCurToken() != tok_semicolon)
      return parseError<JumpLabelDeclarationAST>(tok_semicolon,
                                                 "to end constant declaration");
    lexer.consume(tok_semicolon);

    return std::make_unique<JumpLabelDeclarationAST>(std::move(loc),
                                                     std::move(identifier));
  }

  /// Parse a jump label subsection
  ///
  /// Jump Label Subsection ::=
  /// "LABEL"
  /// { Label Declaration }
  /// "END_LABEL"
  std::unique_ptr<JumpLabelDeclarationSubsectionAST>
  ParseJumpLabelSubsection() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_label);

    std::vector<std::unique_ptr<JumpLabelDeclarationAST>> consts;
    while (lexer.getCurToken() != tok_end_label) {
      auto constant = ParseJumpLabelDeclaration();
      if (!constant)
        break;
      consts.push_back(std::move(constant));
    }
    if (lexer.getCurToken() != tok_end_label)
      return parseError<JumpLabelDeclarationSubsectionAST>(
          tok_end_label, "to end label declaration subsection");
    lexer.consume(tok_end_label);

    return std::make_unique<JumpLabelDeclarationSubsectionAST>(
        loc, std::move(consts));
  }

  /// Parse a  Variable Subsection
  ///
  /// Variable Subsection ::=
  /// ("VAR" | "VAR_TEMP" | "VAR_INPUT" | "VAR_OUTPUT" | "VAR_IN_OUT" )
  /// { Variable Declaration }
  /// "END_VAR"
  std::unique_ptr<VariableDeclarationSubsectionAST> ParseVariableSubsection() {
    auto loc = lexer.getLastLocation();
    VariableDeclarationSubsectionAST::Var_Kind kind;
    switch (lexer.getCurToken()) {
    default:
      assert(false);
    case tok_var:
      kind = VariableDeclarationSubsectionAST::Var;
      lexer.consume(tok_var);
      break;
    case tok_var_temp:
      kind = VariableDeclarationSubsectionAST::VarTemp;
      lexer.consume(tok_var_temp);
      break;
    case tok_var_input:
      kind = VariableDeclarationSubsectionAST::VarInput;
      lexer.consume(tok_var_input);
      break;
    case tok_var_output:
      kind = VariableDeclarationSubsectionAST::VarOutput;
      lexer.consume(tok_var_output);
      break;
    case tok_var_in_out:
      kind = VariableDeclarationSubsectionAST::VarInOut;
      lexer.consume(tok_var_in_out);
      break;
    }

    std::vector<std::unique_ptr<VariableDeclarationAST>> vars;
    while (lexer.getCurToken() != tok_end_var) {
      auto var = ParseVariableDeclaration();
      if (!var)
        break;
      vars.push_back(std::move(var));
    }
    if (lexer.getCurToken() != tok_end_var)
      return parseError<VariableDeclarationSubsectionAST>(
          tok_end_var, "to end variable declaration subsection");
    lexer.consume(tok_end_var);

    return std::make_unique<VariableDeclarationSubsectionAST>(loc, kind,
                                                              std::move(vars));
  }

  /// Parse the identifier and system attributes of a Variable Declaration
  ///
  /// Variable Identifier ::= Identifier [ "{" System Attribute { ";" System
  /// Attribute  } "}" ] System Attribute ::= Identifier  ":=" Symbol
  std::unique_ptr<VariableIdentifierAST> ParseVariableIdentifier() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return parseError<VariableIdentifierAST>(tok_identifier,
                                               "as variable name");
    std::vector<std::unique_ptr<AttributeAST>> attributes;

    if (lexer.getCurToken() == tok_bracket_open) {
      lexer.consume(tok_bracket_open);
      // TODO: parse attributes
      lexer.consume(tok_bracket_close);
    }
    return std::make_unique<VariableIdentifierAST>(
        std::move(loc), std::move(identifier), std::move(attributes));
  }

  /// Parse a Variable Declaration or Instance Declaration
  ///
  /// Variable Declaration ::= Variable Identifier { "," Variable Identifier }
  /// ":=" Data Type Specification [ Data Type Initialization ] ";"
  std::unique_ptr<VariableDeclarationAST> ParseVariableDeclaration() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseVariableIdentifier();
    if (!identifier)
      return nullptr;
    std::vector<std::unique_ptr<VariableIdentifierAST>> vars;
    vars.push_back(std::move(identifier));
    while (lexer.getCurToken() == tok_comma) {
      lexer.consume(tok_comma);

      auto identifier = ParseVariableIdentifier();
      if (!identifier)
        break;
      vars.push_back(std::move(identifier));
    }

    if (lexer.getCurToken() != tok_colon)
      return parseError<VariableDeclarationAST>(tok_colon, "for variable type");
    lexer.consume(tok_colon);

    auto type = ParseDataTypeSpec();
    if (!type)
      return nullptr;

    llvm::Optional<std::unique_ptr<ExpressionAST>> init;
    if (lexer.getCurToken() == tok_assignment) {
      lexer.consume(tok_assignment);

      init = ParseExpression(pred_list);
    }

    if (lexer.getCurToken() != tok_semicolon)
      return parseError<VariableDeclarationAST>(tok_semicolon,
                                                "to end variable declaration");
    lexer.consume(tok_semicolon);

    return std::make_unique<VariableDeclarationAST>(
        std::move(loc), std::move(vars), std::move(type), std::move(init));
  }

  // Instance Declaration is parsed as Variable Declaration

  // Temporary Variable Subsection and Parameter Subsection is parsed as
  // Variable Subsection

  /// Parse a Data Type Specification
  ///
  /// Data Type Specification ::= TBD
  std::unique_ptr<DataTypeSpecAST> ParseDataTypeSpec() {
    auto loc = lexer.getLastLocation();

    ElementaryDataTypeAST::ElementaryTypeASTKind elementaryType;

    // TODO: TBD
    switch (lexer.getCurToken()) {
    default:
      return parseError<DataTypeSpecAST>("data type");
    case tok_void:
      elementaryType = ElementaryDataTypeAST::Type_Void;
      break;
      // Bit Data Type
    case tok_bool:
      elementaryType = ElementaryDataTypeAST::Type_Bool;
      break;
    case tok_byte:
      elementaryType = ElementaryDataTypeAST::Type_Byte;
      break;
    case tok_word:
      elementaryType = ElementaryDataTypeAST::Type_Word;
      break;
    case tok_dword:
      elementaryType = ElementaryDataTypeAST::Type_DWord;
      break;
    // Character Data Type
    case tok_char:
      elementaryType = ElementaryDataTypeAST::Type_Char;
      break;
    // Numerical Data Type
    case tok_int:
      elementaryType = ElementaryDataTypeAST::Type_Int;
      break;
    case tok_dint:
      elementaryType = ElementaryDataTypeAST::Type_DInt;
      break;
    case tok_real:
      elementaryType = ElementaryDataTypeAST::Type_Real;
      break;
    // Time Type
    case tok_s5time:
      elementaryType = ElementaryDataTypeAST::Type_S5Time;
      break;
    case tok_time:
      elementaryType = ElementaryDataTypeAST::Type_Time;
      break;
    case tok_time_of_day:
      elementaryType = ElementaryDataTypeAST::Type_TimeOfDay;
      break;
    case tok_date:
      lexer.consume(tok_word);
      elementaryType = ElementaryDataTypeAST::Type_Date;
      break;
    // String data type
    case tok_string:
      return ParseStringDataTypeSpec();
    // complex types
    case tok_date_and_time:
      elementaryType = ElementaryDataTypeAST::Type_DateAndTime;
      break;
    case tok_array:
      return ParseArrayDataTypeSpec();
    case tok_struct:
      return ParseStructDataTypeSpec();
    // user-defined type
    case tok_identifier:
      return ParseUserDefinedDataType();
    // parameter types
    case tok_timer:
      elementaryType = ElementaryDataTypeAST::Type_Timer;
      break;
    case tok_counter:
      elementaryType = ElementaryDataTypeAST::Type_Counter;
      break;
    case tok_any:
      elementaryType = ElementaryDataTypeAST::Type_Any;
      break;
    case tok_pointer:
      elementaryType = ElementaryDataTypeAST::Type_Pointer;
      break;
    case tok_block_fb:
      elementaryType = ElementaryDataTypeAST::Type_BlockFB;
      break;
    case tok_block_fc:
      elementaryType = ElementaryDataTypeAST::Type_BlockFC;
      break;
    case tok_block_db:
      elementaryType = ElementaryDataTypeAST::Type_BlockDB;
      break;
    case tok_block_sdb:
      elementaryType = ElementaryDataTypeAST::Type_BlockSDB;
      break;
    }
    lexer.getNextToken();

    return std::make_unique<ElementaryDataTypeAST>(std::move(loc),
                                                   elementaryType);
  }

  // MARK: C.3 Data Types in SCL

  /// Parse a string data type
  ///
  /// String Data Type Specification ::= "STRING" [ "[" Simple Expression "]" ]
  std::unique_ptr<StringDataTypeSpecAST> ParseStringDataTypeSpec() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_string);

    uint8_t length = 254;

    if (lexer.getCurToken() == tok_sbracket_open) {
      lexer.consume(tok_sbracket_open);

      auto len = ParseExpression();
      if (len && llvm::isa<IntegerConstantAST>(len)) {
        length = llvm::cast<IntegerConstantAST>(len.get())->getValue();
      } else {
        return parseError<StringDataTypeSpecAST>("integer constant",
                                                 "for string type");
      }

      if (lexer.getCurToken() != tok_sbracket_close)
        return parseError<StringDataTypeSpecAST>(tok_integer_literal,
                                                 "for string type");
      lexer.consume(tok_sbracket_close);
    }

    return std::make_unique<StringDataTypeSpecAST>(std::move(loc),
                                                   std::move(length));
  }

  /// Parse a User Defined Data Type
  ///
  /// User-Defined Data Type ::= Identifier
  std::unique_ptr<UserDefinedTypeIdentifierAST> ParseUserDefinedDataType() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return parseError<UserDefinedTypeIdentifierAST>("identifier",
                                                      "for user defined type");

    return std::make_unique<UserDefinedTypeIdentifierAST>(std::move(loc),
                                                          identifier);
  }

  /// Parse an Array Dimension
  ///
  ///  Array Dimension ::= Expression ".." Expression
  std::unique_ptr<ArrayDimensionAST> ParseArrayDimension() {
    auto loc = lexer.getLastLocation();

    auto min = ParseExpression();

    if (lexer.getCurToken() != tok_range)
      return parseError<ArrayDimensionAST>(tok_range, "for array dimension");
    lexer.consume(tok_range);

    auto max = ParseExpression();

    return std::make_unique<ArrayDimensionAST>(std::move(loc), std::move(min),
                                               std::move(max));
  }

  /// Parse an Array Data Type Specification
  ///
  /// Array Data Type Specification ::= "ARRAY" "[" Index ".." Index { "," Index
  /// ".." Index } "]" "OF" Data Type Specification
  std::unique_ptr<ArrayDataTypeSpecAST> ParseArrayDataTypeSpec() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_array);

    if (lexer.getCurToken() != tok_sbracket_open)
      return parseError<ArrayDataTypeSpecAST>(tok_sbracket_open,
                                              "for array dimensions");
    lexer.consume(tok_sbracket_open);

    std::vector<std::unique_ptr<ArrayDimensionAST>> dimensions;
    auto dim = ParseArrayDimension();
    if (!dim)
      return nullptr;
    dimensions.push_back(std::move(dim));

    while (lexer.getCurToken() == tok_comma) {
      lexer.consume(tok_comma);

      auto dim = ParseArrayDimension();
      if (!dim)
        return nullptr;
      dimensions.push_back(std::move(dim));
    }

    if (lexer.getCurToken() != tok_sbracket_close)
      return parseError<ArrayDataTypeSpecAST>(tok_sbracket_close,
                                              "for array dimensions");
    lexer.consume(tok_sbracket_close);

    if (lexer.getCurToken() != tok_of)
      return parseError<ArrayDataTypeSpecAST>(tok_of, "for array dimensions");
    lexer.consume(tok_of);

    auto dataType = ParseDataTypeSpec();

    return std::make_unique<ArrayDataTypeSpecAST>(
        std::move(loc), std::move(dimensions), std::move(dataType));
  }

  /// Parse a Struct Component Declaration
  ///
  /// Struct Component Declaration ::= Identifier ":" Data Type Specification  [
  /// Data Type Initialization ] ";"
  std::unique_ptr<ComponentDeclarationAST> ParseComponentDeclaration() {
    auto loc = lexer.getLastLocation();

    auto name = ParseIdentifier();

    if (lexer.getCurToken() != tok_colon)
      return parseError<ComponentDeclarationAST>(
          tok_colon, "for struct component declaration");
    lexer.consume(tok_colon);

    auto dataType = ParseDataTypeSpec();

    llvm::Optional<std::unique_ptr<ExpressionAST>> init;
    if (lexer.getCurToken() == tok_assignment) {
      lexer.consume(tok_assignment);

      init = ParseExpression(pred_list);
    }

    if (lexer.getCurToken() != tok_semicolon)
      return parseError<ComponentDeclarationAST>(
          tok_semicolon, "for struct component declaration");
    lexer.consume(tok_semicolon);

    return std::make_unique<ComponentDeclarationAST>(
        std::move(loc), std::move(name), std::move(dataType), std::move(init));
  }

  /// Parse a Struct Data Type Specification
  ///
  /// Struct Data Type Specification ::= "STRUCT" { Component Declaration }
  /// "END_STRUCT"
  std::unique_ptr<StructDataTypeSpecAST> ParseStructDataTypeSpec() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_struct);

    std::vector<std::unique_ptr<ComponentDeclarationAST>> components;
    do {
      auto component = ParseComponentDeclaration();
      if (!component)
        return parseError<StructDataTypeSpecAST>("component declaration",
                                                 "for struct specification");
      components.push_back(std::move(component));
    } while (lexer.getCurToken() != tok_end_struct);
    lexer.consume(tok_end_struct);

    return std::make_unique<StructDataTypeSpecAST>(std::move(loc),
                                                   std::move(components));
  }

  // MARK: C.4 Code Section

  /// end of current code section?
  bool isEndToken(Token tok) {
    switch (tok) {
    default:
      return false;
    case tok_eof:
    case tok_else:
    case tok_elsif:
    case tok_end_case:
    case tok_end_const:
    case tok_end_data_block:
    case tok_end_for:
    case tok_end_function:
    case tok_end_function_block:
    case tok_end_if:
    case tok_end_label:
    case tok_end_type:
    case tok_end_organization_block:
    case tok_end_repeat:
    case tok_end_struct:
    case tok_end_var:
    case tok_end_while:
    case tok_until:
      return true;
    }
  }
  /// Parse a Code Section
  ///
  /// Code Section ::= { { Identifier ":" }  [ Instruction ";" ] }
  std::unique_ptr<CodeSectionAST> ParseCodeSection() {
    auto loc = lexer.getLastLocation();

    std::vector<std::unique_ptr<InstructionAST>> instructions;
    while (!isEndToken(lexer.getCurToken())) {
      if (lexer.getCurToken() == tok_semicolon) {
        lexer.consume(tok_semicolon);
        continue;
      }

      auto instr = ParseInstruction();
      if (!instr)
        return parseError<CodeSectionAST>("instruction", "in code section");

      if (lexer.getCurToken() != tok_semicolon)
        return parseError<CodeSectionAST>(tok_semicolon,
                                          "to end the instruction");
      lexer.consume(tok_semicolon);

      instructions.push_back(std::move(instr));
    }

    return std::make_unique<CodeSectionAST>(std::move(loc),
                                            std::move(instructions));
  }

  std::unique_ptr<InstructionAST> ParseInstruction() {
    switch (lexer.getCurToken()) {
    default:
      return ParseStatement();
    case tok_if:
      return ParseIfThenElse();
    case tok_case:
      return ParseCaseOf();
    case tok_for:
      return ParseForDo();
    case tok_while:
      return ParseWhileDo();
    case tok_repeat:
      return ParseRepeatUntil();
    case tok_continue:
      return ParseContinue();
    case tok_return:
      return ParseReturn();
    case tok_exit:
      return ParseExit();
    case tok_goto:
      return ParseGoto();
    }
  }

  std::unique_ptr<InstructionAST> ParseStatement() {
    auto loc = lexer.getLastLocation();

    auto expr = ParseExpression();
    if (!expr)
      return parseError<InstructionAST>("expression", "in a statement");

    auto binary = llvm::dyn_cast<BinaryExpressionAST>(expr.get());
    if (binary) {
      if (binary->getOp() != tok_assignment)
        return parseError<InstructionAST>("assignment",
                                          "as top-level expression");
      return std::make_unique<ValueAssignmentAST>(std::move(loc),
                                                  std::move(expr));
    }

    auto call = llvm::dyn_cast<FunctionCallAST>(expr.get());
    if (call) {
      return std::make_unique<SubroutineProcessingAST>(std::move(loc),
                                                       std::move(expr));
    }

    // TODO: jump labels

    return parseError<InstructionAST>("instruction", "in code block");
  }

  // MARK: C.5 Value Assignments

  std::unique_ptr<ExpressionAST> ParseExpressionPrimary() {
    switch (lexer.getCurToken()) {
    default:
      return parseError<ExpressionAST>("expression");
    case tok_parenthese_open:
      return ParseParenExpression();
    case tok_not:
    case tok_plus:
    case tok_minus:
      return ParseUnaryExpression();
    case tok_identifier:
      return ParseIdentifierExpression();
    case tok_symbol:
      return ParseSymbolExpression();
    case tok_false:
    case tok_true:
      return ParseBooleanLiteralExpression();
    case tok_integer_literal:
      return ParseIntegerLiteralExpression();
    case tok_real_number_literal:
      return ParseRealNumberLiteralExpression();
    case tok_string_literal:
      return ParseStringLiteralExpression();
    case tok_time_literal:
      return ParseTimeLiteralExpression();
    }
  }

  std::unique_ptr<ExpressionAST> ParseParenExpression() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_parenthese_open);
    auto expr = ParseExpression();
    if (!expr)
      return nullptr;

    if (lexer.getCurToken() != tok_parenthese_close)
      return parseError<ExpressionAST>(tok_parenthese_close,
                                       "to close expression with parentheses");
    lexer.consume(tok_parenthese_close);

    return expr;
  }

  std::unique_ptr<ExpressionAST> ParseUnaryExpression() {
    auto loc = lexer.getLastLocation();

    auto token = lexer.getCurToken();
    lexer.consume(token);

    auto expr = ParseExpression(pred_unary);
    if (!expr)
      return parseError<ExpressionAST>("expression", "after unary operator");

    return std::make_unique<UnaryExpressionAST>(std::move(loc), token,
                                                std::move(expr));
  }

  std::unique_ptr<ExpressionAST> ParseIdentifierExpression() {
    auto loc = lexer.getLastLocation();

    std::string variable(lexer.getIdentifier());
    if (variable.empty())
      return parseError<ExpressionAST>("variable", "for expression");
    lexer.consume(tok_identifier);

    return std::make_unique<SimpleVariableAST>(std::move(loc),
                                               std::move(variable), false);
  }

  std::unique_ptr<ExpressionAST> ParseSymbolExpression() {
    auto loc = lexer.getLastLocation();

    std::string name(lexer.getSymbol());
    if (name.empty())
      return parseError<ExpressionAST>("name", "for expression");
    lexer.consume(tok_symbol);

    return std::make_unique<SimpleVariableAST>(std::move(loc), std::move(name),
                                               true);
  }

  std::unique_ptr<ExpressionAST> ParseBooleanLiteralExpression() {
    auto loc = lexer.getLastLocation();

    switch (lexer.getCurToken()) {
    default:
      assert(false);
    case tok_false:
      lexer.consume(tok_false);
      return std::make_unique<IntegerConstantAST>(std::move(loc), 0, tok_bool);
    case tok_true:
      lexer.consume(tok_true);
      return std::make_unique<IntegerConstantAST>(std::move(loc), 1, tok_bool);
    }
  }

  std::unique_ptr<ConstantAST> ParseIntegerLiteralExpression() {
    auto loc = lexer.getLastLocation();

    auto value = lexer.getIntegerValue();
    Token type = lexer.getLiteralType();
    lexer.consume(tok_integer_literal);

    return std::make_unique<IntegerConstantAST>(std::move(loc), value, type);
  }

  std::unique_ptr<ConstantAST> ParseStringLiteralExpression() {
    auto loc = lexer.getLastLocation();

    auto value = lexer.getStringValue();
    Token type = lexer.getLiteralType();
    lexer.consume(tok_string_literal);

    return std::make_unique<StringConstantAST>(std::move(loc), value, type);
  }

  std::unique_ptr<ConstantAST> ParseRealNumberLiteralExpression() {
    auto loc = lexer.getLastLocation();

    auto value = lexer.getRealNumberValue();
    Token type = lexer.getLiteralType();
    lexer.consume(tok_real_number_literal);

    return std::make_unique<RealConstantAST>(std::move(loc), value, type);
  }

  std::unique_ptr<ConstantAST> ParseTimeLiteralExpression() {
    auto loc = lexer.getLastLocation();

    int year, mon, day, hour, min, sec, msec;
    lexer.getTimeValue(year, mon, day, hour, min, sec, msec);
    Token type = lexer.getLiteralType();
    lexer.consume(tok_time_literal);

    return std::make_unique<TimeConstantAST>(std::move(loc), year, mon, day,
                                             hour, min, sec, msec, type);
  }

  std::unique_ptr<ExpressionAST>
  ParseBinaryExpressionRHS(int exprPrecedence,
                           std::unique_ptr<ExpressionAST> lhs) {
    while (true) {
      auto tokPrecedence = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrecedence < exprPrecedence)
        return lhs;

      // Okay, we know this is a binop.
      auto binOp = lexer.getCurToken();
      if (binOp == tok_comma) {
        lhs = ParseListExpression(std::move(lhs));
        continue;
      } else if (binOp == tok_parenthese_open) {
        if (llvm::isa<IntegerConstantAST>(lhs))
          lhs = ParseRepeatedExpression(std::move(lhs));
        else
          lhs = ParseCallExpression(std::move(lhs));
        continue;
      } else if (binOp == tok_sbracket_open) {
        lhs = ParseIndexedVariable(std::move(lhs));
        continue;
      } else if (binOp == tok_integer_literal ||
                 binOp == tok_real_number_literal) {
        binOp = tok_minus;
        lexer.consumeMinus();
      } else {
        lexer.consume(binOp);
      }
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = ParseExpressionPrimary();
      if (!rhs)
        return parseError<ExpressionAST>("expression",
                                         "to complete binary operator");

      // If BinOp binds less tightly with RHS than the operator after RHS, let
      // the pending operator take RHS as its LHS.
      int nextPrecedence = GetTokPrecedence();
      if (tokPrecedence < nextPrecedence) {
        rhs = ParseBinaryExpressionRHS(tokPrecedence + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge LHS/RHS.
      lhs = std::make_unique<BinaryExpressionAST>(
          std::move(loc), binOp, std::move(lhs), std::move(rhs));
    }
  }

  static const int pred_none = -404;
  static const int pred_list = -101;
  static const int pred_default = -100;
  static const int pred_assignment = -11;
  static const int pred_or = -10;
  static const int pred_xor = -9;
  static const int pred_and = -8;
  static const int pred_eq = -7;
  static const int pred_cmp = -6;
  static const int pred_add = -5;
  static const int pred_mult = -4;
  static const int pred_unary = -3;
  static const int pred_exponent = -2;
  static const int pred_paren = -1;
  static const int pred_dot = 0;

  int getTokPrecedence() {
    switch (lexer.getCurToken()) {
    default:
      return pred_none;

    case tok_comma:
      return pred_list;

    case tok_assignment:
    case tok_assign_output:
      return pred_assignment;

    case tok_or:
      return pred_or;

    case tok_xor:
      return pred_xor;

    case tok_ampersand:
    case tok_and:
      return pred_and;

    case tok_cmp_eq:
    case tok_cmp_ne:
      return pred_eq;

    case tok_cmp_lt:
    case tok_cmp_le:
    case tok_cmp_gt:
    case tok_cmp_ge:
      return pred_cmp;

    case tok_minus:
    case tok_plus:
      return pred_add;
    case tok_integer_literal:
    case tok_real_number_literal:
      if (lexer.isNegativeValue())
        return pred_add;
      else
        return pred_none;

    case tok_times:
    case tok_divide:
    case tok_mod:
    case tok_div:
      return pred_mult;

    case tok_exponent:
      return pred_exponent;

    case tok_parenthese_open:
    case tok_sbracket_open:
      return pred_paren;
    case tok_dot:
      return pred_dot;
    }
  }

  std::unique_ptr<ExpressionAST> ParseExpression(int pred = pred_default) {
    auto lhs = ParseExpressionPrimary();
    if (!lhs)
      return nullptr;

    return ParseBinaryExpressionRHS(pred, std::move(lhs));
  }

  std::unique_ptr<ExpressionAST>
  ParseRepeatedExpression(std::unique_ptr<ExpressionAST> repetitions) {
    int32_t reps =
        llvm::cast<IntegerConstantAST>(repetitions.get())->getValue();
    lexer.consume(tok_parenthese_open);

    auto repeated = ParseExpression(pred_list);

    if (lexer.getCurToken() != tok_parenthese_close)
      return parseError<ExpressionAST>(tok_parenthese_close,
                                       "to end constant list");
    lexer.consume(tok_parenthese_close);

    return std::make_unique<RepeatedConstantAST>(repetitions->loc(), reps,
                                                 std::move(repeated));
  }

  std::unique_ptr<ExpressionAST>
  ParseListExpression(std::unique_ptr<ExpressionAST> first) {

    std::vector<std::unique_ptr<ExpressionAST>> list;
    list.reserve(2);
    list.push_back(std::move(first));

    while (lexer.getCurToken() == tok_comma) {
      lexer.consume(tok_comma);

      auto next = ParseExpression();
      if (!next)
        return parseError<ExpressionAST>("expression", "list expression");

      list.push_back(std::move(next));
    }
    return std::make_unique<ExpressionListAST>(first->loc(), std::move(list));
  }

  std::unique_ptr<ExpressionAST>
  ParseIndexedVariable(std::unique_ptr<ExpressionAST> base) {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_sbracket_open);

    std::vector<std::unique_ptr<ExpressionAST>> indices;
    bool anotherIndex = lexer.getCurToken() != tok_sbracket_close;
    while (anotherIndex) {
      auto index = ParseExpression();
      if (!index)
        return parseError<ExpressionAST>("index", "index");
      indices.push_back(std::move(index));

      if (lexer.getCurToken() == tok_comma)
        lexer.consume(tok_comma);
      else
        anotherIndex = false;
    }

    if (lexer.getCurToken() != tok_sbracket_close)
      return parseError<ExpressionAST>(tok_sbracket_close, "to end indices");
    lexer.consume(tok_sbracket_close);

    return std::make_unique<IndexedVariableAST>(std::move(loc), std::move(base),
                                                std::move(indices));
  }

  // MARK: C.6 Function and Function Block Calls

  std::unique_ptr<ExpressionAST>
  ParseCallExpression(std::unique_ptr<ExpressionAST> function) {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_parenthese_open);

    std::vector<std::unique_ptr<ExpressionAST>> parameters;
    bool anotherParam = lexer.getCurToken() != tok_parenthese_close;
    while (anotherParam) {
      auto param = ParseExpression();
      if (!param)
        return parseError<ExpressionAST>("parameter", "function call");
      parameters.push_back(std::move(param));

      if (lexer.getCurToken() == tok_comma)
        lexer.consume(tok_comma);
      else
        anotherParam = false;
    }

    if (lexer.getCurToken() != tok_parenthese_close)
      return parseError<ExpressionAST>(tok_parenthese_close,
                                       "to end function call parameters");
    lexer.consume(tok_parenthese_close);

    return std::make_unique<FunctionCallAST>(
        std::move(loc), std::move(function), std::move(parameters));
  }

  // MARK: C.7 Control Statements

  std::unique_ptr<IfThenAST> ParseIfThen() {
    auto loc = lexer.getLastLocation();

    auto condition = ParseExpression();
    if (!condition)
      parseError<IfThenAST>("expression", "for if condition");

    if (lexer.getCurToken() != tok_then)
      return parseError<IfThenAST>(tok_then, "for the conditional code block");
    lexer.consume(tok_then);

    auto code = ParseCodeSection();
    if (!code)
      parseError<IfThenAST>("code block", "for if statement");

    return std::make_unique<IfThenAST>(std::move(loc), std::move(condition),
                                       std::move(code));
  }

  std::unique_ptr<IfThenElseAST> ParseIfThenElse() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_if);
    bool another = true;

    std::vector<std::unique_ptr<IfThenAST>> thens;
    while (another) {
      auto then = ParseIfThen();
      if (!then)
        return parseError<IfThenElseAST>("then block", "if");
      thens.push_back(std::move(then));

      if (lexer.getCurToken() == tok_elsif)
        lexer.consume(tok_elsif);
      else
        another = false;
    }

    llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock;
    if (lexer.getCurToken() == tok_else) {
      lexer.consume(tok_else);

      elseBlock = ParseCodeSection();
      if (!elseBlock)
        return parseError<IfThenElseAST>("code section", "as else block");
    }

    if (lexer.getCurToken() != tok_end_if)
      return parseError<IfThenElseAST>(tok_end_if, "to end if block");
    lexer.consume(tok_end_if);

    return std::make_unique<IfThenElseAST>(std::move(loc), std::move(thens),
                                           std::move(elseBlock));
  }

  std::unique_ptr<CaseOfAST> ParseCaseOf() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_case);

    auto expr = ParseExpression();
    if (!expr)
      return parseError<CaseOfAST>("expression", "for case");

    if (lexer.getCurToken() != tok_of)
      return parseError<CaseOfAST>(tok_of, "for case values");
    lexer.consume(tok_of);

    auto code = ParseCodeSection();
    if (!code)
      return parseError<CaseOfAST>("code section", "for case");

    llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock;
    if (lexer.getCurToken() == tok_else) {
      lexer.consume(tok_else);

      elseBlock = ParseCodeSection();
      if (!elseBlock)
        return parseError<CaseOfAST>("code section", "as else block");
    }

    if (lexer.getCurToken() != tok_end_case)
      return parseError<CaseOfAST>(tok_end_case, "to end case block");
    lexer.consume(tok_end_case);

    return std::make_unique<CaseOfAST>(std::move(loc), std::move(expr),
                                       std::move(code), std::move(elseBlock));
  }

  std::unique_ptr<ForDoAST> ParseForDo() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_for);

    auto assignment = ParseExpression();

    if (lexer.getCurToken() != tok_to)
      return parseError<ForDoAST>(tok_to, "for loop");
    lexer.consume(tok_to);

    auto last = ParseExpression();
    if (!last)
      return parseError<ForDoAST>("expression", "for end value");

    llvm::Optional<std::unique_ptr<ExpressionAST>> increment;
    if (lexer.getCurToken() == tok_by) {
      lexer.consume(tok_by);

      auto inc = ParseExpression();
      if (!inc)
        return parseError<ForDoAST>("expression", "for increment");
      increment = std::move(inc);
    }

    if (lexer.getCurToken() != tok_do)
      return parseError<ForDoAST>(tok_do, "for loop");
    lexer.consume(tok_do);

    auto code = ParseCodeSection();
    if (!code)
      return parseError<ForDoAST>("code section", "for loop");

    if (lexer.getCurToken() != tok_end_for)
      return parseError<ForDoAST>(tok_end_for, "end of for loop");
    lexer.consume(tok_end_for);

    return std::make_unique<ForDoAST>(std::move(loc), std::move(assignment),
                                      std::move(last), std::move(increment),
                                      std::move(code));
  }

  std::unique_ptr<WhileDoAST> ParseWhileDo() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_while);

    auto condition = ParseExpression();
    if (!condition)
      return parseError<WhileDoAST>("expression", "while loop condition");

    if (lexer.getCurToken() != tok_do)
      return parseError<WhileDoAST>(tok_do, "for code block");
    lexer.consume(tok_do);

    auto code = ParseCodeSection();
    if (!code)
      return parseError<WhileDoAST>("code section", "while loop");

    if (lexer.getCurToken() != tok_end_while)
      return parseError<WhileDoAST>(tok_end_while, "end of while loop");
    lexer.consume(tok_end_while);

    return std::make_unique<WhileDoAST>(std::move(loc), std::move(condition),
                                        std::move(code));
  }

  std::unique_ptr<RepeatUntilAST> ParseRepeatUntil() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_repeat);

    auto code = ParseCodeSection();
    if (!code)
      return parseError<RepeatUntilAST>("code section", "repeat until loop");

    if (lexer.getCurToken() != tok_until)
      return parseError<RepeatUntilAST>(tok_until, "for code block");
    lexer.consume(tok_until);

    auto condition = ParseExpression();
    if (!condition)
      return parseError<RepeatUntilAST>("expression",
                                        "repeat until loop condition");

    return std::make_unique<RepeatUntilAST>(
        std::move(loc), std::move(condition), std::move(code));
  }

  std::unique_ptr<ContinueAST> ParseContinue() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_continue);

    return std::make_unique<ContinueAST>(std::move(loc));
  }

  std::unique_ptr<ReturnAST> ParseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    return std::make_unique<ReturnAST>(std::move(loc));
  }

  std::unique_ptr<ExitAST> ParseExit() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_exit);

    return std::make_unique<ExitAST>(std::move(loc));
  }

  std::unique_ptr<GotoAST> ParseGoto() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_goto);

    auto label = ParseIdentifier();

    return std::make_unique<GotoAST>(std::move(loc), label);
  }

  /// Get the precedence of the pending binary operator token.
  int GetTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace sclang

#endif // SCLANG_PARSER_H
