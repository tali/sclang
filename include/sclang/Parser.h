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

#include "sclang/AST.h"
#include "sclang/Lexer.h"

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
      llvm::errs() << "expected identifier, got " << lexer.getCurToken() << "\n";
      return "";
    }
    std::string identifier = lexer.getIdentifier();
    lexer.consume(tok_identifier);

    return identifier;
  }

  // MARK: B.1.1 Literals


  // MARK: B.2 Remarks

  // MARK: B.3 Block Attributes

  // Appendix C - Syntac Rules
  // MARK: C.1 Subunits of SCL source files

  /// Parse a SCL Program Unit
  /// SCL Program Unit ::= Organization Block | Function | Function block | Data block | User-defined data type
  std::unique_ptr<UnitAST> ParseSCLProgramUnit() {
    switch(lexer.getCurToken()) {
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
  std::unique_ptr<UnitAST> ParseOrganizationBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_organization_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty()) return nullptr;
    auto decls = ParseDeclarationSection();
    if (!decls) return nullptr;

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();

    auto unit = std::make_unique<OrganizationBlockAST>(
                 std::move(identifier), std::move(loc), std::move(decls), std::move(code));
    if (lexer.getCurToken() != tok_end_organization_block)
      return parseError<OrganizationBlockAST>(tok_end_organization_block, "to end organization block");
    lexer.consume(tok_end_organization_block);

    return unit;
   }

  /// Parse an Function
  std::unique_ptr<UnitAST> ParseFunction() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_function);

    auto identifier = ParseIdentifier();
    if (identifier.empty()) return nullptr;
    if (lexer.getCurToken() != tok_colon)
      return parseError<FunctionAST>(tok_colon, "for function type");
    lexer.consume(tok_colon);

    auto type = ParseDataTypeSpec();
    auto decls = ParseDeclarationSection();

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();

    auto unit = std::make_unique<FunctionAST>(std::move(identifier), std::move(loc), std::move(type),
                                          std::move(decls), std::move(code));

    if (lexer.getCurToken() != tok_end_function)
      return parseError<FunctionAST>(tok_end_function, "to end function");
    lexer.consume(tok_end_function);

    return unit;
  }

  /// Parse a Function Block
  std::unique_ptr<UnitAST> ParseFunctionBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_function_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty()) return nullptr;
    auto decls = ParseDeclarationSection();

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto code = ParseCodeSection();

    auto unit = std::make_unique<FunctionBlockAST>(std::move(identifier), std::move(loc),
                                               std::move(decls), std::move(code));

    if (lexer.getCurToken() != tok_end_function_block)
      return parseError<FunctionBlockAST>(tok_end_function_block, "to end function block");
    lexer.consume(tok_end_function_block);

    return unit;
  }

  /// Parse a Data Block
  std::unique_ptr<UnitAST> ParseDataBlock() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_data_block);

    auto identifier = ParseIdentifier();
    if (identifier.empty()) return nullptr;
    auto decls = ParseDeclarationSection();

    if (lexer.getCurToken() == tok_begin)
      lexer.consume(tok_begin);
    auto assignments = ParseAssignmentsSection();

    auto unit = std::make_unique<DataBlockAST>(std::move(identifier), std::move(loc),
                                               std::move(decls), std::move(assignments));

    if (lexer.getCurToken() != tok_end_data_block)
      return parseError<DataBlockAST>(tok_end_data_block, "to end data block");
    lexer.consume(tok_end_data_block);

    return unit;
  }

  /// Parse a User-Defined Data Type
  std::unique_ptr<UnitAST> ParseUDType() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_type);

    auto identifier = ParseIdentifier();
    if (identifier.empty()) return nullptr;
    auto type = ParseDataTypeSpec();

    auto block = std::make_unique<UserDefinedTypeAST>(std::move(identifier),
                                               std::move(loc), std::move(type));

    if (lexer.getCurToken() != tok_end_type)
      return parseError<UserDefinedTypeAST>(tok_end_type, "to end user defined type");
    lexer.consume(tok_end_type);

    return block;
  }

// MARK: C.2 Structure of Declaration Sections

  std::unique_ptr<DeclarationSectionAST> ParseDeclarationSection() {
    auto loc = lexer.getLastLocation();

    // Parse subsections one at a time and accumulate in this vector.
    std::vector<std::unique_ptr<DeclarationSubsectionAST>> subsections;
    while (auto unit = ParseDeclarationSubsection()) {
      subsections.push_back(std::move(unit));
    }

    return std::make_unique<DeclarationSectionAST>(std::move(loc), std::move(subsections));
  }

  std::unique_ptr<DeclarationSubsectionAST> ParseDeclarationSubsection() {
    switch(lexer.getCurToken()) {
    default:
      return nullptr;
    case tok_var:
      return ParseVarSubsection();
    case tok_var_temp:
      return ParseVarTempSubsection();
    }
  }

  std::unique_ptr<VariableDeclarationSubsectionAST> ParseVarSubsection() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_var);

    std::vector<std::unique_ptr<VariableDeclarationAST>> vars;
    while (lexer.getCurToken() != tok_end_var) {
      auto var = ParseVariableDeclaration();
      if (!var) break;
      vars.push_back(std::move(var));
    }
    if (lexer.getCurToken() != tok_end_var)
      return parseError<VariableDeclarationSubsectionAST>(tok_end_var, "to end variable declaration subsection");
    lexer.consume(tok_end_var);

    return std::make_unique<VariableDeclarationSubsectionAST>(loc, std::move(vars));
  }

  std::unique_ptr<AssignmentsSectionAST> ParseAssignmentsSection() {
    auto loc = lexer.getLastLocation();
    return std::make_unique<AssignmentsSectionAST>(std::move(loc));
  }

  std::unique_ptr<VariableIdentifierAST> ParseVariableIdentifier() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseIdentifier();
    if (identifier.empty())
      return parseError<VariableIdentifierAST>(tok_identifier, "as variable name");
    std::vector<std::unique_ptr<VariableAttributeAST>> attributes;

    if (lexer.getCurToken() == tok_bracket_open) {
      lexer.consume(tok_bracket_open);
      // TBD: parse attributes
      lexer.consume(tok_bracket_close);
    }
    return std::make_unique<VariableIdentifierAST>(std::move(loc), std::move(identifier), std::move(attributes));
  }

  std::unique_ptr<VariableDeclarationAST> ParseVariableDeclaration() {
    auto loc = lexer.getLastLocation();

    auto identifier = ParseVariableIdentifier();
    if (!identifier) return nullptr;
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
    if (!type) return nullptr;

    llvm::Optional<std::unique_ptr<DataTypeInitAST>> init;
    if (lexer.getCurToken() == tok_assignment)
      init = ParseDataTypeInitializer();

    if (lexer.getCurToken() != tok_semicolon)
      return parseError<VariableDeclarationAST>(tok_semicolon, "to end variable declaration");
    lexer.consume(tok_semicolon);

    return std::make_unique<VariableDeclarationAST>(std::move(loc), std::move(vars), std::move(type), std::move(init));
  }

  std::unique_ptr<DataTypeInitAST> ParseDataTypeInitializer() {
    if (lexer.getCurToken() != tok_assignment)
      return parseError<DataTypeInitAST>(tok_assignment, "for variable initializer");
    lexer.consume(tok_assignment);

    return ParseArrayInitializationList();
  }

  std::unique_ptr<DataTypeInitAST> ParseArrayInitializationList() {
    auto loc = lexer.getLastLocation();

    auto list = std::vector<std::unique_ptr<Constant>>();

    // TODO: support for repetition factor
    auto value = ParseConstant();
    list.push_back(std::move(value));

    while(lexer.getCurToken() == tok_comma) {
      lexer.consume(tok_comma);

      auto value = ParseConstant();
      list.push_back(std::move(value));
    }

    return std::make_unique<DataTypeInitAST>(std::move(loc), std::move(list));
  }

  std::unique_ptr<TempVariableDeclarationSubsectionAST> ParseVarTempSubsection() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_var_temp);

    std::vector<std::unique_ptr<VariableDeclarationAST>> vars;

    return std::make_unique<TempVariableDeclarationSubsectionAST>(loc, std::move(vars));
  }

  std::unique_ptr<DataTypeSpecAST> ParseDataTypeSpec() {
    auto loc = lexer.getLastLocation();

    // TODO: TBD
    ParseIdentifier();

    return std::make_unique<DataTypeSpecAST>(std::move(loc));
  }


// MARK: C.3 Data Types in SCL


// MARK: C.4 Code Section

  std::unique_ptr<CodeSectionAST> ParseCodeSection() {
    auto loc = lexer.getLastLocation();
    return std::make_unique<CodeSectionAST>(std::move(loc));
  }

// MARK: C.5 Value Assignments

  std::unique_ptr<Constant> ParseConstant() {
    auto loc = lexer.getLastLocation();

    switch(lexer.getCurToken()) {
    default:
      return parseError<Constant>("<numeric value>, <character string>, or <constant name>", "in constant expression");
    case tok_identifier: {
      auto value = lexer.getIdentifier();
      lexer.consume(tok_identifier);
      return std::make_unique<Constant>(std::move(loc), value);
    }
    case tok_number: {
      auto value = lexer.getValue();
      lexer.consume(tok_number);
      return std::make_unique<Constant>(std::move(loc), value);
    }
    }
  }

// MARK: #if 0 -- old examples
// ======================================================================= //
#if 0

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> ParseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    // return takes an optional argument
    llvm::Optional<std::unique_ptr<ExprAST>> expr;
    if (lexer.getCurToken() != tok_comma) {
      expr = ParseExpression();
      if (!expr)
        return nullptr;
    }
    return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> ParseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto Result =
        std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(Result);
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> ParseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(ParseTensorLiteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(ParseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]
    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());
    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
          return llvm::isa<LiteralExprAST>(expr.get());
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      auto &firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }
    return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                            std::move(dims));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> ParseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto V = ParseExpression();
    if (!V)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return V;
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> ParseIdentifierExpr() {
    std::string name = lexer.getId();

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    if (lexer.getCurToken() != '(') // Simple variable ref.
      return std::make_unique<VariableExprAST>(std::move(loc), name);

    // This is a function call.
    lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto Arg = ParseExpression())
          Args.push_back(std::move(Arg));
        else
          return nullptr;

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (name == "print") {
      if (Args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return std::make_unique<PrintExprAST>(std::move(loc), std::move(Args[0]));
    }

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(std::move(loc), name, std::move(Args));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> ParsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return ParseIdentifierExpr();
    case tok_number:
      return ParseNumberExpr();
    case '(':
      return ParseParenExpr();
    case '[':
      return ParseTensorLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                         std::unique_ptr<ExprAST> LHS) {
    // If this is a binop, find its precedence.
    while (true) {
      int TokPrec = GetTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (TokPrec < ExprPrec)
        return LHS;

      // Okay, we know this is a binop.
      int BinOp = lexer.getCurToken();
      lexer.consume(Token(BinOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto RHS = ParsePrimary();
      if (!RHS)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with RHS than the operator after RHS, let
      // the pending operator take RHS as its LHS.
      int NextPrec = GetTokPrecedence();
      if (TokPrec < NextPrec) {
        RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
        if (!RHS)
          return nullptr;
      }

      // Merge LHS/RHS.
      LHS = std::make_unique<BinaryExprAST>(std::move(loc), BinOp,
                                            std::move(LHS), std::move(RHS));
    }
  }

  /// expression::= primary binoprhs
  std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS)
      return nullptr;

    return ParseBinOpRHS(0, std::move(LHS));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::unique_ptr<VarType> ParseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    lexer.getNextToken(); // eat <

    auto type = std::make_unique<VarType>();

    while (lexer.getCurToken() == tok_number) {
      type->shape.push_back(lexer.getValue());
      lexer.getNextToken();
      if (lexer.getCurToken() == ',')
        lexer.getNextToken();
    }

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    lexer.getNextToken(); // eat >
    return type;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= var identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> ParseDeclaration() {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat var

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identified",
                                        "after 'var' declaration");
    std::string id = lexer.getId();
    lexer.getNextToken(); // eat id

    std::unique_ptr<VarType> type; // Type is optional, it can be inferred
    if (lexer.getCurToken() == '<') {
      type = ParseType();
      if (!type)
        return nullptr;
    }

    if (!type)
      type = std::make_unique<VarType>();
    lexer.consume(Token('='));
    auto expr = ParseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                            std::move(*type), std::move(expr));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> ParseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto varDecl = ParseDeclaration();
        if (!varDecl)
          return nullptr;
        exprList->push_back(std::move(varDecl));
      } else if (lexer.getCurToken() == tok_return) {
        // Return statement
        auto ret = ParseReturn();
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else {
        // General expression
        auto expr = ParseExpression();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return exprList;
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> ParsePrototype() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_def);
    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string FnName = lexer.getId();
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<VariableExprAST>> args;
    if (lexer.getCurToken() != ')') {
      do {
        std::string name = lexer.getId();
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
        args.push_back(std::move(decl));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>("}", "to end function prototype");

    // success.
    lexer.consume(Token(')'));
    return std::make_unique<PrototypeAST>(std::move(loc), FnName,
                                          std::move(args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> ParseDefinition() {
    auto Proto = ParsePrototype();
    if (!Proto)
      return nullptr;

    if (auto block = ParseBlock())
      return std::make_unique<FunctionAST>(std::move(Proto), std::move(block));
    return nullptr;
  }
  #endif

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
