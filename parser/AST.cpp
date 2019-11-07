//===- AST.cpp - Helper for printing out the SCL AST ----------------------===//
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
// This file implements the AST dump for the SCL language.
//
//===----------------------------------------------------------------------===//

#include "sclang/AST.h"

#include <string>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace sclang;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *Node);

private:
  // Subunits of SCL Source Files
  void dump(UnitAST *Node);
  void dump(OrganizationBlockAST *node);
  void dump(FunctionAST *node);
  void dump(FunctionBlockAST *node);
  void dump(DataBlockAST *node);
  void dump(UserDefinedTypeAST *node);
  // Structure of Declaration Sections
  void dump(DeclarationSectionAST *node);
  void dump(DeclarationSubsectionAST *node);
  void dump(ConstantDeclarationSubsectionAST *node);
  void dump(JumpLabelDeclarationSubsectionAST *node);
  void dump(VariableDeclarationSubsectionAST *node);
  void dump(VariableDeclarationAST *node);
  void dump(VariableAttributeAST *node);
  void dump(VariableIdentifierAST *node);
  void dump(DataTypeInitAST *node);
  void dump(DataTypeSpecAST *node);

  //
#if 0
  void dump(VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *Node);
  void dump(VariableExprAST *Node);
  void dump(ReturnExprAST *Node);
  void dump(BinaryExprAST *Node);
  void dump(CallExprAST *Node);
  void dump(PrintExprAST *Node);
  void dump(PrototypeAST *Node);
  void dump(FunctionAST *Node);
#endif

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *Node) {
  const auto &loc = Node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(UnitAST *unit) {
#define dispatch(CLASS)                                                        \
  if (CLASS *node = llvm::dyn_cast<CLASS>(unit))                               \
    return dump(node);
  dispatch(OrganizationBlockAST);
  dispatch(FunctionAST);
  dispatch(FunctionBlockAST);
  dispatch(DataBlockAST);
  dispatch(UserDefinedTypeAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown Expr, kind " << unit->getKind() << ">\n";
}

// MARK: C.1 Subunits of SCL Program Files

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(OrganizationBlockAST *unit) {
  INDENT();
  llvm::errs() << "OrganizationBlock " << unit->getIdentifier() << "\n";
  dump(unit->getDeclarations());
}

void ASTDumper::dump(FunctionAST *unit) {
  INDENT();
  llvm::errs() << "Function " << unit->getIdentifier() << "\n";
  dump(unit->getDeclarations());
}

void ASTDumper::dump(FunctionBlockAST *unit) {
  INDENT();
  llvm::errs() << "FunctionBlock " << unit->getIdentifier() << "\n";
  dump(unit->getDeclarations());
}

void ASTDumper::dump(DataBlockAST *unit) {
  INDENT();
  llvm::errs() << "DataBlock " << unit->getIdentifier() << "\n";
  dump(unit->getDeclarations());
}

void ASTDumper::dump(UserDefinedTypeAST *unit) {
  INDENT();
  llvm::errs() << "UserDefinedType " << unit->getIdentifier() << "\n";
}

// MARK: C.2 Structure of Declaration Sections

void ASTDumper::dump(DeclarationSectionAST *section) {
  INDENT();
  llvm::errs() << "DeclarationSection\n";
  for (auto &subsection : section->getDecls()) {
    dump(subsection.get());
  }
}

void ASTDumper::dump(DeclarationSubsectionAST *subsection) {
#define dispatch(CLASS)                                                        \
  if (CLASS *node = llvm::dyn_cast<CLASS>(subsection))                         \
    return dump(node);
  dispatch(ConstantDeclarationSubsectionAST);
  dispatch(JumpLabelDeclarationSubsectionAST);
  dispatch(VariableDeclarationSubsectionAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknwon declaration subsection, kinde" << subsection->getKind() << "\n";
}

void ASTDumper::dump(ConstantDeclarationSubsectionAST *node) {
  INDENT();
  llvm::errs() << "ConstantDeclarationSubsectionAST\n";
}

void ASTDumper::dump(JumpLabelDeclarationSubsectionAST *node) {
  INDENT();
  llvm::errs() << "JumpLabelDeclarationSubsectionAST\n";
}

void ASTDumper::dump(VariableDeclarationSubsectionAST *node) {
  INDENT();
  llvm::errs() << "VariableDeclarationSubsection\n";
  auto values = node->getValues();
  for (auto &decl : values)
    dump(decl.get());
}

void ASTDumper::dump(VariableDeclarationAST *node) {
  INDENT();
  llvm::errs() << "VariableDeclaration\n";
  for (auto &var : node->getVars())
    dump(var.get());
  dump(node->getDataType());
  auto init = node->getInitializer();
  if (init.hasValue())
    dump(init.getValue());
}

void ASTDumper::dump(VariableAttributeAST *node) {
  INDENT();
  llvm::errs() << "VariableAttribute\n";
}

void ASTDumper::dump(VariableIdentifierAST *node) {
  INDENT();
  llvm::errs() << "VariableIdentifier `" << node->getIdentifier() << "`\n";
  for (auto &attr : node->getAttributes())
    dump(attr.get());
}

void ASTDumper::dump(DataTypeInitAST *node) {
  INDENT();
  llvm::errs() << "DataTypeInit\n";
}

void ASTDumper::dump(DataTypeSpecAST *node) {
  INDENT();
  llvm::errs() << "DataTypeSpec\n";
}


// ====================================================================== //


#if 0
/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
#define dispatch(CLASS)                                                        \
  if (CLASS *node = llvm::dyn_cast<CLASS>(expr))                               \
    return dump(node);
  dispatch(VarDeclExprAST);
  dispatch(LiteralExprAST);
  dispatch(NumberExprAST);
  dispatch(VariableExprAST);
  dispatch(ReturnExprAST);
  dispatch(BinaryExprAST);
  dispatch(CallExprAST);
  dispatch(PrintExprAST);
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &expr : *exprList)
    dump(expr.get());
  indent();
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void printLitHelper(ExprAST *lit_or_num) {
  // Inside a literal expression we can have either a number or another literal
  if (auto num = llvm::dyn_cast<NumberExprAST>(lit_or_num)) {
    llvm::errs() << num->getValue();
    return;
  }
  auto *literal = llvm::cast<LiteralExprAST>(lit_or_num);

  // Print the dimension for this literal first
  llvm::errs() << "<";
  {
    const char *sep = "";
    for (auto dim : literal->getDims()) {
      llvm::errs() << sep << dim;
      sep = ", ";
    }
  }
  llvm::errs() << ">";

  // Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  const char *sep = "";
  for (auto &elt : literal->getValues()) {
    llvm::errs() << sep;
    printLitHelper(elt.get());
    sep = ", ";
  }
  llvm::errs() << "]";
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(LiteralExprAST *Node) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(Node);
  llvm::errs() << " " << loc(Node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *Node) {
  INDENT();
  llvm::errs() << "var: " << Node->getName() << " " << loc(Node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *Node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (Node->getExpr().hasValue())
    return dump(*Node->getExpr());
  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *Node) {
  INDENT();
  llvm::errs() << "BinOp: " << Node->getOp() << " " << loc(Node) << "\n";
  dump(Node->getLHS());
  dump(Node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *Node) {
  INDENT();
  llvm::errs() << "Call '" << Node->getCallee() << "' [ " << loc(Node) << "\n";
  for (auto &arg : Node->getArgs())
    dump(arg.get());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *Node) {
  INDENT();
  llvm::errs() << "Print [ " << loc(Node) << "\n";
  dump(Node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(VarType &type) {
  llvm::errs() << "<";
  const char *sep = "";
  for (auto shape : type.shape) {
    llvm::errs() << sep << shape;
    sep = ", ";
  }
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *Node) {
  INDENT();
  llvm::errs() << "Proto '" << Node->getName() << "' " << loc(Node) << "'\n";
  indent();
  llvm::errs() << "Params: [";
  const char *sep = "";
  for (auto &arg : Node->getArgs()) {
    llvm::errs() << sep << arg->getName();
    sep = ", ";
  }
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *Node) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(Node->getProto());
  dump(Node->getBody());
}
#endif

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *Node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &unit : *Node)
    dump(unit.get());
}

namespace sclang {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace sclang
