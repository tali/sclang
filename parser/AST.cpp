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
  void dump(const ModuleAST *Node);

private:
  // Subunits of SCL Source Files
  void dump(const UnitAST *Node);
  void dump(const llvm::ArrayRef<std::unique_ptr<BlockAttributeAST>> attr);
  void dump(const BlockAttributeAST *attr);
  void dump(const OrganizationBlockAST *node);
  void dump(const FunctionAST *node);
  void dump(const FunctionBlockAST *node);
  void dump(const DataBlockAST *node);
  void dump(const UserDefinedTypeAST *node);
  // Structure of Declaration Sections
  void dump(const DeclarationSectionAST *node);
  void dump(const DeclarationSubsectionAST *node);
  void dump(const ConstantDeclarationAST *node);
  void dump(const ConstantDeclarationSubsectionAST *node);
  void dump(const JumpLabelDeclarationSubsectionAST *node);
  void dump(const VariableDeclarationSubsectionAST *node);
  void dump(const VariableDeclarationAST *node);
  void dump(const VariableAttributeAST *node);
  void dump(const VariableIdentifierAST *node);
  void dump(const DataTypeInitAST *node);
  void dump(const DataTypeSpecAST *node);
  void dump(const ElementaryDataTypeAST *node);
  void dump(const StringDataTypeSpecAST *node);
  void dump(const ArrayDataTypeSpecAST *node);
  void dump(const ComponentDeclarationAST *node);
  void dump(const StructDataTypeSpecAST *node);
  void dump(const UserDefinedTypeIdentifierAST *node);
  void dump(const DBAssignmentAST *node);
  void dump(const DBAssignmentSectionAST *node);
  // Code Section
  void dump(const CodeSectionAST *node);
  void dump(const InstructionAST *node);
  void dump(const ValueAssignmentAST *node);
  void dump(const SimpleVariableAST *node);
  // Value Assignments
  void dump(const ExpressionAST *node);
  void dump(const BinaryExpressionAST *node);
  void dump(const UnaryExpressionAST *node);
  void dump(const IntegerConstantAST *node);
  void dump(const RealConstantAST *node);
  void dump(const StringConstantAST *node);
  // Function and Function Block Calls
  void dump(const SubroutineProcessingAST *node);
  void dump(const llvm::ArrayRef<std::unique_ptr<ExpressionAST>>);
  void dump(const FunctionCallAST *node);
  // Control Statements
  void dump(const IfThenAST *node);
  void dump(const IfThenElseAST *node);
  void dump(const CaseValueAST *node);
  void dump(const CaseValueSingleAST *node);
  void dump(const CaseValueRangeAST *node);
  void dump(const CaseBlockAST *node);
  void dump(const CaseOfAST *node);
  void dump(const ForDoAST *node);
  void dump(const WhileDoAST *node);
  void dump(const RepeatUntilAST *node);
  void dump(const ContinueAST *node);
  void dump(const ReturnAST *node);
  void dump(const ExitAST *node);
  void dump(const GotoAST *node);

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
void ASTDumper::dump(const UnitAST *unit) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(unit))                               \
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

void ASTDumper::dump(const BlockAttributeAST *attr) {
  INDENT();
  llvm::errs() << "Attribute " << attr->getName() << " : " << attr->getValue() << "\n";
}

void ASTDumper::dump(llvm::ArrayRef<std::unique_ptr<BlockAttributeAST>> attrs) {
  if (attrs.empty()) return;
  INDENT();
  llvm::errs() << "Attributes\n";
  for (auto const & attr : attrs) {
    dump(attr.get());
  }
}

// MARK: C.1 Subunits of SCL Program Files

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(const OrganizationBlockAST *unit) {
  INDENT();
  llvm::errs() << "OrganizationBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getCode());
}

void ASTDumper::dump(const FunctionAST *unit) {
  INDENT();
  llvm::errs() << "Function " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getType());
  dump(unit->getCode());
}

void ASTDumper::dump(const FunctionBlockAST *unit) {
  INDENT();
  llvm::errs() << "FunctionBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getCode());
}

void ASTDumper::dump(const DataBlockAST *unit) {
  INDENT();
  llvm::errs() << "DataBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getAssignments());
}

void ASTDumper::dump(const UserDefinedTypeAST *unit) {
  INDENT();
  llvm::errs() << "UserDefinedType " << unit->getIdentifier() << "\n";
  dump(unit->getType());
  dump(unit->getAttributes());
}

// MARK: C.2 Structure of Declaration Sections

void ASTDumper::dump(const DeclarationSectionAST *section) {
  INDENT();
  llvm::errs() << "DeclarationSection\n";
  for (auto &subsection : section->getDecls()) {
    dump(subsection.get());
  }
}

void ASTDumper::dump(const DeclarationSubsectionAST *subsection) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(subsection))                         \
    return dump(node);
  dispatch(ConstantDeclarationSubsectionAST);
  dispatch(JumpLabelDeclarationSubsectionAST);
  dispatch(VariableDeclarationSubsectionAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown declaration subsection, kind" << subsection->getKind() << ">\n";
}

void ASTDumper::dump(const ConstantDeclarationSubsectionAST *node) {
  INDENT();
  llvm::errs() << "ConstantDeclarationSubsection\n";
  for (auto const & decl : node->getValues()) {
    dump(decl.get());
  }
}

void ASTDumper::dump(const ConstantDeclarationAST *node) {
  INDENT();
  llvm::errs() << "ConstantDeclaration " << node->getName() << "\n";
  dump(node->getValue());
}

void ASTDumper::dump(const JumpLabelDeclarationSubsectionAST *node) {
  INDENT();
  llvm::errs() << "JumpLabelDeclarationSubsection\n";
}

void ASTDumper::dump(const VariableDeclarationSubsectionAST *node) {
  INDENT();
  switch(node->getKind()) {
  case VariableDeclarationSubsectionAST::Var:
    llvm::errs() << "VariableSubsection\n";
    break;
  case VariableDeclarationSubsectionAST::VarTemp:
    llvm::errs() << "TemporaryVariableSubsection\n";
    break;
  case VariableDeclarationSubsectionAST::VarInput:
    llvm::errs() << "ParameterSubsection Input\n";
    break;
  case VariableDeclarationSubsectionAST::VarOutput:
    llvm::errs() << "ParameterSubsection Output\n";
    break;
  case VariableDeclarationSubsectionAST::VarInOut:
    llvm::errs() << "ParameterSubsection InOut\n";
    break;
  }
  auto values = node->getValues();
  for (auto &decl : values)
    dump(decl.get());
}

void ASTDumper::dump(const VariableDeclarationAST *node) {
  INDENT();
  llvm::errs() << "VariableDeclaration\n";
  for (auto &var : node->getVars())
    dump(var.get());
  dump(node->getDataType());
  auto init = node->getInitializer();
  if (init.hasValue())
    dump(init.getValue());
}

void ASTDumper::dump(const VariableAttributeAST *node) {
  INDENT();
  llvm::errs() << "VariableAttribute\n";
}

void ASTDumper::dump(const VariableIdentifierAST *node) {
  INDENT();
  llvm::errs() << "VariableIdentifier `" << node->getIdentifier() << "`\n";
  for (auto &attr : node->getAttributes())
    dump(attr.get());
}

void ASTDumper::dump(const DataTypeInitAST *node) {
  INDENT();
  llvm::errs() << "DataTypeInit\n";
  for (auto &init : node->getList())
    dump(init.get());
}

void ASTDumper::dump(const DataTypeSpecAST *dataType) {
  #define dispatch(CLASS)                                                      \
    if (const CLASS *node = llvm::dyn_cast<CLASS>(dataType))                         \
      return dump(node);
    dispatch(ElementaryDataTypeAST);
    dispatch(StringDataTypeSpecAST);
    dispatch(ArrayDataTypeSpecAST);
    dispatch(StructDataTypeSpecAST);
    dispatch(UserDefinedTypeIdentifierAST);
  #undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown data type, kind" << dataType->getKind() << ">\n";
}

void ASTDumper::dump(const ElementaryDataTypeAST *node) {
  INDENT();
  switch (node->getType()) {
  case sclang::ElementaryDataTypeAST::Type_Void:
    llvm::errs() << "ElementaryDataType Void\n"; return;
  case ElementaryDataTypeAST::Type_Bool:
    llvm::errs() << "ElementaryDataType Bool\n"; return;
  case ElementaryDataTypeAST::Type_Byte:
    llvm::errs() << "ElementaryDataType Byte\n"; return;
  case ElementaryDataTypeAST::Type_Word:
    llvm::errs() << "ElementaryDataType Word\n"; return;
  case ElementaryDataTypeAST::Type_DWord:
    llvm::errs() << "ElementaryDataType DWord\n"; return;
  case ElementaryDataTypeAST::Type_Char:
    llvm::errs() << "ElementaryDataType Char\n"; return;
  case ElementaryDataTypeAST::Type_Int:
    llvm::errs() << "ElementaryDataType Int\n"; return;
  case ElementaryDataTypeAST::Type_DInt:
    llvm::errs() << "ElementaryDataType DInt\n"; return;
  case ElementaryDataTypeAST::Type_Real:
    llvm::errs() << "ElementaryDataType Real\n"; return;
  case ElementaryDataTypeAST::Type_S5Time:
    llvm::errs() << "ElementaryDataType S5Time\n"; return;
  case ElementaryDataTypeAST::Type_Time:
    llvm::errs() << "ElementaryDataType Time\n"; return;
  case ElementaryDataTypeAST::Type_TimeOfDay:
    llvm::errs() << "ElementaryDataType TimeOfDay\n"; return;
  case ElementaryDataTypeAST::Type_Date:
    llvm::errs() << "ElementaryDataType Date\n"; return;
  case ElementaryDataTypeAST::Type_DateAndTime:
    llvm::errs() << "ElementaryDataType DateAndTime\n"; return;
  case ElementaryDataTypeAST::Type_Timer:
    llvm::errs() << "ElementaryDataType Timer\n"; return;
  case ElementaryDataTypeAST::Type_Counter:
    llvm::errs() << "ElementaryDataType Counter\n"; return;
  case ElementaryDataTypeAST::Type_Any:
    llvm::errs() << "ElementaryDataType Any\n"; return;
  case ElementaryDataTypeAST::Type_Pointer:
    llvm::errs() << "ElementaryDataType Pointer\n"; return;
  case ElementaryDataTypeAST::Type_BlockFC:
    llvm::errs() << "ElementaryDataType BlockFC\n"; return;
  case ElementaryDataTypeAST::Type_BlockFB:
    llvm::errs() << "ElementaryDataType BlockFB\n"; return;
  case ElementaryDataTypeAST::Type_BlockDB:
    llvm::errs() << "ElementaryDataType BlockDB\n"; return;
  case ElementaryDataTypeAST::Type_BlockSDB:
    llvm::errs() << "ElementaryDataType BlockSDB\n"; return;
  }
  llvm::errs() << "<unknown ElementaryDataType, type " << node->getType() << ">\n";
}

void ASTDumper::dump(const StringDataTypeSpecAST *node) {
  INDENT();
  llvm::errs() << "StringDataTypeSpec [" << node->getMaxLen() << "]\n";
}

void ASTDumper::dump(const ArrayDataTypeSpecAST *node) {
  INDENT();
  llvm::errs() << "ArrayDataTypeSpec";
  for (auto const &dim : node->getDimensions()) {
    llvm::errs() << " [" << dim.first << ".." << dim.second << "]";
  }
  llvm::errs() << "\n";
  dump(node->getDataType());
}

void ASTDumper::dump(const ComponentDeclarationAST *node) {
  INDENT();
  llvm::errs() << "ComponentDeclaration '" << node->getName() << "'\n";
  dump(node->getDataType());
  auto init = node->getInitializer();
  if (init.hasValue())
    dump(init.getValue());
}

void ASTDumper::dump(const StructDataTypeSpecAST *node) {
  INDENT();
  llvm::errs() << "StructDataTypeSpec\n";
  for (auto const &component : node->getComponents())
    dump(component.get());
}

void ASTDumper::dump(const UserDefinedTypeIdentifierAST *node) {
  INDENT();
  llvm::errs() << "UserDefinedTypeIdentifier '" << node->getName() << "'\n";
}

void ASTDumper::dump(const DBAssignmentAST *node) {
  INDENT();
  llvm::errs() << "DBAssignment ''\n";
}

void ASTDumper::dump(const DBAssignmentSectionAST *node) {
  INDENT();
  llvm::errs() << "DBAssignmentSection\n";
  for (auto const &assign : node->getAssignments())
    dump(assign.get());
}

// MARK: C.3

// MARK: C.4 Code Section

void ASTDumper::dump(const CodeSectionAST *node) {
  INDENT();
  llvm::errs() << "CodeSection\n";
  for (auto const &instr : node->getInstructions())
    dump(instr.get());
}

void ASTDumper::dump(const InstructionAST *code) {
    #define dispatch(CLASS)                                                        \
      if (const CLASS *node = llvm::dyn_cast<CLASS>(code))                         \
        return dump(node);
      dispatch(JumpLabelAST);
      dispatch(ValueAssignmentAST);
      dispatch(SubroutineProcessingAST);
      dispatch(IfThenElseAST);
      dispatch(CaseOfAST);
      dispatch(ForDoAST);
      dispatch(WhileDoAST);
      dispatch(RepeatUntilAST);
      dispatch(ContinueAST);
      dispatch(ReturnAST);
      dispatch(ExitAST);
      dispatch(GotoAST);
    #undef dispatch
      // No match, fallback to a generic message
      INDENT();
      llvm::errs() << "<unknown instruction, kind" << code->getKind() << ">\n";
}

// MARK: C.5 Value Assignments

void ASTDumper::dump(const ValueAssignmentAST *node) {
  INDENT();
  llvm::errs() << "ValueAssignment\n";
  dump(node->getExpression());
}

void ASTDumper::dump(const ExpressionAST *expr) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(expr))                         \
    return dump(node);
  dispatch(IntegerConstantAST);
  dispatch(RealConstantAST);
  dispatch(StringConstantAST);
  dispatch(SimpleVariableAST);
//    dispatch(AbsoluteVariableAST);
//    dispatch(VariableInDBAST);
  dispatch(FunctionCallAST);
  dispatch(BinaryExpressionAST);
  dispatch(UnaryExpressionAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown expression, kind" << expr->getKind() << ">\n";
}

void ASTDumper::dump(const SimpleVariableAST *node) {
  INDENT();
  llvm::errs() << "SimpleVariable " << node->getName() << "\n";
}

void ASTDumper::dump(const BinaryExpressionAST *node) {
  INDENT();
  llvm::errs() << "BinaryExpression " << node->getOp() << "\n";
  dump(node->getLhs());
  dump(node->getRhs());
}

void ASTDumper::dump(const UnaryExpressionAST *node) {
  INDENT();
  llvm::errs() << "UnaryExpression " << node->getOp() << "\n";
  dump(node->getRhs());
}

void ASTDumper::dump(const IntegerConstantAST *node) {
  INDENT();
  llvm::errs() << "IntegerConstant " << node->getValue();
  if (node->getType())
    llvm::errs() << " Type " << node->getType();
  llvm::errs() << "\n";
}

void ASTDumper::dump(const RealConstantAST *node) {
  INDENT();
  llvm::errs() << "RealConstant " << node->getValue();
  if (node->getType())
    llvm::errs() << " Type " << node->getType();
  llvm::errs() << "\n";
}

void ASTDumper::dump(const StringConstantAST *node) {
  INDENT();
  llvm::errs() << "StringConstant '" << node->getValue() << "'\n";
}

// MARK: C.6 Function and Function Block Calls

void ASTDumper::dump(const SubroutineProcessingAST *node) {
  INDENT();
  llvm::errs() << "Subroutine\n";
  dump(node->getCall());
}

void ASTDumper::dump(llvm::ArrayRef<std::unique_ptr<ExpressionAST>> parameters) {
  INDENT();
  llvm::errs() << "Parameters\n";
  for (const auto & param : parameters) {
    dump(param.get());
  }
}

void ASTDumper::dump(const FunctionCallAST *node) {
  INDENT();
  llvm::errs() << "FunctionCall\n";
  dump(node->getFunction());
  dump(node->getParameters());
}


// MARK: C.7 Control Statements

void ASTDumper::dump(const IfThenAST *node) {
  INDENT();
  llvm::errs() << "IfThen\n";
  dump(node->getCondition());
  dump(node->getCodeBlock());
}

void ASTDumper::dump(const IfThenElseAST *node) {
  INDENT();
  llvm::errs() << "IfThenElse\n";

  for (auto const & then : node->getThens())
    dump(then.get());

  auto elseBlock = node->getElseBlock();
  if (elseBlock.hasValue()) {
    INDENT();
    llvm::errs() << "Else\n";
    dump(elseBlock.getValue());
  }
}

void ASTDumper::dump(const CaseValueAST *value) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(value))                         \
    return dump(node);
  dispatch(CaseValueSingleAST);
  dispatch(CaseValueRangeAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown case value, kind" << value->getKind() << ">\n";
}

void ASTDumper::dump(const CaseValueSingleAST *node) {
  INDENT();
  llvm::errs() << "CaseValueSingle\n";
  dump(node->getValue());
}

void ASTDumper::dump(const CaseValueRangeAST *node) {
  INDENT();
  llvm::errs() << "CaseValueRange\n";
  dump(node->getMin());
  dump(node->getMax());
}

void ASTDumper::dump(const CaseBlockAST *node) {
  INDENT();
  llvm::errs() << "CaseBlock\n";
  for (auto const & value : node->getValues())
    dump(value.get());
  dump(node->getCodeBlock());
}

void ASTDumper::dump(const CaseOfAST *node) {
  INDENT();
  llvm::errs() << "Case\n";
  dump(node->getExpr());
  llvm::errs() << "Of\n";
  dump(node->getCode());
  auto elseBlock = node->getElseBlock();
  if (elseBlock.hasValue())
    dump(node->getElseBlock().getValue());
}

void ASTDumper::dump(const ForDoAST *node) {
  INDENT();
  llvm::errs() << "For\n";
  dump(node->getInitial());
  llvm::errs() << "To\n";
  dump(node->getLast());
  llvm::errs() << "By\n";
  dump(node->getIncrement());
  llvm::errs() << "Do\n";
  dump(node->getCode());
}

void ASTDumper::dump(const WhileDoAST *node) {
  INDENT();
  llvm::errs() << "While\n";
  dump(node->getCondition());
  llvm::errs() << "Do\n";
  dump(node->getCode());
}

void ASTDumper::dump(const RepeatUntilAST *node) {
  INDENT();
  llvm::errs() << "Repeat\n";
  dump(node->getCode());
  llvm::errs() << "Until\n";
  dump(node->getCondition());
}

void ASTDumper::dump(const ContinueAST *node) {
  INDENT();
  llvm::errs() << "Continue\n";
}

void ASTDumper::dump(const ReturnAST *node) {
  INDENT();
  llvm::errs() << "Return\n";
}

void ASTDumper::dump(const ExitAST *node) {
  INDENT();
  llvm::errs() << "Exit\n";
}

void ASTDumper::dump(const GotoAST *node) {
  INDENT();
  llvm::errs() << "Goto " << node->getLabel() << "\n";
}



/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(const ModuleAST *Node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto const &unit : *Node)
    dump(unit.get());
}

namespace sclang {

// Public API
void dump(const ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace sclang
