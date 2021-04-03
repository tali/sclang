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

#include "sclang/SclGen/AST.h"

#include <string>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
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
  ASTDumper(llvm::raw_ostream &outs) : outs(outs) {}
  void dump(const ModuleAST *Node);

private:
  // Subunits of SCL Source Files
  void dump(const UnitAST *Node);
  void dump(const llvm::ArrayRef<std::unique_ptr<AttributeAST>> attr);
  void dump(const AttributeAST *attr);
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
  void dump(const JumpLabelDeclarationAST *node);
  void dump(const JumpLabelDeclarationSubsectionAST *node);
  void dump(const VariableDeclarationSubsectionAST *node);
  void dump(const VariableDeclarationAST *node);
  void dump(const VariableIdentifierAST *node);
  void dump(const DataTypeSpecAST *node);
  void dump(const ElementaryDataTypeAST *node);
  void dump(const StringDataTypeSpecAST *node);
  void dump(const ArrayDimensionAST *node);
  void dump(const ArrayDataTypeSpecAST *node);
  void dump(const ComponentDeclarationAST *node);
  void dump(const StructDataTypeSpecAST *node);
  void dump(const UserDefinedTypeIdentifierAST *node);
  // Code Section
  void dump(const CodeSectionAST *node);
  void dump(const InstructionAST *node);
  void dump(const ValueAssignmentAST *node);
  void dump(const SimpleVariableAST *node);
  void dump(const IndexedVariableAST *node);
  // Value Assignments
  void dump(const ExpressionAST *node);
  void dump(const ExpressionListAST *node);
  void dump(const RepeatedConstantAST *node);
  void dump(const BinaryExpressionAST *node);
  void dump(const UnaryExpressionAST *node);
  void dump(const IntegerConstantAST *node);
  void dump(const RealConstantAST *node);
  void dump(const StringConstantAST *node);
  void dump(const TimeConstantAST *node);
  void dump(const FunctionCallAST *node);
  // Function and Function Block Calls
  void dump(const SubroutineProcessingAST *node);
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

  void dump(std::string, const llvm::ArrayRef<std::unique_ptr<ExpressionAST>>);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      outs << "  ";
  }
  int curIndent = 0;
  llvm::raw_ostream &outs;
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
  if (const CLASS *node = llvm::dyn_cast<CLASS>(unit))                         \
    return dump(node);
  dispatch(OrganizationBlockAST);
  dispatch(FunctionAST);
  dispatch(FunctionBlockAST);
  dispatch(DataBlockAST);
  dispatch(UserDefinedTypeAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  outs << "<unknown Expr, kind " << unit->getKind() << ">\n";
}

void ASTDumper::dump(const AttributeAST *attr) {
  INDENT();
  outs << "Attribute " << attr->getName() << " : " << attr->getValue() << "\n";
}

void ASTDumper::dump(llvm::ArrayRef<std::unique_ptr<AttributeAST>> attrs) {
  if (attrs.empty())
    return;
  INDENT();
  outs << "Attributes\n";
  for (auto const &attr : attrs) {
    dump(attr.get());
  }
}

// MARK: C.1 Subunits of SCL Program Files

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(const OrganizationBlockAST *unit) {
  INDENT();
  outs << "OrganizationBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getCode());
}

void ASTDumper::dump(const FunctionAST *unit) {
  INDENT();
  outs << "Function " << unit->getIdentifier() << "\n";
  dump(unit->getType());
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getCode());
}

void ASTDumper::dump(const FunctionBlockAST *unit) {
  INDENT();
  outs << "FunctionBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getDeclarations());
  dump(unit->getCode());
}

void ASTDumper::dump(const DataBlockAST *unit) {
  INDENT();
  outs << "DataBlock " << unit->getIdentifier() << "\n";
  dump(unit->getAttributes());
  dump(unit->getType());
  dump(unit->getAssignments());
}

void ASTDumper::dump(const UserDefinedTypeAST *unit) {
  INDENT();
  outs << "UserDefinedType " << unit->getIdentifier() << "\n";
  dump(unit->getType());
  dump(unit->getAttributes());
}

// MARK: C.2 Structure of Declaration Sections

void ASTDumper::dump(const DeclarationSectionAST *section) {
  INDENT();
  outs << "DeclarationSection\n";
  for (auto &subsection : section->getDecls()) {
    dump(subsection.get());
  }
}

void ASTDumper::dump(const DeclarationSubsectionAST *subsection) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(subsection))                   \
    return dump(node);
  dispatch(ConstantDeclarationSubsectionAST);
  dispatch(JumpLabelDeclarationSubsectionAST);
  dispatch(VariableDeclarationSubsectionAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  outs << "<unknown declaration subsection, kind" << subsection->getKind()
       << ">\n";
}

void ASTDumper::dump(const ConstantDeclarationSubsectionAST *node) {
  INDENT();
  outs << "ConstantDeclarationSubsection\n";
  for (auto const &decl : node->getValues()) {
    dump(decl.get());
  }
}

void ASTDumper::dump(const ConstantDeclarationAST *node) {
  INDENT();
  outs << "ConstantDeclaration " << node->getName() << "\n";
  dump(node->getValue());
}

void ASTDumper::dump(const JumpLabelDeclarationSubsectionAST *node) {
  INDENT();
  outs << "JumpLabelDeclarationSubsection\n";
  for (auto const &decl : node->getValues()) {
    dump(decl.get());
  }
}

void ASTDumper::dump(const JumpLabelDeclarationAST *node) {
  INDENT();
  outs << "JumpLabelDeclaration " << node->getIdentifier() << "\n";
}

void ASTDumper::dump(const VariableDeclarationSubsectionAST *node) {
  INDENT();
  switch (node->getKind()) {
  case VariableDeclarationSubsectionAST::Var:
    outs << "VariableSubsection\n";
    break;
  case VariableDeclarationSubsectionAST::VarTemp:
    outs << "TemporaryVariableSubsection\n";
    break;
  case VariableDeclarationSubsectionAST::VarInput:
    outs << "ParameterSubsection Input\n";
    break;
  case VariableDeclarationSubsectionAST::VarOutput:
    outs << "ParameterSubsection Output\n";
    break;
  case VariableDeclarationSubsectionAST::VarInOut:
    outs << "ParameterSubsection InOut\n";
    break;
  }
  auto values = node->getValues();
  for (auto &decl : values)
    dump(decl.get());
}

void ASTDumper::dump(const VariableDeclarationAST *node) {
  INDENT();
  outs << "VariableDeclaration\n";
  for (auto &var : node->getVars())
    dump(var.get());
  dump(node->getDataType());
  auto init = node->getInitializer();
  if (init.hasValue())
    dump(init.getValue());
}

void ASTDumper::dump(const VariableIdentifierAST *node) {
  INDENT();
  outs << "VariableIdentifier " << node->getIdentifier() << "\n";
  for (auto &attr : node->getAttributes())
    dump(attr.get());
}

void ASTDumper::dump(const DataTypeSpecAST *dataType) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(dataType))                     \
    return dump(node);
  dispatch(ElementaryDataTypeAST);
  dispatch(StringDataTypeSpecAST);
  dispatch(ArrayDataTypeSpecAST);
  dispatch(StructDataTypeSpecAST);
  dispatch(UserDefinedTypeIdentifierAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  outs << "<unknown data type, kind" << dataType->getKind() << ">\n";
}

void ASTDumper::dump(const ElementaryDataTypeAST *node) {
  INDENT();
  switch (node->getType()) {
  case sclang::ElementaryDataTypeAST::Type_Void:
    outs << "ElementaryDataType Void\n";
    return;
  case ElementaryDataTypeAST::Type_Bool:
    outs << "ElementaryDataType Bool\n";
    return;
  case ElementaryDataTypeAST::Type_Byte:
    outs << "ElementaryDataType Byte\n";
    return;
  case ElementaryDataTypeAST::Type_Word:
    outs << "ElementaryDataType Word\n";
    return;
  case ElementaryDataTypeAST::Type_DWord:
    outs << "ElementaryDataType DWord\n";
    return;
  case ElementaryDataTypeAST::Type_Char:
    outs << "ElementaryDataType Char\n";
    return;
  case ElementaryDataTypeAST::Type_Int:
    outs << "ElementaryDataType Int\n";
    return;
  case ElementaryDataTypeAST::Type_DInt:
    outs << "ElementaryDataType DInt\n";
    return;
  case ElementaryDataTypeAST::Type_Real:
    outs << "ElementaryDataType Real\n";
    return;
  case ElementaryDataTypeAST::Type_S5Time:
    outs << "ElementaryDataType S5Time\n";
    return;
  case ElementaryDataTypeAST::Type_Time:
    outs << "ElementaryDataType Time\n";
    return;
  case ElementaryDataTypeAST::Type_TimeOfDay:
    outs << "ElementaryDataType TimeOfDay\n";
    return;
  case ElementaryDataTypeAST::Type_Date:
    outs << "ElementaryDataType Date\n";
    return;
  case ElementaryDataTypeAST::Type_DateAndTime:
    outs << "ElementaryDataType DateAndTime\n";
    return;
  case ElementaryDataTypeAST::Type_Timer:
    outs << "ElementaryDataType Timer\n";
    return;
  case ElementaryDataTypeAST::Type_Counter:
    outs << "ElementaryDataType Counter\n";
    return;
  case ElementaryDataTypeAST::Type_Any:
    outs << "ElementaryDataType Any\n";
    return;
  case ElementaryDataTypeAST::Type_Pointer:
    outs << "ElementaryDataType Pointer\n";
    return;
  case ElementaryDataTypeAST::Type_BlockFC:
    outs << "ElementaryDataType BlockFC\n";
    return;
  case ElementaryDataTypeAST::Type_BlockFB:
    outs << "ElementaryDataType BlockFB\n";
    return;
  case ElementaryDataTypeAST::Type_BlockDB:
    outs << "ElementaryDataType BlockDB\n";
    return;
  case ElementaryDataTypeAST::Type_BlockSDB:
    outs << "ElementaryDataType BlockSDB\n";
    return;
  }
  outs << "<unknown ElementaryDataType, type " << node->getType() << ">\n";
}

void ASTDumper::dump(const StringDataTypeSpecAST *node) {
  INDENT();
  outs << "StringDataTypeSpec [" << node->getMaxLen() << "]\n";
}

void ASTDumper::dump(const ArrayDimensionAST *node) {
  INDENT();
  outs << "ArrayDimension\n";
  dump(node->getMin());
  dump(node->getMax());
}

void ASTDumper::dump(const ArrayDataTypeSpecAST *node) {
  INDENT();
  outs << "ArrayDataTypeSpec\n";
  for (auto const &dim : node->getDimensions())
    dump(dim.get());
  dump(node->getDataType());
}

void ASTDumper::dump(const ComponentDeclarationAST *node) {
  INDENT();
  outs << "ComponentDeclaration " << node->getName() << "\n";
  dump(node->getDataType());
  auto init = node->getInitializer();
  if (init.hasValue())
    dump(init.getValue());
}

void ASTDumper::dump(const StructDataTypeSpecAST *node) {
  INDENT();
  outs << "StructDataTypeSpec\n";
  for (auto const &component : node->getComponents())
    dump(component.get());
}

void ASTDumper::dump(const UserDefinedTypeIdentifierAST *node) {
  INDENT();
  outs << "UserDefinedTypeIdentifier " << node->getName() << "\n";
}

// MARK: C.3

// MARK: C.4 Code Section

void ASTDumper::dump(const CodeSectionAST *node) {
  INDENT();
  outs << "CodeSection\n";
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
  outs << "<unknown instruction, kind" << code->getKind() << ">\n";
}

// MARK: C.5 Value Assignments

void ASTDumper::dump(const ValueAssignmentAST *node) {
  INDENT();
  outs << "ValueAssignment\n";
  dump(node->getExpression());
}

void ASTDumper::dump(const ExpressionAST *expr) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(expr))                         \
    return dump(node);
  dispatch(ExpressionListAST);
  dispatch(RepeatedConstantAST);
  dispatch(IntegerConstantAST);
  dispatch(RealConstantAST);
  dispatch(StringConstantAST);
  dispatch(TimeConstantAST);
  dispatch(SimpleVariableAST);
  dispatch(IndexedVariableAST);
  //    dispatch(AbsoluteVariableAST);
  //    dispatch(VariableInDBAST);
  dispatch(FunctionCallAST);
  dispatch(BinaryExpressionAST);
  dispatch(UnaryExpressionAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  outs << "<unknown expression, kind " << expr->getKind() << ">\n";
}

void ASTDumper::dump(const ExpressionListAST *node) {
  INDENT();
  outs << "ExpressionList\n";
  for (auto &value : node->getValues())
    dump(value.get());
}

void ASTDumper::dump(const RepeatedConstantAST *node) {
  INDENT();
  outs << "RepeatedConstant " << node->getRepetitions() << "\n";
  dump(node->getValue());
}

void ASTDumper::dump(const SimpleVariableAST *node) {
  INDENT();
  if (node->isSymbol()) {
    outs << "Symbol " << node->getName() << "\n";
  } else {
    outs << "SimpleVariable " << node->getName() << "\n";
  }
}

void ASTDumper::dump(const IndexedVariableAST *node) {
  INDENT();
  outs << "IndexedVariable\n";
  dump(node->getBase());
  dump("Indices", node->getIndices());
}

void ASTDumper::dump(const BinaryExpressionAST *node) {
  INDENT();
  outs << "BinaryExpression " << node->getOp() << "\n";
  dump(node->getLhs());
  dump(node->getRhs());
}

void ASTDumper::dump(const UnaryExpressionAST *node) {
  INDENT();
  outs << "UnaryExpression " << node->getOp() << "\n";
  dump(node->getRhs());
}

void ASTDumper::dump(const IntegerConstantAST *node) {
  INDENT();
  outs << "IntegerConstant " << node->getValue();
  if (node->getType())
    outs << " Type " << node->getType();
  outs << "\n";
}

void ASTDumper::dump(const RealConstantAST *node) {
  INDENT();
  outs << "RealConstant " << node->getValue();
  if (node->getType())
    outs << " Type " << node->getType();
  outs << "\n";
}

void ASTDumper::dump(const StringConstantAST *node) {
  INDENT();
  outs << "StringConstant '" << node->getValue() << "'\n";
}

void ASTDumper::dump(const TimeConstantAST *node) {
  INDENT();
  outs << "TimeConstant ";
  outs << llvm::formatv(
      "{0,0+4}-{1,0+2}-{2,0+2} {3,0+2}:{4,0+2}:{5,0+2}.{6,0+3}",
      node->getYear(), node->getMonth(), node->getDay(), node->getHour(),
      node->getMinute(), node->getSec(), node->getMSec());
  outs << " Type " << node->getType() << "\n";
}

// MARK: C.6 Function and Function Block Calls

void ASTDumper::dump(const SubroutineProcessingAST *node) {
  INDENT();
  outs << "Subroutine\n";
  dump(node->getCall());
}

void ASTDumper::dump(const FunctionCallAST *node) {
  INDENT();
  outs << "FunctionCall\n";
  dump(node->getFunction());
  dump("Parameters", node->getParameters());
}

// MARK: C.7 Control Statements

void ASTDumper::dump(const IfThenAST *node) {
  INDENT();
  outs << "IfThen\n";
  dump(node->getCondition());
  dump(node->getCodeBlock());
}

void ASTDumper::dump(const IfThenElseAST *node) {
  INDENT();
  outs << "IfThenElse\n";

  for (auto const &then : node->getThens())
    dump(then.get());

  auto elseBlock = node->getElseBlock();
  if (elseBlock.hasValue()) {
    INDENT();
    outs << "Else\n";
    dump(elseBlock.getValue());
  }
}

void ASTDumper::dump(const CaseValueAST *value) {
#define dispatch(CLASS)                                                        \
  if (const CLASS *node = llvm::dyn_cast<CLASS>(value))                        \
    return dump(node);
  dispatch(CaseValueSingleAST);
  dispatch(CaseValueRangeAST);
#undef dispatch
  // No match, fallback to a generic message
  INDENT();
  outs << "<unknown case value, kind" << value->getKind() << ">\n";
}

void ASTDumper::dump(const CaseValueSingleAST *node) {
  INDENT();
  outs << "CaseValueSingle\n";
  dump(node->getValue());
}

void ASTDumper::dump(const CaseValueRangeAST *node) {
  INDENT();
  outs << "CaseValueRange\n";
  dump(node->getMin());
  dump(node->getMax());
}

void ASTDumper::dump(const CaseBlockAST *node) {
  INDENT();
  outs << "CaseBlock\n";
  for (auto const &value : node->getValues())
    dump(value.get());
  dump(node->getCodeBlock());
}

void ASTDumper::dump(const CaseOfAST *node) {
  INDENT();
  outs << "Case\n";
  dump(node->getExpr());
  outs << "Of\n";
  dump(node->getCode());
  auto elseBlock = node->getElseBlock();
  if (elseBlock.hasValue())
    dump(elseBlock.getValue());
}

void ASTDumper::dump(const ForDoAST *node) {
  INDENT();
  outs << "For\n";
  dump(node->getAssignment());
  indent();
  outs << "To\n";
  dump(node->getLast());
  auto increment = node->getIncrement();
  if (increment.hasValue()) {
    indent();
    outs << "By\n";
    dump(increment.getValue());
  }
  indent();
  outs << "Do\n";
  dump(node->getCode());
}

void ASTDumper::dump(const WhileDoAST *node) {
  INDENT();
  outs << "While\n";
  dump(node->getCondition());
  outs << "Do\n";
  dump(node->getCode());
}

void ASTDumper::dump(const RepeatUntilAST *node) {
  INDENT();
  outs << "Repeat\n";
  dump(node->getCode());
  outs << "Until\n";
  dump(node->getCondition());
}

void ASTDumper::dump(const ContinueAST *node) {
  INDENT();
  outs << "Continue\n";
}

void ASTDumper::dump(const ReturnAST *node) {
  INDENT();
  outs << "Return\n";
}

void ASTDumper::dump(const ExitAST *node) {
  INDENT();
  outs << "Exit\n";
}

void ASTDumper::dump(const GotoAST *node) {
  INDENT();
  outs << "Goto " << node->getLabel() << "\n";
}

void ASTDumper::dump(
    std::string name,
    llvm::ArrayRef<std::unique_ptr<ExpressionAST>> parameters) {
  INDENT();
  outs << name << "\n";
  for (const auto &param : parameters) {
    dump(param.get());
  }
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(const ModuleAST *Node) {
  INDENT();
  outs << "Module:\n";
  for (auto const &unit : *Node)
    dump(unit.get());
}

namespace sclang {

// Public API
void dump(const ModuleAST &module) { ASTDumper(llvm::outs()).dump(&module); }

} // namespace sclang
