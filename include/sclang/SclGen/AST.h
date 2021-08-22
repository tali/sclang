//===- AST.h - Node definition for the Toy AST ----------------------------===//
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
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_AST_H_
#define SCLANG_AST_H_

#include "sclang/SclGen/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace sclang {

// MARK: Attributes

class AttributeAST {
  Location location;
  std::string name;
  std::string value;

public:
  AttributeAST(Location location, llvm::StringRef name, llvm::StringRef value)
      : location(std::move(location)), name(std::move(name)),
        value(std::move(value)) {}

  const Location &loc() const { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::StringRef getValue() const { return value; }
};

// MARK: Expressions

class ExpressionAST {
public:
  enum ExprASTKind {
    Expr_List,
    Expr_RepeatedConstant,
    Expr_IntegerConstant,
    Expr_RealConstant,
    Expr_StringConstant,
    Expr_TimeConstant,
    Expr_SimpleVariable,
    Expr_IndexedVariable,
    Expr_FunctionCall,
    Expr_Binary,
    Expr_Unary,
  };

  ExpressionAST(Location loc, ExprASTKind kind)
      : location(std::move(loc)), kind(kind) {}
  virtual ~ExpressionAST() = default;

  const Location &loc() const { return location; }
  ExprASTKind getKind() const { return kind; }

private:
  Location location;
  ExprASTKind kind;
};

class ExpressionListAST : public ExpressionAST {
  std::vector<std::unique_ptr<ExpressionAST>> values;

public:
  ExpressionListAST(Location loc,
                    std::vector<std::unique_ptr<ExpressionAST>> values)
      : ExpressionAST(std::move(loc), Expr_List), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<ExpressionAST>> getValues() const {
    return values;
  }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_List;
  }
};

class SimpleVariableAST : public ExpressionAST {
  std::string name;
  bool symbol;

public:
  SimpleVariableAST(Location loc, std::string name, bool symbol)
      : ExpressionAST(std::move(loc), Expr_SimpleVariable),
        name(std::move(name)), symbol(symbol) {}

  llvm::StringRef getName() const { return name; }
  bool isSymbol() const { return symbol; }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_SimpleVariable;
  }
};

class IndexedVariableAST : public ExpressionAST {
  std::unique_ptr<ExpressionAST> base;
  std::vector<std::unique_ptr<ExpressionAST>> indices;

public:
  IndexedVariableAST(Location loc, std::unique_ptr<ExpressionAST> base,
                     std::vector<std::unique_ptr<ExpressionAST>> indices)
      : ExpressionAST(std::move(loc), Expr_IndexedVariable),
        base(std::move(base)), indices(std::move(indices)) {}

  const ExpressionAST *getBase() const { return base.get(); }
  llvm::ArrayRef<std::unique_ptr<ExpressionAST>> getIndices() const {
    return indices;
  }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_IndexedVariable;
  }
};

class BinaryExpressionAST : public ExpressionAST {
  Token op;
  std::unique_ptr<ExpressionAST> lhs;
  std::unique_ptr<ExpressionAST> rhs;

public:
  BinaryExpressionAST(Location loc, Token op,
                      std::unique_ptr<ExpressionAST> lhs,
                      std::unique_ptr<ExpressionAST> rhs)
      : ExpressionAST(std::move(loc), Expr_Binary), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  const ExpressionAST *getLhs() const { return lhs.get(); }
  Token getOp() const { return op; }
  const ExpressionAST *getRhs() const { return rhs.get(); }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_Binary;
  }
};

class UnaryExpressionAST : public ExpressionAST {
  Token op;
  std::unique_ptr<ExpressionAST> rhs;

public:
  UnaryExpressionAST(Location loc, Token op, std::unique_ptr<ExpressionAST> rhs)
      : ExpressionAST(std::move(loc), Expr_Unary), op(op), rhs(std::move(rhs)) {
  }

  Token getOp() const { return op; }
  const ExpressionAST *getRhs() const { return rhs.get(); }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_Unary;
  }
};

// MARK: Constants

class ConstantAST : public ExpressionAST {
  Token type;

public:
  ConstantAST(Location loc, ExprASTKind kind, Token type)
      : ExpressionAST(std::move(loc), kind), type(type) {}

  Token getType() const { return type; }
};

class RepeatedConstantAST : public ExpressionAST {
  int32_t repetitions;
  std::unique_ptr<ExpressionAST> value;

public:
  RepeatedConstantAST(Location loc, int repetitions,
                      std::unique_ptr<ExpressionAST> value)
      : ExpressionAST(std::move(loc), Expr_RepeatedConstant),
        repetitions(repetitions), value(std::move(value)) {}

  int getRepetitions() const { return repetitions; }
  const ExpressionAST *getValue() const { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_RepeatedConstant;
  }
};

class IntegerConstantAST : public ConstantAST {
  int32_t value;

public:
  IntegerConstantAST(Location loc, int32_t value, Token type)
      : ConstantAST(std::move(loc), Expr_IntegerConstant, type), value(value) {}

  int32_t getValue() const { return value; }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_IntegerConstant;
  }
};

class RealConstantAST : public ConstantAST {
  float value;

public:
  RealConstantAST(Location loc, float value, Token type)
      : ConstantAST(std::move(loc), Expr_RealConstant, type), value(value) {}

  float getValue() const { return value; }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_RealConstant;
  }
};

class StringConstantAST : public ConstantAST {
  std::string value;

public:
  StringConstantAST(Location loc, llvm::StringRef value, Token type)
      : ConstantAST(std::move(loc), Expr_StringConstant, type), value(value) {}

  llvm::StringRef getValue() const { return value; }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_StringConstant;
  }
};

class TimeConstantAST : public ConstantAST {
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int sec;
  int msec;

public:
  TimeConstantAST(Location loc, int year, int month, int day, int hour,
                  int minute, int sec, int msec, Token type)
      : ConstantAST(std::move(loc), Expr_TimeConstant, type), year(year),
        month(month), day(day), hour(hour), minute(minute), sec(sec),
        msec(msec) {}

  int getYear() const { return year; }
  int getMonth() const { return month; }
  int getDay() const { return day; }
  int getHour() const { return hour; }
  int getMinute() const { return minute; }
  int getSec() const { return sec; }
  int getMSec() const { return msec; }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_TimeConstant;
  }
};

// MARK: Data Types

class DataTypeSpecAST {
public:
  enum DataTypeASTKind {
    DataType_Elementary,
    DataType_String,
    DataType_Array,
    DataType_Struct,
    DataType_UDT,
    // Type_Parameter, // uses Type_Elementary
  };

  DataTypeSpecAST(Location loc, DataTypeASTKind kind)
      : location(std::move(loc)), kind(kind) {}
  virtual ~DataTypeSpecAST() = default;

  const Location &loc() const { return location; }
  DataTypeASTKind getKind() const { return kind; }

private:
  Location location;
  DataTypeASTKind kind;
};

class ElementaryDataTypeAST : public DataTypeSpecAST {
public:
  enum ElementaryTypeASTKind {
    // Void return type
    Type_Void,
    // Bit Data Type
    Type_Bool,
    Type_Byte,
    Type_Word,
    Type_DWord,
    // Character Type
    Type_Char,
    // Numeric Data Type
    Type_Int,
    Type_DInt,
    Type_Real,
    // Time Type
    Type_S5Time,
    Type_Time,
    Type_TimeOfDay,
    Type_Date,
    Type_DateAndTime,
    // Parameter Data Type
    Type_Timer,
    Type_Counter,
    Type_Any,
    Type_Pointer,
    Type_BlockFC,
    Type_BlockFB,
    Type_BlockDB,
    Type_BlockSDB,
  };

  ElementaryDataTypeAST(Location loc, ElementaryTypeASTKind type)
      : DataTypeSpecAST(std::move(loc), DataType_Elementary), typeKind(type) {}

  ElementaryTypeASTKind getType() const { return typeKind; }

  /// LLVM style RTTI
  static bool classof(const DataTypeSpecAST *d) {
    return d->getKind() == DataType_Elementary;
  }

private:
  ElementaryTypeASTKind typeKind;
};

class StringDataTypeSpecAST : public DataTypeSpecAST {
  uint8_t maxLen;

public:
  StringDataTypeSpecAST(Location loc, uint8_t maxLen = 254)
      : DataTypeSpecAST(std::move(loc), DataType_String), maxLen(maxLen) {}

  uint8_t getMaxLen() const { return maxLen; }

  /// LLVM style RTTI
  static bool classof(const DataTypeSpecAST *d) {
    return d->getKind() == DataType_String;
  }
};

class ArrayDimensionAST {
  Location location;
  std::unique_ptr<ExpressionAST> min;
  std::unique_ptr<ExpressionAST> max;

public:
  ArrayDimensionAST(Location loc, std::unique_ptr<ExpressionAST> min,
                    std::unique_ptr<ExpressionAST> max)
      : location(std::move(loc)), min(std::move(min)), max(std::move(max)) {}

  const Location &loc() const { return location; }
  const ExpressionAST *getMin() const { return min.get(); }
  const ExpressionAST *getMax() const { return max.get(); }
};

class ArrayDataTypeSpecAST : public DataTypeSpecAST {
  std::vector<std::unique_ptr<ArrayDimensionAST>> dimensions;
  std::unique_ptr<DataTypeSpecAST> dataType;

public:
  ArrayDataTypeSpecAST(
      Location loc, std::vector<std::unique_ptr<ArrayDimensionAST>> dimensions,
      std::unique_ptr<DataTypeSpecAST> dataType)
      : DataTypeSpecAST(std::move(loc), DataType_Array),
        dimensions(std::move(dimensions)), dataType(std::move(dataType)) {}

  llvm::ArrayRef<std::unique_ptr<ArrayDimensionAST>> getDimensions() const {
    return dimensions;
  }
  const DataTypeSpecAST *getDataType() const { return dataType.get(); }
  /// LLVM style RTTI
  static bool classof(const DataTypeSpecAST *d) {
    return d->getKind() == DataType_Array;
  }
};

class ComponentDeclarationAST {
  Location location;
  std::string name;
  std::unique_ptr<DataTypeSpecAST> dataType;
  llvm::Optional<std::unique_ptr<ExpressionAST>> initializer;

public:
  ComponentDeclarationAST(Location loc, std::string name,
                          std::unique_ptr<DataTypeSpecAST> dataType,
                          llvm::Optional<std::unique_ptr<ExpressionAST>> init)
      : location(std::move(loc)), name(std::move(name)),
        dataType(std::move(dataType)), initializer(std::move(init)) {}

  const Location &loc() const { return location; }
  llvm::StringRef getName() const { return name; }
  const DataTypeSpecAST *getDataType() const { return dataType.get(); }
  llvm::Optional<const ExpressionAST *> getInitializer() const {
    if (!initializer.hasValue())
      return llvm::NoneType();
    return initializer.getValue().get();
  }
};

class StructDataTypeSpecAST : public DataTypeSpecAST {
  Location location;
  std::vector<std::unique_ptr<ComponentDeclarationAST>> components;

public:
  StructDataTypeSpecAST(
      Location loc,
      std::vector<std::unique_ptr<ComponentDeclarationAST>> components)
      : DataTypeSpecAST(std::move(loc), DataType_Struct),
        components(std::move(components)) {}

  llvm::ArrayRef<std::unique_ptr<ComponentDeclarationAST>>
  getComponents() const {
    return components;
  }

  /// LLVM style RTTI
  static bool classof(const DataTypeSpecAST *d) {
    return d->getKind() == DataType_Struct;
  }
};

class UserDefinedTypeIdentifierAST : public DataTypeSpecAST {
  std::string name;

public:
  UserDefinedTypeIdentifierAST(Location loc, std::string name)
      : DataTypeSpecAST(std::move(loc), DataType_UDT), name(std::move(name)) {}

  llvm::StringRef getName() const { return name; }

  /// LLVM style RTTI
  static bool classof(const DataTypeSpecAST *d) {
    return d->getKind() == DataType_UDT;
  }
};

// MARK: Code Section

class InstructionAST {
public:
  enum InstrASTKind {
    Instr_JumpLabel,
    Instr_Assignment,
    Instr_Subroutine,
    Instr_IfThenElse,
    Instr_CaseOf,
    Instr_ForDo,
    Instr_WhileDo,
    Instr_RepeatUntil,
    Instr_Continue,
    Instr_Return,
    Instr_Exit,
    Instr_Goto,
    Instr_DebugPrint,
  };

  InstructionAST(Location loc, InstrASTKind kind)
      : location(std::move(loc)), kind(kind) {}
  virtual ~InstructionAST() = default;

  InstrASTKind getKind() const { return kind; }
  const Location &loc() const { return location; }

private:
  Location location;
  InstrASTKind kind;
};

class JumpLabelAST : public InstructionAST {
  std::string label;

public:
  JumpLabelAST(Location loc, std::string label)
      : InstructionAST(std::move(loc), Instr_JumpLabel),
        label(std::move(label)) {}

  llvm::StringRef getLabel() const { return label; }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_JumpLabel;
  }
};

class ValueAssignmentAST : public InstructionAST {
  std::unique_ptr<ExpressionAST> expression;

public:
  ValueAssignmentAST(Location loc, std::unique_ptr<ExpressionAST> expression)
      : InstructionAST(std::move(loc), Instr_Assignment),
        expression(std::move(expression)) {}

  const ExpressionAST *getExpression() const { return expression.get(); }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Assignment;
  }
};

class CodeSectionAST {
  Location location;
  std::vector<std::unique_ptr<InstructionAST>> instructions;

public:
  CodeSectionAST(Location loc,
                 std::vector<std::unique_ptr<InstructionAST>> instructions)
      : location(std::move(loc)), instructions(std::move(instructions)) {}

  llvm::ArrayRef<std::unique_ptr<InstructionAST>> getInstructions() const {
    return instructions;
  }
};

// MARK: Function Calls

class FunctionCallAST : public ExpressionAST {

  std::unique_ptr<ExpressionAST> function;
  std::vector<std::unique_ptr<ExpressionAST>> parameters;

public:
  FunctionCallAST(Location loc, std::unique_ptr<ExpressionAST> function,
                  std::vector<std::unique_ptr<ExpressionAST>> parameters)
      : ExpressionAST(std::move(loc), Expr_FunctionCall),
        function(std::move(function)), parameters(std::move(parameters)) {}

  const ExpressionAST *getFunction() const { return function.get(); }
  llvm::ArrayRef<std::unique_ptr<ExpressionAST>> getParameters() const {
    return parameters;
  }

  /// LLVM style RTTI
  static bool classof(const ExpressionAST *e) {
    return e->getKind() == Expr_FunctionCall;
  }
};

class SubroutineProcessingAST : public InstructionAST {

  std::unique_ptr<ExpressionAST> call;

public:
  SubroutineProcessingAST(Location loc, std::unique_ptr<ExpressionAST> call)
      : InstructionAST(std::move(loc), Instr_Subroutine),
        call(std::move(call)) {}

  const FunctionCallAST *getCall() const {
    return llvm::cast<FunctionCallAST>(call.get());
  }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Subroutine;
  }
};

// MARK: Control Statements

class IfThenAST {
  Location location;
  std::unique_ptr<ExpressionAST> condition;
  std::unique_ptr<CodeSectionAST> code;

public:
  IfThenAST(Location loc, std::unique_ptr<ExpressionAST> condition,
            std::unique_ptr<CodeSectionAST> code)
      : location(std::move(loc)), condition(std::move(condition)),
        code(std::move(code)) {}

  const Location &loc() const { return location; }
  const ExpressionAST *getCondition() const { return condition.get(); }
  const CodeSectionAST *getCodeBlock() const { return code.get(); }
};

class IfThenElseAST : public InstructionAST {
  std::vector<std::unique_ptr<IfThenAST>> thens;
  llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock;

public:
  IfThenElseAST(Location loc, std::vector<std::unique_ptr<IfThenAST>> thens,
                llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock)
      : InstructionAST(std::move(loc), Instr_IfThenElse),
        thens(std::move(thens)), elseBlock(std::move(elseBlock)) {}

  llvm::ArrayRef<std::unique_ptr<IfThenAST>> getThens() const { return thens; }
  llvm::Optional<const CodeSectionAST *> getElseBlock() const {
    if (!elseBlock.hasValue())
      return llvm::NoneType();
    return elseBlock.getValue().get();
  };

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_IfThenElse;
  }
};

class CaseValueAST {
public:
  enum CaseValueASTKind {
    Value_Single,
    Value_Range,
  };

  CaseValueAST(Location loc, CaseValueASTKind kind)
      : location(std::move(loc)), kind(kind) {}

  const Location &loc() const { return location; }
  CaseValueASTKind getKind() const { return kind; }

private:
  Location location;
  CaseValueASTKind kind;
};

class CaseValueSingleAST : public CaseValueAST {
  std::unique_ptr<ExpressionAST> value;

public:
  CaseValueSingleAST(Location loc, std::unique_ptr<ExpressionAST> value)
      : CaseValueAST(std::move(loc), Value_Single), value(std::move(value)) {}

  const ExpressionAST *getValue() const { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const CaseValueAST *v) {
    return v->getKind() == Value_Single;
  }
};

class CaseValueRangeAST : public CaseValueAST {
  std::unique_ptr<ExpressionAST> min;
  std::unique_ptr<ExpressionAST> max;

public:
  CaseValueRangeAST(Location loc, std::unique_ptr<ExpressionAST> min,
                    std::unique_ptr<ExpressionAST> max)
      : CaseValueAST(std::move(loc), Value_Range), min(std::move(max)),
        max(std::move(max)) {}

  const ExpressionAST *getMin() const { return min.get(); }
  const ExpressionAST *getMax() const { return max.get(); }

  /// LLVM style RTTI
  static bool classof(const CaseValueAST *v) {
    return v->getKind() == Value_Range;
  }
};

class CaseBlockAST {
  Location location;
  std::vector<std::unique_ptr<CaseValueAST>> values;
  std::unique_ptr<CodeSectionAST> code;

public:
  CaseBlockAST(Location loc, std::vector<std::unique_ptr<CaseValueAST>> values,
               std::unique_ptr<CodeSectionAST> code)
      : location(std::move(loc)), values(std::move(values)),
        code(std::move(code)) {}

  const Location &loc() const { return location; }
  llvm::ArrayRef<std::unique_ptr<CaseValueAST>> getValues() const {
    return values;
  }
  const CodeSectionAST *getCodeBlock() const { return code.get(); }
};

class CaseOfAST : public InstructionAST {
  std::unique_ptr<ExpressionAST> expr;
  std::unique_ptr<CodeSectionAST> code;
  llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock;

public:
  CaseOfAST(Location loc, std::unique_ptr<ExpressionAST> expr,
            std::unique_ptr<CodeSectionAST> code,
            llvm::Optional<std::unique_ptr<CodeSectionAST>> elseBlock)
      : InstructionAST(std::move(loc), Instr_CaseOf), expr(std::move(expr)),
        code(std::move(code)), elseBlock(std::move(elseBlock)) {}

  const ExpressionAST *getExpr() const { return expr.get(); }
  const CodeSectionAST *getCode() const { return code.get(); };
  llvm::Optional<const CodeSectionAST *> getElseBlock() const {
    if (!elseBlock.hasValue())
      return llvm::NoneType();
    return elseBlock.getValue().get();
  };

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_CaseOf;
  }
};

class ForDoAST : public InstructionAST {
  std::unique_ptr<ExpressionAST> assignment;
  std::unique_ptr<ExpressionAST> last;
  llvm::Optional<std::unique_ptr<ExpressionAST>> increment;
  std::unique_ptr<CodeSectionAST> code;

public:
  ForDoAST(Location loc, std::unique_ptr<ExpressionAST> assignment,
           std::unique_ptr<ExpressionAST> last,
           llvm::Optional<std::unique_ptr<ExpressionAST>> increment,
           std::unique_ptr<CodeSectionAST> code)
      : InstructionAST(std::move(loc), Instr_ForDo),
        assignment(std::move(assignment)), last(std::move(last)),
        increment(std::move(increment)), code(std::move(code)) {}

  const ExpressionAST *getAssignment() const { return assignment.get(); }
  const ExpressionAST *getLast() const { return last.get(); }
  llvm::Optional<const ExpressionAST *> getIncrement() const {
    if (!increment.hasValue())
      return llvm::NoneType();
    return increment.getValue().get();
  };
  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_ForDo;
  }
};

class WhileDoAST : public InstructionAST {
  std::unique_ptr<ExpressionAST> condition;
  std::unique_ptr<CodeSectionAST> code;

public:
  WhileDoAST(Location loc, std::unique_ptr<ExpressionAST> condition,
             std::unique_ptr<CodeSectionAST> code)
      : InstructionAST(std::move(loc), Instr_WhileDo),
        condition(std::move(condition)), code(std::move(code)) {}

  const ExpressionAST *getCondition() const { return condition.get(); }
  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_WhileDo;
  }
};

class RepeatUntilAST : public InstructionAST {
  std::unique_ptr<ExpressionAST> condition;
  std::unique_ptr<CodeSectionAST> code;

public:
  RepeatUntilAST(Location loc, std::unique_ptr<ExpressionAST> condition,
                 std::unique_ptr<CodeSectionAST> code)
      : InstructionAST(std::move(loc), Instr_RepeatUntil),
        condition(std::move(condition)), code(std::move(code)) {}

  const ExpressionAST *getCondition() const { return condition.get(); }
  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_RepeatUntil;
  }
};

class ContinueAST : public InstructionAST {
public:
  ContinueAST(Location loc) : InstructionAST(std::move(loc), Instr_Continue) {}

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Continue;
  }
};

class ReturnAST : public InstructionAST {
public:
  ReturnAST(Location loc) : InstructionAST(std::move(loc), Instr_Return) {}

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Return;
  }
};

class ExitAST : public InstructionAST {
public:
  ExitAST(Location loc) : InstructionAST(std::move(loc), Instr_Exit) {}

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Exit;
  }
};

class GotoAST : public InstructionAST {
  std::string label;

public:
  GotoAST(Location loc, std::string label)
      : InstructionAST(std::move(loc), Instr_Goto), label(std::move(label)) {}

  llvm::StringRef getLabel() const { return label; }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_Goto;
  }
};

class DebugPrintAST : public InstructionAST {
  std::string msg;

public:
  DebugPrintAST(Location loc, std::string msg)
      : InstructionAST(std::move(loc), Instr_DebugPrint), msg(std::move(msg)) {}

  llvm::StringRef getMsg() const { return msg; }

  /// LLVM style RTTI
  static bool classof(const InstructionAST *i) {
    return i->getKind() == Instr_DebugPrint;
  }
};

// MARK: Declaration Sections

class DeclarationSubsectionAST {
public:
  enum DeclarationASTKind {
    Decl_Constant,
    Decl_JumpLabel,
    Decl_Variable,
  };

  DeclarationSubsectionAST(DeclarationASTKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~DeclarationSubsectionAST() = default;

  DeclarationASTKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const DeclarationASTKind kind;
  Location location;
};

class ConstantDeclarationAST {
  Location location;
  std::string name;
  std::unique_ptr<ExpressionAST> value;

public:
  ConstantDeclarationAST(Location loc, llvm::StringRef name,
                         std::unique_ptr<ExpressionAST> value)
      : location(loc), name(std::move(name)), value(std::move(value)) {}

  const Location &loc() const { return location; }
  llvm::StringRef getName() const { return name; }
  const ExpressionAST *getValue() const { return value.get(); }
};

class ConstantDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<ConstantDeclarationAST>> values;

public:
  ConstantDeclarationSubsectionAST(
      Location loc, std::vector<std::unique_ptr<ConstantDeclarationAST>> values)
      : DeclarationSubsectionAST(Decl_Constant, std::move(loc)),
        values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<ConstantDeclarationAST>> getValues() const {
    return values;
  }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) {
    return D->getKind() == Decl_Constant;
  }
};

class JumpLabelDeclarationAST {
  Location location;
  std::string identifier;

public:
  JumpLabelDeclarationAST(Location loc, llvm::StringRef identifier)
      : location(loc), identifier(identifier) {}

  const Location &loc() const { return location; }
  llvm::StringRef getIdentifier() const { return identifier; }
};

class JumpLabelDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<JumpLabelDeclarationAST>> values;

public:
  JumpLabelDeclarationSubsectionAST(
      Location loc,
      std::vector<std::unique_ptr<JumpLabelDeclarationAST>> values)
      : DeclarationSubsectionAST(Decl_JumpLabel, loc),
        values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<JumpLabelDeclarationAST>> getValues() const {
    return values;
  }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) {
    return D->getKind() == Decl_JumpLabel;
  }
};

class VariableIdentifierAST {
  Location location;
  std::string identifier;
  std::vector<std::unique_ptr<AttributeAST>> attributes;

public:
  const Location &loc() const { return location; }
  llvm::StringRef getIdentifier() const { return identifier; }
  llvm::ArrayRef<std::unique_ptr<AttributeAST>> getAttributes() const {
    return attributes;
  }

  VariableIdentifierAST(Location loc, llvm::StringRef identifier,
                        std::vector<std::unique_ptr<AttributeAST>> attributes)
      : location(loc), identifier(identifier),
        attributes(std::move(attributes)) {}
};

class VariableDeclarationAST {
  Location location;
  std::vector<std::unique_ptr<VariableIdentifierAST>> vars;
  std::unique_ptr<DataTypeSpecAST> dataType;
  llvm::Optional<std::unique_ptr<ExpressionAST>> initializer;

public:
  const Location &loc() const { return location; }
  llvm::ArrayRef<std::unique_ptr<VariableIdentifierAST>> getVars() const {
    return vars;
  }
  const DataTypeSpecAST *getDataType() const { return dataType.get(); }
  llvm::Optional<ExpressionAST *> const getInitializer() const {
    if (!initializer.hasValue())
      return llvm::NoneType();
    return initializer.getValue().get();
  }

  VariableDeclarationAST(
      Location loc, std::vector<std::unique_ptr<VariableIdentifierAST>> vars,
      std::unique_ptr<DataTypeSpecAST> dataType,
      llvm::Optional<std::unique_ptr<ExpressionAST>> init)
      : location(loc), vars(std::move(vars)), dataType(std::move(dataType)),
        initializer(std::move(init)) {}
};

class VariableDeclarationSubsectionAST : public DeclarationSubsectionAST {
public:
  enum Var_Kind { Var, VarTemp, VarInput, VarOutput, VarInOut };

private:
  Var_Kind kind;
  std::vector<std::unique_ptr<VariableDeclarationAST>> values;

public:
  VariableDeclarationSubsectionAST(
      Location loc, Var_Kind kind,
      std::vector<std::unique_ptr<VariableDeclarationAST>> values)
      : DeclarationSubsectionAST(Decl_Variable, loc), kind(kind),
        values(std::move(values)) {}

  Var_Kind getKind() const { return kind; }
  llvm::ArrayRef<std::unique_ptr<VariableDeclarationAST>> getValues() const {
    return values;
  }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) {
    return D->getKind() == Decl_Variable;
  }
};

class DeclarationSectionAST {
  Location location;
  std::vector<std::unique_ptr<DeclarationSubsectionAST>> declarations;

public:
  const Location &loc() { return location; }
  llvm::ArrayRef<std::unique_ptr<DeclarationSubsectionAST>> getDecls() const {
    return declarations;
  }

  DeclarationSectionAST(
      Location loc,
      std::vector<std::unique_ptr<DeclarationSubsectionAST>> decls)
      : location(loc), declarations(std::move(decls)) {}
};

// MARK: Units

/// SCL program unit
class UnitAST {
public:
  enum UnitASTKind {
    Unit_OrganizationBlock,
    Unit_Function,
    Unit_FunctionBlock,
    Unit_DataBlock,
    Unit_UserDefinedDataType,
  };

  UnitAST(UnitASTKind kind, const std::string &identifier, Location location,
          std::vector<std::unique_ptr<AttributeAST>> attrs,
          std::unique_ptr<DeclarationSectionAST> declarations)
      : kind(kind), identifier(identifier), attributes(std::move(attrs)),
        declarations(std::move(declarations)), location(location) {}
  virtual ~UnitAST() = default;

  UnitASTKind getKind() const { return kind; }
  llvm::StringRef getIdentifier() const { return identifier; }
  llvm::ArrayRef<std::unique_ptr<AttributeAST>> getAttributes() const {
    return attributes;
  }
  const DeclarationSectionAST *getDeclarations() const {
    return declarations.get();
  }

  const Location &loc() const { return location; }

private:
  const UnitASTKind kind;
  std::string identifier;
  std::vector<std::unique_ptr<AttributeAST>> attributes;
  std::unique_ptr<DeclarationSectionAST> declarations;
  Location location;
};

/// A block-list of expressions.
using UnitASTList = std::vector<std::unique_ptr<UnitAST>>;

class OrganizationBlockAST : public UnitAST {
  std::unique_ptr<CodeSectionAST> code;

public:
  OrganizationBlockAST(const std::string &identifier, Location loc,
                       std::vector<std::unique_ptr<AttributeAST>> attrs,
                       std::unique_ptr<DeclarationSectionAST> declarations,
                       std::unique_ptr<CodeSectionAST> code)
      : UnitAST(Unit_OrganizationBlock, identifier, std::move(loc),
                std::move(attrs), std::move(declarations)),
        code(std::move(code)) {}

  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const UnitAST *u) {
    return u->getKind() == Unit_OrganizationBlock;
  }
};

class FunctionAST : public UnitAST {
  std::unique_ptr<DataTypeSpecAST> type;
  std::unique_ptr<CodeSectionAST> code;

public:
  FunctionAST(const std::string &identifier, Location loc,
              std::unique_ptr<DataTypeSpecAST> type,
              std::vector<std::unique_ptr<AttributeAST>> attrs,
              std::unique_ptr<DeclarationSectionAST> declarations,
              std::unique_ptr<CodeSectionAST> code)
      : UnitAST(Unit_Function, identifier, std::move(loc), std::move(attrs),
                std::move(declarations)),
        type(std::move(type)), code(std::move(code)) {}

  const DataTypeSpecAST *getType() const { return type.get(); }
  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const UnitAST *u) {
    return u->getKind() == Unit_Function;
  }
};

class FunctionBlockAST : public UnitAST {
  std::unique_ptr<CodeSectionAST> code;

public:
  FunctionBlockAST(const std::string &identifier, Location loc,
                   std::vector<std::unique_ptr<AttributeAST>> attrs,
                   std::unique_ptr<DeclarationSectionAST> declarations,
                   std::unique_ptr<CodeSectionAST> code)
      : UnitAST(Unit_FunctionBlock, identifier, loc, std::move(attrs),
                std::move(declarations)),
        code(std::move(code)) {}

  const CodeSectionAST *getCode() const { return code.get(); }

  /// LLVM style RTTI
  static bool classof(const UnitAST *u) {
    return u->getKind() == Unit_FunctionBlock;
  }
};

class DataBlockAST : public UnitAST {
  std::unique_ptr<DataTypeSpecAST> type;
  std::unique_ptr<CodeSectionAST> assignments;

public:
  DataBlockAST(const std::string &identifier, Location loc,
               std::vector<std::unique_ptr<AttributeAST>> attrs,
               std::unique_ptr<DeclarationSectionAST> declarations,
               std::unique_ptr<DataTypeSpecAST> type,
               std::unique_ptr<CodeSectionAST> assignments)
      : UnitAST(Unit_DataBlock, identifier, loc, std::move(attrs),
                std::move(declarations)),
        type(std::move(type)), assignments(std::move(assignments)) {}

  const DataTypeSpecAST *getType() const { return type.get(); }
  const CodeSectionAST *getAssignments() const { return assignments.get(); }

  /// LLVM style RTTI
  static bool classof(const UnitAST *u) {
    return u->getKind() == Unit_DataBlock;
  }
};

class UserDefinedTypeAST : public UnitAST {
  std::unique_ptr<DataTypeSpecAST> type;

public:
  UserDefinedTypeAST(const std::string &identifier, Location loc,
                     std::vector<std::unique_ptr<AttributeAST>> attrs,
                     std::unique_ptr<DeclarationSectionAST> declarations,
                     std::unique_ptr<DataTypeSpecAST> type)
      : UnitAST(Unit_UserDefinedDataType, std::move(identifier), std::move(loc),
                std::move(attrs), std::move(declarations)),
        type(std::move(type)) {}

  const DataTypeSpecAST *getType() const { return type.get(); }

  /// LLVM style RTTI
  static bool classof(const UnitAST *u) {
    return u->getKind() == Unit_UserDefinedDataType;
  }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<UnitAST>> units;

public:
  ModuleAST(std::vector<std::unique_ptr<UnitAST>> units)
      : units(std::move(units)) {}

  auto begin() const -> decltype(units.begin()) { return units.begin(); }
  auto end() const -> decltype(units.end()) { return units.end(); }
};

void dump(const ModuleAST &);

} // namespace sclang

#endif // SCLANG_AST_H_
