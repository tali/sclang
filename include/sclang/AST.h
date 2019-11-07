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

#include "sclang/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace sclang {

// MARK: B.1.1 Literals

// MARK: C.5 Value Assignments

class Constant {
  Location location;

public:
  Constant(Location loc, std::string str)
    : location(std::move(loc)) {}
  Constant(Location loc, double value)
    : location(std::move(loc)) {}
};

// MARK: C.4 Code Section

// MARK: C.3 Data Types in SCL

// MARK: C.2 Structure of Declaration Sections

/// TODO: TBD
class DataTypeSpecAST {
  Location location;

public:
  DataTypeSpecAST(Location loc)
    : location(std::move(loc)) {}
};

/// TODO: TBD
class DataTypeInitAST {
  Location location;
  std::vector<std::unique_ptr<Constant>> list;

public:
  DataTypeInitAST(Location loc, std::vector<std::unique_ptr<Constant>> list)
   : location(std::move(loc)), list(std::move(list)) {}
};

struct SimpleExpressionAST {};



class DeclarationSubsectionAST {
public:
  enum DeclarationASTKind {
    Decl_Constant,
    Decl_JumpLabel,
    Decl_VarTemp,
    Decl_VarStatic,
    Decl_Parameter,
  };

  DeclarationSubsectionAST(DeclarationASTKind kind, Location location)
        : kind(kind), location(location) {}

  virtual ~DeclarationSubsectionAST() = default;

  DeclarationASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const DeclarationASTKind kind;
  Location location;
};

class ConstantDeclarationAST {
  Location location;
  std::string identifier;
  std::unique_ptr<SimpleExpressionAST> value;

public:
  ConstantDeclarationAST(Location loc, llvm::StringRef identifier, std::unique_ptr<SimpleExpressionAST> value)
    : location(loc), identifier(identifier), value(std::move(value)) {}

  const Location &loc() { return location; }
  llvm::StringRef getIdentifier() { return identifier; }
  SimpleExpressionAST* getValue() { return value.get(); }
};

class ConstantDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<ConstantDeclarationAST>> values;

public:
  ConstantDeclarationSubsectionAST(Location loc, std::vector<std::unique_ptr<ConstantDeclarationAST>> values)
    : DeclarationSubsectionAST(Decl_Constant, std::move(loc)), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<ConstantDeclarationAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) { return D->getKind() == Decl_Constant; }
};

class JumpLabelDeclarationAST {
  Location location;
  std::string identifier;

public:
  JumpLabelDeclarationAST(Location loc, llvm::StringRef identifier)
    : location(loc), identifier(identifier) {}

  const Location &loc() { return location; }
  llvm::StringRef getIdentifier() { return identifier; }
};

class JumpLabelDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<JumpLabelDeclarationAST>> values;

public:
  JumpLabelDeclarationSubsectionAST(Location loc, std::vector<std::unique_ptr<JumpLabelDeclarationAST>> values)
    : DeclarationSubsectionAST(Decl_JumpLabel, loc), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<JumpLabelDeclarationAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) { return D->getKind() == Decl_JumpLabel; }
};

class VariableAttributeAST {
  Location location;
  std::string key;
  std::string value;

public:
  VariableAttributeAST(Location loc, llvm::StringRef key, llvm::StringRef value)
    : location(loc),  key(key), value(value) {}

  const Location &loc() { return location; }
  llvm::StringRef getKey() { return key; }
  llvm::StringRef getValue() { return value; }
};

class VariableIdentifierAST {
  Location location;
  std::string identifier;
  std::vector<std::unique_ptr<VariableAttributeAST>> attributes;

public:
  const Location &loc() { return location; }
    llvm::StringRef getIdentifier() { return identifier; }
    llvm::ArrayRef<std::unique_ptr<VariableAttributeAST>> getAttributes() { return attributes; }

    VariableIdentifierAST(Location loc, llvm::StringRef identifier,
                           std::vector<std::unique_ptr<VariableAttributeAST>> attributes)
      : location(loc), identifier(identifier), attributes(std::move(attributes)) {}
};

class VariableDeclarationAST {
  Location location;
  std::vector<std::unique_ptr<VariableIdentifierAST>> vars;
  std::unique_ptr<DataTypeSpecAST> dataType;
  llvm::Optional<std::unique_ptr<DataTypeInitAST>> initializer;

public:
  const Location &loc() { return location; }
  llvm::ArrayRef<std::unique_ptr<VariableIdentifierAST>> getVars() { return vars; }
  DataTypeSpecAST *getDataType() { return dataType.get(); }
  llvm::Optional<DataTypeInitAST*> getInitializer() {
    if (!initializer.hasValue())
      return llvm::NoneType();
    return initializer.getValue().get();
  }

  VariableDeclarationAST(Location loc,
                         std::vector<std::unique_ptr<VariableIdentifierAST>> vars,
                         std::unique_ptr<DataTypeSpecAST> dataType,
                         llvm::Optional<std::unique_ptr<DataTypeInitAST>> init)
    : location(loc), vars(std::move(vars)),
      dataType(std::move(dataType)), initializer(std::move(init))
  {}
};

class VariableDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<VariableDeclarationAST>> values;

public:
  VariableDeclarationSubsectionAST(Location loc, std::vector<std::unique_ptr<VariableDeclarationAST>> values)
    : DeclarationSubsectionAST(Decl_VarStatic, loc), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<VariableDeclarationAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) { return D->getKind() == Decl_VarStatic; }
};

class TempVariableDeclarationSubsectionAST : public DeclarationSubsectionAST {
  std::vector<std::unique_ptr<VariableDeclarationAST>> values;

public:
  TempVariableDeclarationSubsectionAST(Location loc, std::vector<std::unique_ptr<VariableDeclarationAST>> values)
    : DeclarationSubsectionAST(Decl_VarTemp, loc), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<VariableDeclarationAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const DeclarationSubsectionAST *D) { return D->getKind() == Decl_VarTemp; }
};

class DeclarationSectionAST {
  Location location;
  std::vector<std::unique_ptr<DeclarationSubsectionAST>> declarations;

public:
  const Location &loc() { return location; }
  llvm::ArrayRef<std::unique_ptr<DeclarationSubsectionAST>> getDecls() { return declarations; }

  DeclarationSectionAST(Location loc, std::vector<std::unique_ptr<DeclarationSubsectionAST>> decls)
    : location(loc), declarations(std::move(decls)) {}
};

class CodeSectionAST {
  Location location;

public:
  CodeSectionAST(Location loc)
    : location(loc) {}
};

class AssignmentsSectionAST {
  Location location;

public:
  AssignmentsSectionAST(Location loc)
    : location(loc) {}
};

// MARK: C.1 Subunits of SCL Source Files

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

  UnitAST(UnitASTKind kind, const std::string & identifier, Location location, std::unique_ptr<DeclarationSectionAST> declarations)
      : kind(kind), identifier(identifier), declarations(std::move(declarations)), location(location) {}

  virtual ~UnitAST() = default;

  UnitASTKind getKind() const { return kind; }
  llvm::StringRef getIdentifier() { return identifier; }
  DeclarationSectionAST * getDeclarations() { return declarations.get(); }

  const Location &loc() { return location; }

private:
  const UnitASTKind kind;
  std::string identifier;
  std::unique_ptr<DeclarationSectionAST> declarations;
  Location location;
};

/// A block-list of expressions.
using UnitASTList = std::vector<std::unique_ptr<UnitAST>>;

class OrganizationBlockAST : public UnitAST {
  std::unique_ptr<CodeSectionAST> code;

public:
  OrganizationBlockAST(const std::string & identifier, Location loc, std::unique_ptr<DeclarationSectionAST> declarations, std::unique_ptr<CodeSectionAST> code)
    : UnitAST(Unit_OrganizationBlock, identifier, std::move(loc), std::move(declarations)), code(std::move(code)) {}

  /// LLVM style RTTI
  static bool classof(const UnitAST *C) { return C->getKind() == Unit_OrganizationBlock; }
};

class FunctionAST : public UnitAST {
  std::unique_ptr<DataTypeSpecAST> type;
  std::unique_ptr<CodeSectionAST> code;

public:
  FunctionAST(const std::string & identifier, Location loc, std::unique_ptr<DataTypeSpecAST> type, std::unique_ptr<DeclarationSectionAST> declarations, std::unique_ptr<CodeSectionAST> code)
    : UnitAST(Unit_Function, identifier, std::move(loc), std::move(declarations)), type(std::move(type)), code(std::move(code)) {}

  /// LLVM style RTTI
  static bool classof(const UnitAST *C) { return C->getKind() == Unit_Function; }
};

class FunctionBlockAST : public UnitAST {
  std::unique_ptr<CodeSectionAST> code;

public:
  FunctionBlockAST(const std::string & identifier, Location loc, std::unique_ptr<DeclarationSectionAST> declarations, std::unique_ptr<CodeSectionAST> code)
    : UnitAST(Unit_FunctionBlock, identifier, loc, std::move(declarations)), code(std::move(code)) {}

  /// LLVM style RTTI
  static bool classof(const UnitAST *C) { return C->getKind() == Unit_FunctionBlock; }
};

class DataBlockAST : public UnitAST {
  std::unique_ptr<AssignmentsSectionAST> assignments;

public:
  DataBlockAST(const std::string & identifier, Location loc, std::unique_ptr<DeclarationSectionAST> declarations, std::unique_ptr<AssignmentsSectionAST> assignments)
    : UnitAST(Unit_DataBlock, identifier, loc, std::move(declarations)), assignments(std::move(assignments)) {}

  /// LLVM style RTTI
  static bool classof(const UnitAST *C) { return C->getKind() == Unit_DataBlock; }
};

class UserDefinedTypeAST : public UnitAST {
  std::unique_ptr<DataTypeSpecAST> type;

public:
  UserDefinedTypeAST(const std::string & identifier, Location loc, std::unique_ptr<DataTypeSpecAST> type)
    : UnitAST(Unit_UserDefinedDataType, std::move(identifier), std::move(loc), nullptr), type(std::move(type)) {}

  /// LLVM style RTTI
  static bool classof(const UnitAST *C) { return C->getKind() == Unit_UserDefinedDataType; }
};



// ========================================================= //
// MARK: #if 0 - example code

#if 0
/// A variable type with shape information.
struct VarType {
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}

  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(Location loc, double Val) : ExprAST(Expr_Num, loc), Val(Val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  std::vector<std::unique_ptr<ExprAST>> &getValues() { return values; }
  std::vector<int64_t> &getDims() { return dims; }
  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, const std::string &name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, const std::string &name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_VarDecl; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  llvm::Optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, llvm::Optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

  llvm::Optional<ExprAST *> getExpr() {
    if (expr.hasValue())
      return expr->get();
    return llvm::NoneType();
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  char getOp() { return Op; }
  ExprAST *getLHS() { return LHS.get(); }
  ExprAST *getRHS() { return RHS.get(); }

  BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : ExprAST(Expr_BinOp, loc), Op(Op), LHS(std::move(LHS)),
        RHS(std::move(RHS)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(Location loc, const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : ExprAST(Expr_Call, loc), Callee(Callee), Args(std::move(Args)) {}

  llvm::StringRef getCallee() { return Callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return Args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> Arg)
      : ExprAST(Expr_Print, loc), Arg(std::move(Arg)) {}

  ExprAST *getArg() { return Arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(location), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  const std::string &getName() const { return name; }
  const std::vector<std::unique_ptr<VariableExprAST>> &getArgs() {
    return args;
  }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprASTList> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprASTList> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
  PrototypeAST *getProto() { return Proto.get(); }
  ExprASTList *getBody() { return Body.get(); }
};
#endif // 0


/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<UnitAST>> units;

public:
  ModuleAST(std::vector<std::unique_ptr<UnitAST>> units)
      : units(std::move(units)) {}

  auto begin() -> decltype(units.begin()) { return units.begin(); }
  auto end() -> decltype(units.end()) { return units.end(); }
};

void dump(ModuleAST &);

} // namespace sclang

#endif // SCLANG_AST_H_
