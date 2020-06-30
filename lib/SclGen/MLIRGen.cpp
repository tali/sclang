//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
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
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the SCL language.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclGen/AST.h"
#include "sclang/SclGen/MLIRGen.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>
#include <string>

using namespace mlir::scl;
using namespace sclang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

class VariableSymbol {
  mlir::Type type;
  mlir::Value value;
  bool memref;
  bool direct;

public:
  VariableSymbol() : type(), value(), memref(false), direct(false) {}
  VariableSymbol(mlir::Type type, mlir::Value value, bool memref)
      : type(type), value(value), memref(memref), direct(!memref) {}

  mlir::Type getType() const { return type; }
  mlir::Value getValue() const { return value; }
  bool isMemref() const { return memref; }
  bool isDirect() const { return direct; }
};

/// Implementation of a simple MLIR emission from the SCL AST.
///
/// This will emit operations that are specific to the SCL language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(const ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const auto &unit : moduleAST) {
      auto func = mlirGen(*unit.get());
      if (!func)
        return nullptr;
      theModule.push_back(func);
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, VariableSymbol> symbolTable;

  /// The name of the function or block which is currently being generated.
  llvm::StringRef functionName;
  /// Whether this function returns a value.
  bool functionHasReturnValue;

  /// Helper conversion for a SCL AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Type type,
                              mlir::Value value, bool memref) {
    assert(!var.empty());
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, VariableSymbol(type, value, memref));
    return mlir::success();
  }

  // MARK: C.1 Subunits of SCL Source Files

  mlir::FuncOp mlirGen(const UnitAST &unit) {
    switch (unit.getKind()) {
    case sclang::UnitAST::Unit_Function:
      return mlirGen(cast<FunctionAST>(unit));
    default:
      emitError(loc(unit.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(unit.getKind()) << "'";
      return nullptr;
    }
  }

  /// Append all the types and names of a variable declaration to an array
  void addVariables(const VariableDeclarationSubsectionAST &decls,
                    std::vector<mlir::Type> &types,
                    std::vector<llvm::StringRef> &names) {
    const auto &values = decls.getValues();
    types.reserve(types.size() + types.size());
    names.reserve(names.size() + names.size());
    for (const auto &decl : values) {
      auto type = getType(*decl->getDataType());
      // TODO: initializer
      for (const auto &var : decl->getVars()) {
        // TODO: attributes
        types.push_back(type);
        names.push_back(var->getIdentifier());
      }
    }
  }

  mlir::FuncOp mlirGen(const OrganizationBlockAST &ob) {
    emitError(loc(ob.loc())) << "TBD not implemented";
    return nullptr;
  }

  mlir::FuncOp mlirGen(const FunctionAST &func) {
    auto location = loc(func.loc());
    functionName = func.getIdentifier();
    functionHasReturnValue = false;
    std::string name(func.getIdentifier());

    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<StringRef, VariableSymbol> var_scope(
        symbolTable);

    std::vector<mlir::Type> input_types;
    std::vector<llvm::StringRef> input_names;
    std::vector<mlir::Type> output_types;
    std::vector<llvm::StringRef> output_names;
    std::vector<const VariableDeclarationSubsectionAST *> tempvar;
    // register function result as output variable
    auto retType = getType(*func.getType());
    if (!retType.isa<mlir::NoneType>()) {
      output_types.push_back(retType);
      output_names.push_back(name);
      functionHasReturnValue = true;
    }

    // Parse the declaration subsections
    const auto &declarations = func.getDeclarations()->getDecls();
    for (const auto &decl : declarations) {
      switch (decl->getKind()) {
      case DeclarationSubsectionAST::DeclarationASTKind::Decl_Constant:
        emitError(location) << "TBD constants not implemented";
        return nullptr;
      case DeclarationSubsectionAST::DeclarationASTKind::Decl_JumpLabel:
        emitError(location) << "TBD jump labels not implemented";
        return nullptr;
      case DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable:
        const auto &vardecls =
            llvm::cast<VariableDeclarationSubsectionAST>(*decl);
        switch (vardecls.getKind()) {
        case VariableDeclarationSubsectionAST::VarInput:
          addVariables(vardecls, input_types, input_names);
          break;
        case VariableDeclarationSubsectionAST::VarOutput:
          addVariables(vardecls, output_types, output_names);
          break;
        case VariableDeclarationSubsectionAST::VarInOut:
          addVariables(vardecls, input_types, input_names);
          addVariables(vardecls, output_types, output_names);
          break;
        case VariableDeclarationSubsectionAST::Var:
        case VariableDeclarationSubsectionAST::VarTemp:
          tempvar.push_back(&vardecls);
          break;
        }
      }
    }
    // Create an MLIR function

    auto func_type = builder.getFunctionType(input_types, output_types);
    auto function = mlir::FuncOp::create(location, name, func_type);
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(input_types, input_names, entryBlock.getArguments())) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto value = std::get<2>(name_value);
      if (failed(declare(name, type, value, false)))
        return nullptr;
    }

    // prologue: stack space for temporary variables
    for (const auto subsection : tempvar) {
      for (const auto &decl : subsection->getValues()) {
        auto type = getType(*decl->getDataType());
        auto memRefType = mlir::MemRefType::get({}, type);
        auto init = decl->getInitializer();
        for (const auto &var : decl->getVars()) {
          auto location = loc(var->loc());
          auto varStorage = builder.create<TempVariableOp>(
              location, memRefType,
              builder.getStringAttr(var->getIdentifier()));
          declare(var->getIdentifier(), type, varStorage, true);
          if (init) {
            builder.create<StoreOp>(location, varStorage,
                                    mlirGen(*init.getValue()));
          }
        }
      }
    }

    // Declare all the function outputs in the symbol table.
    for (const auto &name_value : llvm::zip(output_types, output_names)) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto memRefType = mlir::MemRefType::get({}, type);
      auto varStorage = builder.create<TempVariableOp>(
          location, memRefType, builder.getStringAttr(name));
      declare(name, type, varStorage, true);
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*func.getCode()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return if no return statement was emitted.
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      mlirgenReturn(location);
    }

    return function;
  }

  void mlirgenReturn(mlir::Location location) {
    if (functionHasReturnValue) {
      auto returnValue = symbolTable.lookup(functionName).getValue();
      builder.create<ReturnOp>(location, returnValue);
    } else {
      builder.create<ReturnOp>(location, mlir::Value());
    }
  }

  mlir::FuncOp mlirgen(const FunctionBlockAST &fb) {
    emitError(loc(fb.loc())) << "TBD not implemented";
    return nullptr;
  }

  mlir::FuncOp mlirgen(const DataBlockAST &db) {
    emitError(loc(db.loc())) << "TBD not implemented";
    return nullptr;
  }

  mlir::FuncOp mlirgen(const UserDefinedTypeAST &udt) {
    emitError(loc(udt.loc())) << "TBD not implemented";
    return nullptr;
  }

  // MARK: C.2 Structure of Declaration Sections

  // MARK: C.3 Data Types in SCL

  /// Codegen a code section, return failure if one statement hit an error.
  mlir::LogicalResult mlirGen(const CodeSectionAST &code) {
    // ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);

    for (auto &instr : code.getInstructions()) {
      // Generic expression dispatch codegen.
      if (failed(mlirGen(*instr)))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Codegen an instruction.
  mlir::LogicalResult mlirGen(const InstructionAST &instr) {
    auto location = loc(instr.loc());

    switch (instr.getKind()) {
    case InstructionAST::Instr_Assignment:
      return mlirGen(llvm::cast<ValueAssignmentAST>(instr));
    case InstructionAST::Instr_Continue:
      builder.create<ContinueOp>(location);
      break;
    case InstructionAST::Instr_IfThenElse:
      return mlirGen(llvm::cast<IfThenElseAST>(instr));
    case InstructionAST::Instr_Return:
      mlirgenReturn(location);
      break;
    case InstructionAST::Instr_Exit:
      builder.create<ExitOp>(location);
      break;
    default:
      emitError(loc(instr.loc()))
          << "MLIR codegen encountered an unhandled instruction kind '"
          << Twine(instr.getKind()) << "'";
      return mlir::failure();
    }
    return mlir::success();
  }

  /// Codegen a variable assignment
  mlir::LogicalResult mlirGen(const ValueAssignmentAST &instr) {
    auto location = loc(instr.loc());

    const auto &expr = llvm::cast<BinaryExpressionAST>(*instr.getExpression());
    assert(expr.getOp() == tok_assignment);

    mlir::Value lhs = mlirGenLValue(*expr.getLhs());
    if (!lhs)
      return mlir::failure();
    mlir::Value rhs = mlirGen(*expr.getRhs());
    if (!rhs)
      return mlir::failure();

    builder.create<StoreOp>(location, lhs, rhs);

    return mlir::success();
  }

  mlir::Value mlirGenLValue(const ExpressionAST &expr) {
    auto location = loc(expr.loc());

    switch (expr.getKind()) {
    default:
      emitError(location) << "not a lvalue, kind " << (int)expr.getKind();
      return nullptr;
    case ExpressionAST::Expr_SimpleVariable:
      return mlirGenLValue(llvm::cast<SimpleVariableAST>(expr));
    case ExpressionAST::Expr_IndexedVariable:
      emitError(loc(expr.loc()))
          << "IndexedVariable not implemented"; // TODO: TBD
      return nullptr;
    }
  }

  mlir::Value mlirGen(const ExpressionAST &expr) {
    switch (expr.getKind()) {
    case ExpressionAST::Expr_IntegerConstant:
      return mlirGen(llvm::cast<IntegerConstantAST>(expr));
    case ExpressionAST::Expr_RealConstant:
      return mlirGen(llvm::cast<RealConstantAST>(expr));
    case ExpressionAST::Expr_StringConstant:
      return mlirGen(llvm::cast<StringConstantAST>(expr));
    case ExpressionAST::Expr_SimpleVariable:
      return mlirGen(llvm::cast<SimpleVariableAST>(expr));
    case ExpressionAST::Expr_Binary:
      return mlirGen(llvm::cast<BinaryExpressionAST>(expr));
    case ExpressionAST::Expr_Unary:
      return mlirGen(llvm::cast<UnaryExpressionAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "expression kind not implemented"; // TODO: TBD
      return nullptr;
    }
  }

  mlir::Value mlirGen(const IntegerConstantAST &expr) {
    auto location = loc(expr.loc());
    auto type = builder.getIntegerType(16);
    auto value = builder.getIntegerAttr(type, expr.getValue());
    return builder.create<ConstantOp>(location, type, value);
  }

  mlir::Value mlirGen(const RealConstantAST &expr) {
    auto location = loc(expr.loc());
    auto type = builder.getF32Type();
    auto value = builder.getF32FloatAttr(expr.getValue());
    return builder.create<ConstantOp>(location, type, value);
  }

  mlir::Value mlirGen(const StringConstantAST &expr) {
    emitError(loc(expr.loc())) << "StringConstantAST not implemented";
    return nullptr;
  }

  mlir::Value mlirGen(const SimpleVariableAST &expr) {
    auto location = loc(expr.loc());
    auto name = llvm::cast<SimpleVariableAST>(expr).getName();

    auto variable = symbolTable.lookup(name);
    if (variable.isMemref())
      return builder.create<LoadOp>(location, variable.getValue());
    if (variable.isDirect())
      return variable.getValue();
    emitError(location) << "unknown variable '" << name << "'";
    return nullptr;
  }

  mlir::Value mlirGenLValue(const SimpleVariableAST &expr) {
    auto location = loc(expr.loc());
    auto name = llvm::cast<SimpleVariableAST>(expr).getName();
    auto variable = symbolTable.lookup(name);
    if (variable.isMemref())
      return variable.getValue();
    emitError(location) << "not a lvalue, variable " << name;
    return nullptr;
  }

  mlir::Value mlirGen(const IndexedVariableAST &expr) {
    emitError(loc(expr.loc())) << "IndexedVariableAST not implemented";
    return nullptr;
  }

  mlir::Value mlirGen(const BinaryExpressionAST &expr) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*expr.getLhs());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*expr.getRhs());
    if (!rhs)
      return nullptr;
    auto location = loc(expr.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (expr.getOp()) {
    default:
      emitError(location, "invalid binary operator '")
          << (int)expr.getOp() << "'";
      return nullptr;
    case tok_or:
      return builder.create<OrOp>(location, lhs, rhs);
    case tok_xor:
      return builder.create<XOrOp>(location, lhs, rhs);
    case sclang::tok_ampersand:
    case sclang::tok_and:
      return builder.create<AndOp>(location, lhs, rhs);
    case sclang::tok_cmp_eq:
      return builder.create<EqualOp>(location, builder.getI1Type(), lhs, rhs);
    case sclang::tok_cmp_ne:
      return builder.create<NotEqualOp>(location, builder.getI1Type(), lhs,
                                        rhs);
    case sclang::tok_cmp_lt:
      return builder.create<LessThanOp>(location, builder.getI1Type(), lhs,
                                        rhs);
    case sclang::tok_cmp_le:
      return builder.create<LessEqualOp>(location, builder.getI1Type(), lhs,
                                         rhs);
    case sclang::tok_cmp_gt:
      return builder.create<GreaterThanOp>(location, builder.getI1Type(), lhs,
                                           rhs);
    case sclang::tok_cmp_ge:
      return builder.create<GreaterEqualOp>(location, builder.getI1Type(), lhs,
                                            rhs);
    case sclang::tok_minus:
      return builder.create<SubOp>(location, lhs, rhs);
    case sclang::tok_plus:
      return builder.create<AddOp>(location, lhs, rhs);
    case sclang::tok_times:
      return builder.create<MulOp>(location, lhs, rhs);
    case sclang::tok_div:
    case sclang::tok_divide:
      return builder.create<DivOp>(location, lhs, rhs);
    case sclang::tok_mod:
      return builder.create<ModOp>(location, lhs, rhs);
    case sclang::tok_exponent:
      return builder.create<ExpOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '")
        << (int)expr.getOp() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(const UnaryExpressionAST &expr) {
    mlir::Value rhs = mlirGen(*expr.getRhs());
    if (!rhs)
      return nullptr;
    auto location = loc(expr.loc());

    switch (expr.getOp()) {
    case tok_minus:
      return builder.create<UnaryMinusOp>(location, rhs);
    case tok_not:
      return builder.create<UnaryNotOp>(location, rhs);
    case tok_plus:
      return rhs;
    default:
      emitError(location, "invalid binary operator '")
          << (int)expr.getOp() << "'";
      return nullptr;
    }
  }

  mlir::Type getType(const ElementaryDataTypeAST &type) {
    switch (type.getType()) {
    case ElementaryDataTypeAST::Type_Void:
      return builder.getNoneType();
    case ElementaryDataTypeAST::Type_Bool:
      return builder.getI1Type();
    case ElementaryDataTypeAST::Type_Byte:
      return builder.getIntegerType(8);
    case ElementaryDataTypeAST::Type_Word:
      return builder.getIntegerType(16);
    case ElementaryDataTypeAST::Type_DWord:
      return builder.getIntegerType(32);
    case ElementaryDataTypeAST::Type_Char:
      return builder.getIntegerType(8);
    case ElementaryDataTypeAST::Type_Int:
      return builder.getIntegerType(16);
    case ElementaryDataTypeAST::Type_DInt:
      return builder.getIntegerType(32);
    case ElementaryDataTypeAST::Type_Real:
      return builder.getF32Type();
    // TODO: TBD more types
    default:
      emitError(loc(type.loc()))
          << "MLIR codegen encountered an unhandled type '"
          << Twine(type.getType()) << "'";
      return nullptr;
    }
  }

  mlir::LogicalResult getConstantInteger(const ExpressionAST &expr,
                                         int &value) {
    switch (expr.getKind()) {
    default:
      emitError(loc(expr.loc())) << "constant integer expected";
      return mlir::failure();
    case ExpressionAST::Expr_IntegerConstant:
      value = llvm::cast<IntegerConstantAST>(expr).getValue();
      return mlir::success();
    }
  }

  mlir::Type getType(const ArrayDataTypeSpecAST &type) {
    mlir::Type elementType = getType(*type.getDataType());
    std::vector<ArrayType::DimTy> dimensions;
    dimensions.reserve(type.getDimensions().size());
    for (const auto &dim : type.getDimensions()) {
      int min, max;
      if (mlir::failed(getConstantInteger(*dim.get()->getMin(), min)))
        return nullptr;
      if (mlir::failed(getConstantInteger(*dim.get()->getMax(), max)))
        return nullptr;
      dimensions.push_back(std::make_pair(min, max));
    }
    return ArrayType::get(dimensions, elementType);
  }

  mlir::Type getType(const StructDataTypeSpecAST &type) {
    std::vector<mlir::Type> elements;

    auto components = type.getComponents();
    elements.reserve(components.size());

    for (auto &component : components) {
      mlir::Type type = getType(*component.get()->getDataType());
      if (!type)
        return nullptr;
      elements.push_back(type);
    }
    return StructType::get(elements);
  }

  mlir::Type getType(const DataTypeSpecAST &type) {
    switch (type.getKind()) {
    case DataTypeSpecAST::DataType_Elementary:
      return getType(llvm::cast<ElementaryDataTypeAST>(type));
    case DataTypeSpecAST::DataType_Array:
      return getType(llvm::cast<ArrayDataTypeSpecAST>(type));
    case DataTypeSpecAST::DataType_Struct:
      return getType(llvm::cast<StructDataTypeSpecAST>(type));
    default:
      emitError(loc(type.loc()))
          << "MLIR codegen encountered an unhandled type kind '"
          << Twine(type.getKind()) << "'";
      return nullptr;
    }
  }

  // MARK: C.7 Control Statements

  /// Codegen an if-then-else block
  mlir::LogicalResult mlirGen(const IfThenElseAST &ifThenElse) {
    mlir::Value condition;

    auto old = builder.saveInsertionPoint();
    bool first = true;

    for (const auto &ifThen : ifThenElse.getThens()) {
      auto location = loc(ifThen->loc());
      auto condition = mlirGen(*ifThen->getCondition());
      if (!condition)
        return mlir::failure();
      auto cond = builder.create<IfThenElseOp>(location, condition);
      if (!cond)
        return mlir::failure();
      if (first) {
        old = builder.saveInsertionPoint();
        first = false;
      } else {
        builder.create<EndOp>(location);
      }
      builder.createBlock(&cond.thenBody());
      mlirGen(*ifThen->getCodeBlock());
      builder.create<EndOp>(location);
      builder.createBlock(&cond.elseBody());
    }
    if (ifThenElse.getElseBlock()) {
      mlirGen(*ifThenElse.getElseBlock().getValue());
    }
    builder.create<EndOp>(loc(ifThenElse.loc()));

    builder.restoreInsertionPoint(old);

    return mlir::success();
  }
};

} // namespace

namespace sclang {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace sclang