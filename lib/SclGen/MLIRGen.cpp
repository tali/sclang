//===- MLIRGen.cpp - MLIR Generation from SCL AST -------------------------===//
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

#include "sclang/SclGen/MLIRGen.h"
#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclGen/AST.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>
#include <string>

using namespace mlir::scl;
using namespace sclang;

using llvm::ArrayRef;
using llvm::isa;
using llvm::dyn_cast;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;
using llvm::TypeSwitch;

namespace {

class VariableSymbol {
  mlir::Type type;
  mlir::Value value;

public:
  VariableSymbol() : type(), value() {}
  VariableSymbol(mlir::Type type, mlir::Value value)
      : type(type), value(value) {}

  mlir::Type getType() const { return type; }
  mlir::Value getValue() const { return value; }
  bool isMemref() const { return type.isa<mlir::MemRefType>(); }
  bool isDirect() const { return value != nullptr; }
  bool isElement() const { return type && value == nullptr; }
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
  mlir::ModuleOp mlirGen(const ModuleAST *moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const auto &unit : *moduleAST) {
      builder.clearInsertionPoint();
      auto func = mlirGen(unit.get());
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
                              mlir::Value value) {
    assert(!var.empty());
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, VariableSymbol(type, value));
    return mlir::success();
  }

  // MARK: C.1 Subunits of SCL Source Files

  mlir::Operation *mlirGen(const UnitAST *unit) {
    return TypeSwitch<const UnitAST *, mlir::Operation *>(unit)
        .Case<DataBlockAST>([&](auto db) { return mlirGen(db); })
        .Case<FunctionAST>([&](auto function) { return mlirGen(function); })
        .Case<FunctionBlockAST>([&](auto fb) { return mlirGen(fb); })
        .Case<OrganizationBlockAST>([&](auto ob) { return mlirGen(ob); })
        .Case<UserDefinedTypeAST>([&](auto udt) { return mlirGen(udt); })
        .Default([&](auto unit) {
          emitError(loc(unit->loc()))
              << "MLIR codegen encountered an unhandled unit kind '"
              << Twine(unit->getKind()) << "'";
          return nullptr;
        });
  }

  /// Append all the types and names of a variable declaration to an array
  void addVariables(const VariableDeclarationSubsectionAST &decls,
                    std::vector<mlir::Type> &types,
                    std::vector<llvm::StringRef> &names) {
    const auto &values = decls.getValues();
    types.reserve(types.size() + types.size());
    names.reserve(names.size() + names.size());
    for (const auto &decl : values) {
      auto type = getType(decl->getDataType());
      // TODO: initializer
      for (const auto &var : decl->getVars()) {
        // TODO: attributes
        types.push_back(type);
        names.push_back(var->getIdentifier());
      }
    }
  }

  mlir::LogicalResult mlirGen(const VariableDeclarationSubsectionAST *decls,
                              bool isInput, bool isOutput) {
    for (const auto &decl : decls->getValues()) {
      auto type = getType(decl->getDataType());
      // TODO: initializer
      for (const auto &var : decl->getVars()) {
        auto location = loc(var->loc());
        auto name = var->getIdentifier();
        declare(name, type, nullptr);
        builder.create<VariableOp>(location, type, isInput, isOutput, name);
      }
    }
    return mlir::success();
  }

  mlir::FuncOp mlirGen(const OrganizationBlockAST *ob) {
    emitError(loc(ob->loc())) << "TBD not implemented";
    return nullptr;
  }

  FunctionOp mlirGen(const FunctionAST *func) {
    auto location = loc(func->loc());
    functionName = func->getIdentifier();
    functionHasReturnValue = false;
    std::string name(func->getIdentifier());

    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<StringRef, VariableSymbol> var_scope(
        symbolTable);

    std::vector<mlir::Type> input_types;
    std::vector<llvm::StringRef> input_names;
    std::vector<mlir::Type> output_types;
    std::vector<llvm::StringRef> output_names;
    std::vector<const VariableDeclarationSubsectionAST *> tempvar;
    // register function result as output variable
    auto retType = getType(func->getType());
    if (!retType.isa<mlir::NoneType>()) {
      output_types.push_back(retType);
      output_names.push_back(name);
      functionHasReturnValue = true;
    }

    // Parse the declaration subsections
    const auto &declarations = func->getDeclarations()->getDecls();
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
    auto function = builder.create<FunctionOp>(location, name, func_type);
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
    auto nameId = mlir::Identifier::get("scl.name", builder.getContext());
    int argIndex = 0;
    for (const auto name_value :
         llvm::zip(input_types, input_names, entryBlock.getArguments())) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto value = std::get<2>(name_value);
      auto memRefType = AddressType::get(type);
      auto varStorage = builder.create<TempVariableOp>(
          location, memRefType, builder.getStringAttr(name));
      if (failed(declare(name, type, varStorage)))
        return nullptr;
      builder.create<StoreOp>(location, varStorage, value);
      auto nameAttr = builder.getStringAttr(name);
      function.setArgAttr(argIndex, nameId, nameAttr);
      argIndex++;
    }

    // prologue: stack space for temporary variables
    for (const auto subsection : tempvar) {
      for (const auto &decl : subsection->getValues()) {
        auto type = getType(decl->getDataType());
        auto memRefType = AddressType::get(type);
        auto init = decl->getInitializer();
        for (const auto &var : decl->getVars()) {
          auto location = loc(var->loc());
          auto varStorage = builder.create<TempVariableOp>(
              location, memRefType,
              builder.getStringAttr(var->getIdentifier()));
          declare(var->getIdentifier(), type, varStorage);
          if (init) {
            builder.create<StoreOp>(location, varStorage,
                                    mlirGen(init.getValue()));
          }
        }
      }
    }

    // Declare all the function outputs in the symbol table.
    for (const auto name_value : llvm::zip(output_types, output_names)) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto memRefType = AddressType::get(type);
      auto varStorage = builder.create<TempVariableOp>(
          location, memRefType, builder.getStringAttr(name));
      declare(name, type, varStorage);
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(func->getCode()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return if no return statement was emitted.
    if (entryBlock.empty() || (!isa<ReturnOp>(entryBlock.back()) &&
                               !isa<ReturnValueOp>(entryBlock.back()))) {
      mlirgenReturn(location);
    }

    return function;
  }

  void mlirgenReturn(mlir::Location location) {
    if (functionHasReturnValue) {
      auto returnValue = symbolTable.lookup(functionName).getValue();
      builder.create<ReturnValueOp>(location, returnValue);
    } else {
      builder.create<ReturnOp>(location);
    }
  }

  FunctionBlockOp mlirGen(const FunctionBlockAST *fb) {
    auto location = loc(fb->loc());
    functionName = fb->getIdentifier();
    functionHasReturnValue = false;
    std::string name(fb->getIdentifier());

    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<StringRef, VariableSymbol> var_scope(
        symbolTable);

    std::vector<const VariableDeclarationSubsectionAST *> tempvar;

    // Create an MLIR function
    auto function = builder.create<FunctionBlockOp>(location, name);
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

    // declare argument to instance db
    auto idb = InstanceDbType::get(builder.getContext(), name);
    declare("$self", idb, entryBlock.getArgument(0));


    // Parse the declaration subsections
    const auto &declarations = fb->getDeclarations()->getDecls();
    for (const auto &decl : declarations) {
      TypeSwitch<DeclarationSubsectionAST*>(decl.get())
      .Case<VariableDeclarationSubsectionAST>([&](auto decl) {
        switch (decl->getKind()) {
        case VariableDeclarationSubsectionAST::VarInput:
          mlirGen(decl, /*isInput=*/true, /*isOutput=*/false);
          break;
        case VariableDeclarationSubsectionAST::VarOutput:
          mlirGen(decl, /*isInput=*/false, /*isOutput=*/true);
          break;
        case VariableDeclarationSubsectionAST::VarInOut:
          mlirGen(decl, /*isInput=*/true, /*isOutput=*/true);
          break;
        case VariableDeclarationSubsectionAST::Var:
          mlirGen(decl, /*isInput=*/false, /*isOutput=*/false);
          break;
        case VariableDeclarationSubsectionAST::VarTemp:
          tempvar.push_back(decl);
          break;
        }
      });
    }

    // prologue: stack space for temporary variables
    for (const auto subsection : tempvar) {
      for (const auto &decl : subsection->getValues()) {
        auto type = getType(decl->getDataType());
        auto memRefType = AddressType::get(type);
        auto init = decl->getInitializer();
        for (const auto &var : decl->getVars()) {
          auto location = loc(var->loc());
          auto varStorage = builder.create<TempVariableOp>(
              location, memRefType,
              builder.getStringAttr(var->getIdentifier()));
          declare(var->getIdentifier(), type, varStorage);
          if (init) {
            builder.create<StoreOp>(location, varStorage,
                                    mlirGen(init.getValue()));
          }
        }
      }
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(fb->getCode()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return if no return statement was emitted.
    if (entryBlock.empty() || !isa<ReturnOp>(entryBlock.back())) {
      mlirgenReturn(location);
    }

    return function;
  }

  DataBlockOp mlirGen(const DataBlockAST *db) {
    auto location = loc(db->loc());
    StringRef name = db->getIdentifier();
    mlir::Type type = getType(db->getType());
    if (!type) return nullptr;
    mlir::TypeAttr typeAttr = mlir::TypeAttr::get(type);
    return builder.create<DataBlockOp>(location, typeAttr, name);
  }

  mlir::FuncOp mlirGen(const UserDefinedTypeAST *udt) {
    emitError(loc(udt->loc())) << "TBD not implemented";
    return nullptr;
  }

  // MARK: C.2 Structure of Declaration Sections

  // MARK: C.3 Data Types in SCL

  /// Codegen a code section, return failure if one statement hit an error.
  mlir::LogicalResult mlirGen(const CodeSectionAST *code) {
    // ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);

    for (auto &instr : code->getInstructions()) {
      // Generic expression dispatch codegen.
      if (failed(mlirGen(instr.get())))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Codegen an instruction.
  mlir::LogicalResult mlirGen(const InstructionAST *instr) {
    auto location = loc(instr->loc());

    return TypeSwitch<const InstructionAST *, mlir::LogicalResult>(instr)
        .Case<ValueAssignmentAST>([&](auto assign) { return mlirGen(assign); })
        .Case<SubroutineProcessingAST>([&](auto call) { return mlirGen(call); })
        .Case<ContinueAST>([&](auto instr) {
          builder.create<ContinueOp>(location);
          return mlir::success();
        })
        .Case<IfThenElseAST>(
            [&](auto ifThenElse) { return mlirGen(ifThenElse); })
        .Case<ReturnAST>([&](auto instr) {
          mlirgenReturn(location);
          return mlir::success();
        })
        .Case<ExitAST>([&](auto instr) {
          builder.create<ExitOp>(location);
          return mlir::success();
        })
        .Default([&](auto instr) {
          emitError(loc(instr->loc()))
              << "MLIR codegen encountered an unhandled instruction kind '"
              << Twine(instr->getKind()) << "'";
          return mlir::failure();
        });
  }

  /// Codegen a variable assignment
  mlir::LogicalResult mlirGen(const ValueAssignmentAST *instr) {
    auto location = loc(instr->loc());

    const auto expr = llvm::cast<BinaryExpressionAST>(instr->getExpression());
    assert(expr->getOp() == tok_assignment);

    mlir::Value lhs = mlirGenLValue(expr->getLhs());
    if (!lhs)
      return mlir::failure();
    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return mlir::failure();

    builder.create<StoreOp>(location, lhs, rhs);

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const SubroutineProcessingAST *call) {
    mlir::Value result = mlirGen(call->getCall());
    if (!result)
      return mlir::failure();
    return mlir::success();
  }

  mlir::Value mlirGenLValue(const ExpressionAST *expr) {
    auto location = loc(expr->loc());

    return TypeSwitch<const ExpressionAST *, mlir::Value>(expr)
        .Case<SimpleVariableAST>([&](auto expr) { return mlirGenLValue(expr); })
        .Case<IndexedVariableAST>([&](auto expr) {
          emitError(loc(expr->loc()))
              << "IndexedVariable not implemented"; // TODO: TBD
          return nullptr;
        })
        .Default([&](auto expr) {
          emitError(location) << "not a lvalue, kind " << (int)expr->getKind();
          return nullptr;
        });
  }

  mlir::Value mlirGen(const ExpressionAST *expr) {
    return TypeSwitch<const ExpressionAST *, mlir::Value>(expr)
        .Case<IntegerConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<RealConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<StringConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<SimpleVariableAST>([&](auto expr) { return mlirGen(expr); })
        .Case<BinaryExpressionAST>([&](auto expr) { return mlirGen(expr); })
        .Case<UnaryExpressionAST>([&](auto expr) { return mlirGen(expr); })
        .Case<FunctionCallAST>([&](auto expr) { return mlirGen(expr); })
        .Default([&](auto expr) {
          emitError(loc(expr->loc())) << "expression kind not implemented";
          return nullptr; // TODO: TBD
        });
  }

  mlir::Value mlirGenRValue(const ExpressionAST *expr) {
    return mlirGenRValue(mlirGen(expr));
  }
  mlir::Value mlirGenRValue(mlir::Value expr) {
    if (!expr)
      return nullptr;
    if (expr.getType().isa<AddressType>()) {
      return builder.create<LoadOp>(expr.getLoc(), expr);
    }
    return expr;
  }


  mlir::Value mlirGen(const IntegerConstantAST *expr) {
    auto location = loc(expr->loc());
    auto type = getType(tok_int); // TBD use expr.getType()
    auto attrType = builder.getIntegerType(16);
    auto value = builder.getIntegerAttr(attrType, expr->getValue());
    return builder.create<ConstantOp>(location, type, value);
  }

  mlir::Value mlirGen(const RealConstantAST *expr) {
    auto location = loc(expr->loc());
    auto type = getType(tok_real);
    auto value = builder.getF32FloatAttr(expr->getValue());
    return builder.create<ConstantOp>(location, type, value);
  }

  mlir::Value mlirGen(const StringConstantAST *expr) {
    emitError(loc(expr->loc())) << "StringConstantAST not implemented";
    return nullptr;
  }

  mlir::Value mlirGen(const SimpleVariableAST *expr) {
    auto location = loc(expr->loc());
    auto name = llvm::cast<SimpleVariableAST>(expr)->getName();

    auto variable = symbolTable.lookup(name);
    if (variable.isElement()) {
      auto type = variable.getType();
      auto self = symbolTable.lookup("$self").getValue();
      assert(self);
      return builder.create<GetElementOp>(location, type, self, name);
    }
    if (variable.isMemref())
      return variable.getValue();
    if (variable.isDirect())
      return variable.getValue();
    emitError(location) << "unknown variable '" << name << "'";
    return nullptr;
  }

  mlir::Value mlirGenLValue(const SimpleVariableAST *expr) {
    auto location = loc(expr->loc());
    auto name = llvm::cast<SimpleVariableAST>(expr)->getName();
    auto variable = symbolTable.lookup(name);
    if (variable.isElement()) {
      auto type = variable.getType();
      auto self = symbolTable.lookup("$self").getValue();
      assert(self);
      return builder.create<GetElementOp>(location, type, self, name);
    }
    if (variable.isMemref())
      return variable.getValue();
    emitError(location) << "not a lvalue, variable " << name;
    return nullptr;
  }

  mlir::Value mlirGen(const IndexedVariableAST *expr) {
    emitError(loc(expr->loc())) << "IndexedVariableAST not implemented";
    return nullptr;
  }

  mlir::Value mlirGen(const BinaryExpressionAST *expr) {
    auto location = loc(expr->loc());
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
    mlir::Value lhs = mlirGen(expr->getLhs());
    if (!lhs)
      return nullptr;
    if (expr->getOp() == tok_dot) {
      // this BinaryExpression is a STRUCT element access:
      // the right hand side is special, it's not a normal expression but a name
      if (!isa<SimpleVariableAST>(expr->getRhs())) {
        emitError(loc(expr->getRhs()->loc()), "expected element name");
        return nullptr;
      }
      auto name = llvm::cast<SimpleVariableAST>(expr->getRhs())->getName();
      return builder.create<GetElementOp>(location, lhs, name);
    }
    lhs = mlirGenRValue(lhs);
    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return nullptr;

    // Derive the operation name from the binary operator.
    switch (expr->getOp()) {
    default:
      emitError(location, "invalid binary operator '")
          << (int)expr->getOp() << "'";
      return nullptr;
    case tok_or:
      return builder.create<OrOp>(location, lhs, rhs);
    case tok_xor:
      return builder.create<XOrOp>(location, lhs, rhs);
    case tok_ampersand:
    case tok_and:
      return builder.create<AndOp>(location, lhs, rhs);
    case tok_cmp_eq:
      return builder.create<EqualOp>(location, lhs, rhs);
    case tok_cmp_ne:
      return builder.create<NotEqualOp>(location, lhs, rhs);
    case tok_cmp_lt:
      return builder.create<LessThanOp>(location, lhs, rhs);
    case tok_cmp_le:
      return builder.create<LessEqualOp>(location, lhs, rhs);
    case tok_cmp_gt:
      return builder.create<GreaterThanOp>(location, lhs, rhs);
    case tok_cmp_ge:
      return builder.create<GreaterEqualOp>(location, lhs, rhs);
    case tok_minus:
      return builder.create<SubOp>(location, lhs, rhs);
    case tok_plus:
      return builder.create<AddOp>(location, lhs, rhs);
    case tok_times:
      return builder.create<MulOp>(location, lhs, rhs);
    case tok_div:
    case tok_divide:
      return builder.create<DivOp>(location, lhs, rhs);
    case tok_mod:
      return builder.create<ModOp>(location, lhs, rhs);
    case tok_exponent:
      return builder.create<ExpOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '")
        << (int)expr->getOp() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(const UnaryExpressionAST *expr) {
    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return nullptr;
    auto location = loc(expr->loc());

    switch (expr->getOp()) {
    case tok_minus:
      return builder.create<UnaryMinusOp>(location, rhs);
    case tok_not:
      return builder.create<UnaryNotOp>(location, rhs);
    case tok_plus:
      return rhs;
    default:
      emitError(location, "invalid binary operator '")
          << (int)expr->getOp() << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGen(const FunctionCallAST *expr) {
    auto location = loc(expr->loc());
    mlir::Value call = nullptr;

    TypeSwitch<const ExpressionAST *>(expr->getFunction())
    .Case<SimpleVariableAST>([&](auto callee) {
      call = mlirGenFcCall(expr, callee->getName());
    })
    .Case<BinaryExpressionAST>([&](auto callee) {
      if (callee->getOp() != tok_dot) return;
      auto idbVar = dyn_cast<SimpleVariableAST>(callee->getLhs());
      if (!idbVar) {
        emitError(location, "invalid instance db in function call");
      }
      auto fbVar = dyn_cast<SimpleVariableAST>(callee->getRhs());
      if (!fbVar) {
        emitError(location, "invalid fb in function call");
      }
      mlirGenFbCall(expr, idbVar->getName(), fbVar->getName());
    })
    .Default([&](auto callee) {
      emitError(location, "invalid callee in function call");
    });
    return call;
  }

  mlir::Value mlirGenFcCall(const FunctionCallAST *expr, StringRef callee) {
    auto location = loc(expr->loc());
    auto context = builder.getContext();

    // get arguments
    SmallVector<mlir::Value, 4> arguments;
    SmallVector<mlir::Attribute, 4> argNames;
    for (auto &arg : expr->getParameters()) {
      auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
      if (binary && binary->getOp() == tok_assignment) {
        auto lhs = binary->getLhs();
        auto name = dyn_cast<SimpleVariableAST>(lhs);
        if (!name || name->isSymbol()) {
          emitError(loc(lhs->loc()), "invalid parameter name");
          return nullptr;
        }
        auto nameAttr = mlir::StringAttr::get(context, name->getName());
        auto value = mlirGenRValue(binary->getRhs());
        argNames.push_back(nameAttr);
        arguments.push_back(value);
      } else {
        if (expr->getParameters().size() != 1) {
          emitError(loc(arg->loc()), "parameter without name");
          return nullptr;
        }
        arguments.push_back(mlirGenRValue(arg.get()));
      }
    }

    mlir::Type resultType = getType(tok_int); // TODO: TBD

    auto argNamesAttr = mlir::ArrayAttr::get(context, argNames);
    return builder.create<CallFcOp>(location, resultType, callee, arguments, argNamesAttr);
  }

  mlir::LogicalResult mlirGenFbCall(const FunctionCallAST *expr, StringRef idb, StringRef fb) {
    auto location = loc(expr->loc());

    auto idbType = InstanceDbType::get(builder.getContext(), fb);
    auto idbRef = builder.create<GetGlobalOp>(location, idbType, idb);

    // get arguments
    for (auto &arg : expr->getParameters()) {
      auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
      if (binary && binary->getOp() == tok_assignment) {
        auto lhs = binary->getLhs();
        auto nameVar = dyn_cast<SimpleVariableAST>(lhs);
        if (!nameVar || nameVar->isSymbol()) {
          emitError(loc(lhs->loc()), "invalid parameter name");
          return mlir::failure();
        }
        auto name = nameVar->getName();
        auto value = mlirGenRValue(binary->getRhs());

        auto elem = builder.create<GetElementOp>(location, idbRef, name);
        builder.create<StoreOp>(location, elem, value);
      } else {
        emitError(loc(arg->loc()), "parameter without name");
        return mlir::failure();
      }
    }

    builder.create<CallFbOp>(location, fb, idbRef);
    return mlir::success();
  }

  mlir::Type getType(Token token) {
    switch (token) {
    default:
      assert(false);
    case tok_bool:
      return LogicalType::get(builder.getContext(), 1);
    case tok_byte:
      return LogicalType::get(builder.getContext(), 8);
    case tok_word:
      return LogicalType::get(builder.getContext(), 16);
    case tok_dword:
      return LogicalType::get(builder.getContext(), 32);
    case tok_char:
      return IntegerType::get(builder.getContext(), 8);
    case tok_int:
      return IntegerType::get(builder.getContext(), 16);
    case tok_dint:
      return IntegerType::get(builder.getContext(), 32);
    case tok_real:
      return RealType::get(builder.getContext());
    }
  }
  mlir::Type getType(const ElementaryDataTypeAST *type) {
    switch (type->getType()) {
    case ElementaryDataTypeAST::Type_Void:
      return builder.getNoneType();
    case ElementaryDataTypeAST::Type_Bool:
      return getType(tok_bool);
    case ElementaryDataTypeAST::Type_Byte:
      return getType(tok_byte);
    case ElementaryDataTypeAST::Type_Word:
      return getType(tok_word);
    case ElementaryDataTypeAST::Type_DWord:
      return getType(tok_dword);
    case ElementaryDataTypeAST::Type_Char:
      return getType(tok_char);
    case ElementaryDataTypeAST::Type_Int:
      return getType(tok_int);
    case ElementaryDataTypeAST::Type_DInt:
      return getType(tok_dint);
    case ElementaryDataTypeAST::Type_Real:
      return getType(tok_real);
    // TODO: TBD more types
    default:
      emitError(loc(type->loc()))
          << "MLIR codegen encountered an unhandled type '"
          << Twine(type->getType()) << "'";
      return nullptr;
    }
  }

  mlir::LogicalResult getConstantInteger(const ExpressionAST *expr,
                                         int &value) {
    return TypeSwitch<const ExpressionAST *, mlir::LogicalResult>(expr)
        .Case<IntegerConstantAST>([&](auto expr) {
          value = expr->getValue();
          return mlir::success();
        })
        .Default([&](auto expr) {
          emitError(loc(expr->loc())) << "constant integer expected";
          return mlir::failure();
        });
  }

  mlir::Type getType(const ArrayDataTypeSpecAST *type) {
    mlir::Type elementType = getType(type->getDataType());
    std::vector<ArrayType::DimTy> dimensions;
    dimensions.reserve(type->getDimensions().size());
    for (const auto &dim : type->getDimensions()) {
      int min, max;
      if (mlir::failed(getConstantInteger(dim.get()->getMin(), min)))
        return nullptr;
      if (mlir::failed(getConstantInteger(dim.get()->getMax(), max)))
        return nullptr;
      dimensions.push_back(std::make_pair(min, max));
    }
    return ArrayType::get(dimensions, elementType);
  }

  mlir::Type getType(const StructDataTypeSpecAST *type) {
    llvm::SmallVector<mlir::Identifier, 1> names;
    llvm::SmallVector<mlir::Type, 1> types;

    auto context = builder.getContext();
    auto components = type->getComponents();
    names.reserve(components.size());
    types.reserve(components.size());

    for (auto &component : components) {
      auto name = mlir::Identifier::get(component.get()->getName(), context);
      names.push_back(name);

      mlir::Type type = getType(component.get()->getDataType());
      if (!type)
        return nullptr;
      types.push_back(type);
    }
    return StructType::get(names, types);
  }

  mlir::Type getType(const UserDefinedTypeIdentifierAST *type) {
    // TODO: TBD other types
    return InstanceDbType::get(builder.getContext(), type->getName());
  }

  mlir::Type getType(const DataTypeSpecAST *type) {
    return TypeSwitch<const DataTypeSpecAST *, mlir::Type>(type)
        .Case<ElementaryDataTypeAST>([&](auto type) { return getType(type); })
        .Case<ArrayDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<StructDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<UserDefinedTypeIdentifierAST>([&](auto type) { return getType(type); })
        .Default([&](auto type) {
          emitError(loc(type->loc()))
              << "MLIR codegen encountered an unhandled type kind '"
              << Twine(type->getKind()) << "'";
          return nullptr;
        });
  }

  // MARK: C.7 Control Statements

  /// Codegen an if-then-else block
  mlir::LogicalResult mlirGen(const IfThenElseAST *ifThenElse) {
    mlir::Value condition;

    auto old = builder.saveInsertionPoint();
    bool first = true;

    for (const auto &ifThen : ifThenElse->getThens()) {
      auto location = loc(ifThen->loc());
      auto condition = mlirGenRValue(ifThen->getCondition());
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
      mlirGen(ifThen->getCodeBlock());
      builder.create<EndOp>(location);
      builder.createBlock(&cond.elseBody());
    }
    if (ifThenElse->getElseBlock()) {
      mlirGen(ifThenElse->getElseBlock().getValue());
    }
    builder.create<EndOp>(loc(ifThenElse->loc()));

    builder.restoreInsertionPoint(old);

    return mlir::success();
  }
};

} // namespace

namespace sclang {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(&moduleAST);
}

} // namespace sclang
