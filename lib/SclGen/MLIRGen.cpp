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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <functional>
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
  std::optional<mlir::Location> loc;
  mlir::Type type;
  mlir::Value value;
  std::optional<const ExpressionAST *> constValue;

public:
  VariableSymbol() : loc(), type(), value() {}
  VariableSymbol(mlir::Location loc, mlir::Type type, mlir::Value value)
      : loc(loc), type(type), value(value), constValue() {}
  VariableSymbol(mlir::Location loc, mlir::Type type, mlir::Value value,
                 const ExpressionAST *constValue)
      : loc(loc), type(type), value(value), constValue(constValue) {}

  mlir::Location getLocation() const { return loc.value(); }
  mlir::Type getType() const { return type; }
  mlir::Value getValue() const { return value; }
  const ExpressionAST* getConstValue() const { return constValue.value(); }
  bool isAddress() const { return type && type.isa<AddressType>(); }
  bool isDirect() const { return value != nullptr; }
  bool isElement() const { return type && value == nullptr; }
  bool isConst() const { return constValue.has_value(); }
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

    // Pre-pass: register user-defined types and FB names first, so that type
    // references can be resolved regardless of definition order.
    for (const auto &unit : *moduleAST) {
      if (auto *udt = llvm::dyn_cast<UserDefinedTypeAST>(unit.get())) {
        if (failed(mlirGen(udt)))
          return nullptr;
      } else if (auto *fb = llvm::dyn_cast<FunctionBlockAST>(unit.get())) {
        fbNames[fb->getIdentifier()] = true;
      }
    }
    // Second pre-pass: collect function result types and the declared types
    // of all variables per callable, so that call arguments (incl. string
    // literals) can be typed even before the callee is generated.
    for (const auto &unit : *moduleAST) {
      if (auto *fn = llvm::dyn_cast<FunctionAST>(unit.get())) {
        functionResultTypes[fn->getIdentifier()] = getType(fn->getType());
        collectVarTypes(fn->getDeclarations(),
                        callableVarTypes[fn->getIdentifier()]);
      } else if (auto *ob = llvm::dyn_cast<OrganizationBlockAST>(unit.get())) {
        functionResultTypes[ob->getIdentifier()] = builder.getNoneType();
        collectVarTypes(ob->getDeclarations(),
                        callableVarTypes[ob->getIdentifier()]);
      } else if (auto *fb = llvm::dyn_cast<FunctionBlockAST>(unit.get())) {
        collectVarTypes(fb->getDeclarations(),
                        callableVarTypes[fb->getIdentifier()]);
      }
    }

    for (const auto &unit : *moduleAST) {
      builder.clearInsertionPoint();
      if (auto *udt = llvm::dyn_cast<UserDefinedTypeAST>(unit.get())) {
        if (failed(mlirGen(udt)))
          return nullptr;
        continue;
      }
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

  /// Jump label name -> block, for the function currently being generated.
  llvm::StringMap<mlir::Block *> labelBlocks;
  /// User-defined type name -> struct type.
  llvm::StringMap<mlir::Type> udtTypes;
  /// Byte-view address value -> declared SCL type of the member it points
  /// into (FB/struct members are addressed as i8 byte views; the declared
  /// type is needed to type loads and to resolve nested member access).
  llvm::DenseMap<mlir::Value, mlir::Type> byteViewTypes;
  /// Function/OB name -> result type, collected in a pre-pass over the
  /// module so calls can be typed even before the callee is generated.
  llvm::StringMap<mlir::Type> functionResultTypes;
  /// Callable name (function/OB/FB) -> variable name -> declared type,
  /// collected in a pre-pass so call arguments can be typed (and string
  /// literals materialized) even before the callee is generated.
  llvm::StringMap<llvm::StringMap<mlir::Type>> callableVarTypes;

  /// Collect the declared types of all variables of a declaration section
  /// into `map`, keyed by variable name.
  void collectVarTypes(const DeclarationSectionAST *declarations,
                       llvm::StringMap<mlir::Type> &map) {
    if (!declarations)
      return;
    for (const auto &decl : declarations->getDecls()) {
      if (decl->getKind() !=
          DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable)
        continue;
      const auto &vardecls =
          llvm::cast<VariableDeclarationSubsectionAST>(*decl);
      for (const auto &value : vardecls.getValues()) {
        auto type = getType(value->getDataType());
        if (!type)
          continue;
        for (const auto &var : value->getVars())
          map[var->getIdentifier()] = type;
      }
    }
  }
  /// Names of all function blocks in the module (for type resolution).
  llvm::StringMap<bool> fbNames;

  /// Does the current insertion block already end with a terminator?
  bool blockHasTerminator() {
    auto block = builder.getInsertionBlock();
    return block && !block->empty() &&
           block->back().hasTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Create the EndOp terminator for the current block, unless it already
  /// has a terminator (e.g. from a goto).
  void createEndOp(mlir::Location location) {
    if (!blockHasTerminator())
      builder.create<EndOp>(location);
  }

  /// Helper conversion for a SCL AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(mlir::Location loc, llvm::StringRef var,
                              mlir::Type type, mlir::Value value) {
    assert(!var.empty());
    if (symbolTable.count(var)) {
      emitError(loc) << "variable already declared: " << var;
      return mlir::failure();
    }
    symbolTable.insert(var, VariableSymbol(loc, type, value));
    return mlir::success();
  }

  mlir::LogicalResult declare(mlir::Location loc, llvm::StringRef name,
                              const ExpressionAST *constValue) {
    mlir::Value value = mlirGen(constValue);
    mlir::Type type = value.getType();

    assert(!name.empty());
    if (symbolTable.count(name)) {
      emitError(loc) << "symbol already declared: " << name;
      return mlir::failure();
    }
    symbolTable.insert(name, VariableSymbol(loc, type, value, constValue));
    return mlir::success();
  }

  // MARK: C.1 Subunits of SCL Source Files

  mlir::Operation *mlirGen(const UnitAST *unit) {
    return TypeSwitch<const UnitAST *, mlir::Operation *>(unit)
        .Case<DataBlockAST>([&](auto db) { return mlirGen(db); })
        .Case<FunctionAST>([&](auto function) { return mlirGen(function); })
        .Case<FunctionBlockAST>([&](auto fb) { return mlirGen(fb); })
        .Case<OrganizationBlockAST>([&](auto ob) { return mlirGen(ob); })
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
    types.reserve(types.size() + values.size());
    names.reserve(names.size() + values.size());
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
      mlir::Value init = nullptr;
      if (decl->getInitializer().has_value()) {
        init = mlirGen(decl->getInitializer().value());
        if (!init)
          return mlir::failure();
      }
      for (const auto &var : decl->getVars()) {
        auto location = loc(var->loc());
        auto name = var->getIdentifier();
        if (failed(declare(location, name, type, nullptr)))
          return mlir::failure();
        builder.create<VariableOp>(location, type, isInput, isOutput, name, init);
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGenTempVar(const VariableDeclarationSubsectionAST *decls) {
    for (const auto &decl : decls->getValues()) {
      auto type = getType(decl->getDataType());
      auto addressType = AddressType::get(type);
      auto init = decl->getInitializer();
      for (const auto &var : decl->getVars()) {
        auto location = loc(var->loc());
        StringRef name = var->getIdentifier();
        mlir::StringAttr nameAttr = builder.getStringAttr(name);
        auto varStorage = builder.create<TempVariableOp>(location, addressType,
                                                         nameAttr);
        if (!varStorage)
          return mlir::failure();
        if (failed(declare(location, name, addressType, varStorage)))
          return mlir::failure();
        if (init) {
          auto value = mlirGen(init.value());
          builder.create<StoreOp>(location, varStorage, value);
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const ConstantDeclarationSubsectionAST *decls) {
    for (const auto &decl : decls->getValues()) {
      auto location = loc(decl->loc());

      StringRef name = decl->getName();
      const ExpressionAST *constValue = decl->getValue();

      if (failed(declare(location, name, constValue)))
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const JumpLabelDeclarationSubsectionAST *decls) {
    // Labels are defined by label statements in the code section; the
    // declaration only reserves the names, so there is nothing to emit.
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const DeclarationSubsectionAST *decl) {
    return TypeSwitch<const DeclarationSubsectionAST*, mlir::LogicalResult>(decl)
    .Case<VariableDeclarationSubsectionAST>([&](auto decl) {
      switch (decl->getKind()) {
      case VariableDeclarationSubsectionAST::VarInput:
        return mlirGen(decl, /*isInput=*/true, /*isOutput=*/false);
      case VariableDeclarationSubsectionAST::VarOutput:
        return mlirGen(decl, /*isInput=*/false, /*isOutput=*/true);
      case VariableDeclarationSubsectionAST::VarInOut:
        return mlirGen(decl, /*isInput=*/true, /*isOutput=*/true);
      case VariableDeclarationSubsectionAST::Var:
        return mlirGen(decl, /*isInput=*/false, /*isOutput=*/false);
      case VariableDeclarationSubsectionAST::VarTemp:
        return mlirGenTempVar(decl);
      }
      assert(false);
    })
    .Case<ConstantDeclarationSubsectionAST>([&](auto decl) {
      return mlirGen(decl);
    })
    .Case<JumpLabelDeclarationSubsectionAST>([&](auto decl) {
      return mlirGen(decl);
    });
  }

  mlir::Operation *mlirGen(const OrganizationBlockAST *ob) {
    // An organization block is an entry point without parameters or return
    // value: model it as a void function. Mark it so the lowering can emit
    // the scan-cycle hooks (scl_rt_update_inputs/outputs, ADR 002).
    auto fn = mlirGenFunctionLike(ob->getIdentifier(), builder.getNoneType(),
                                  ob->getDeclarations(), ob->getCode(),
                                  loc(ob->loc()));
    if (fn)
      fn->setAttr("scl.organization_block", builder.getUnitAttr());
    return fn;
  }

  FunctionOp mlirGen(const FunctionAST *func) {
    return mlirGenFunctionLike(func->getIdentifier(),
                               getType(func->getType()),
                               func->getDeclarations(), func->getCode(),
                               loc(func->loc()));
  }

  /// Shared codegen for functions and organization blocks.
  FunctionOp mlirGenFunctionLike(llvm::StringRef identifier, mlir::Type retType,
                                 const DeclarationSectionAST *declarations,
                                 const CodeSectionAST *code,
                                 mlir::Location location) {
    functionName = identifier;
    functionHasReturnValue = !retType.isa<mlir::NoneType>();
    labelBlocks.clear();
    std::string name(identifier);

    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<StringRef, VariableSymbol> var_scope(
        symbolTable);

    std::vector<mlir::Type> input_types;
    std::vector<llvm::StringRef> input_names;
    std::vector<mlir::Type> output_types;
    std::vector<llvm::StringRef> output_names;
    std::vector<const VariableDeclarationSubsectionAST *> tempvar;
    // register function result as output variable
    if (functionHasReturnValue) {
      output_types.push_back(retType);
      output_names.push_back(name);
    }

    // Parse the declaration subsections
    for (const auto &decl : declarations->getDecls()) {
      switch (decl->getKind()) {
      case DeclarationSubsectionAST::DeclarationASTKind::Decl_Constant:
      case DeclarationSubsectionAST::DeclarationASTKind::Decl_JumpLabel:
        if (failed(mlirGen(decl.get())))
          return nullptr;
        break;
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
    auto nameId = mlir::StringAttr::get(builder.getContext(), "scl.name");
    int argIndex = 0;
    for (const auto name_value :
         llvm::zip(input_types, input_names, entryBlock.getArguments())) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto value = std::get<2>(name_value);
      auto addressType = AddressType::get(type);
      auto varStorage = builder.create<TempVariableOp>(
          location, addressType, builder.getStringAttr(name));
      if (failed(declare(location, name, addressType, varStorage)))
        return nullptr;
      builder.create<StoreOp>(location, varStorage, value);
      auto nameAttr = builder.getStringAttr(name);
      function.setArgAttr(argIndex, nameId, nameAttr);
      argIndex++;
    }

    // prologue: stack space for temporary variables
    for (const auto subsection : tempvar) {
      if (failed(mlirGenTempVar(subsection)))
        return nullptr;
    }

    // Declare all the function outputs in the symbol table.
    for (const auto name_value : llvm::zip(output_types, output_names)) {
      auto type = std::get<0>(name_value);
      auto name = std::get<1>(name_value);
      auto addressType = AddressType::get(type);
      auto varStorage = builder.create<TempVariableOp>(
          location, addressType, builder.getStringAttr(name));
      if (failed(declare(location, name, addressType, varStorage)))
        return nullptr;
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(code))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return if the current block has no terminator yet.
    if (!blockHasTerminator())
      mlirgenReturn(location);

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
    auto selfType = AddressType::get(idb);
    if (failed(declare(location, "$self", selfType, entryBlock.getArgument(0))))
      return nullptr;

    // Parse the declaration subsections
    const auto &declarations = fb->getDeclarations()->getDecls();
    for (const auto &decl : declarations) {
      if (failed(mlirGen(decl.get())))
        return nullptr;
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(fb->getCode()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return if the current block has no terminator yet.
    if (!blockHasTerminator())
      mlirgenReturn(location);

    return function;
  }

  DataBlockOp mlirGen(const DataBlockAST *db) {
    auto location = loc(db->loc());
    auto context = builder.getContext();

    StringRef name = db->getIdentifier();
    mlir::Type type = getType(db->getType());
    if (!type) return nullptr;
    mlir::TypeAttr typeAttr = mlir::TypeAttr::get(type);
    mlir::StringAttr nameAttr = mlir::StringAttr::get(context, name);

    return builder.create<DataBlockOp>(location, typeAttr, nameAttr);
  }

  /// Register a user-defined type. Emits no op; returns success/failure.
  mlir::LogicalResult mlirGen(const UserDefinedTypeAST *udt) {
    auto type = getType(udt->getType());
    if (!type)
      return mlir::failure();
    udtTypes[udt->getIdentifier()] = type;
    return mlir::success();
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
        .Case<ForDoAST>([&](auto forDo) {
          return mlirGen(forDo);
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
        .Case<RepeatUntilAST>([&](auto repeatUntil) {
          return mlirGen(repeatUntil);
        })
        .Case<WhileDoAST>([&](auto whileDo) {
          return mlirGen(whileDo);
        })
        .Case<DebugPrintAST>([&](auto debugPrint) {
          return mlirGen(debugPrint);
        })
        .Case<CaseOfAST>([&](auto caseOf) { return mlirGen(caseOf); })
        .Case<JumpLabelAST>([&](auto label) { return mlirGen(label); })
        .Case<GotoAST>([&](auto goto_) { return mlirGen(goto_); })
        .Default([&](auto instr) {
          emitError(loc(instr->loc()))
              << "MLIR codegen encountered an unhandled instruction kind '"
              << Twine(instr->getKind()) << "'";
          return mlir::failure();
        });
  }

  /// The declared array type of an expression if it is an array/string
  /// variable (local, parameter, output or FB member), without codegen.
  /// Size of an SCL type in bytes (flat buffer layout).
  int64_t byteSizeOf(mlir::Type t) {
    if (auto it = t.dyn_cast<IntegerType>())
      return (it.getWidth() + 7) / 8;
    if (auto lt = t.dyn_cast<LogicalType>())
      return (lt.getWidth() + 7) / 8;
    if (t.isa<RealType>())
      return 4;
    if (t.isa<S5TimeType, DateType>())
      return 2;
    if (t.isa<TimeType, DateAndTimeType>())
      return t.isa<TimeType>() ? 4 : 8;
    if (auto at = t.dyn_cast<ArrayType>()) {
      int64_t n = 1;
      for (auto d : at.getDimensions())
        n *= d.second - d.first + 1;
      return n * byteSizeOf(at.getElementType());
    }
    if (auto st = t.dyn_cast<StructType>()) {
      int64_t sz = 0;
      for (auto e : st.getElementTypes())
        sz += byteSizeOf(e);
      return sz;
    }
    return -1;
  }

  ArrayType arrayTypeOfAST(const ExpressionAST *expr) {
    auto *var = llvm::dyn_cast<SimpleVariableAST>(expr);
    if (!var || var->isSymbol())
      return nullptr;
    auto sym = symbolTable.lookup(var->getName());
    if (!sym.getType())
      return nullptr;
    mlir::Type t = sym.getType();
    if (auto at = t.dyn_cast<AddressType>())
      t = at.getElementType();
    return t.dyn_cast<ArrayType>();
  }

  /// Array type behind an address value (direct element type or byte view).
  ArrayType arrayTypeOfAddress(mlir::Value addr) {
    if (!addr || !addr.getType().isa<AddressType>())
      return nullptr;
    auto elem = addr.getType().cast<AddressType>().getElementType();
    if (auto at = elem.dyn_cast<ArrayType>())
      return at;
    auto it = byteViewTypes.find(addr);
    if (it != byteViewTypes.end())
      return it->second.dyn_cast<ArrayType>();
    return nullptr;
  }

  /// Address of element `index` of an array-typed lvalue, plus its load.
  mlir::Value arrayElemAddr(mlir::Location loc, mlir::Value arrAddr,
                            ArrayType arrTy, int64_t index,
                            mlir::TypeAttr attr) {
    auto elemTy = arrTy.getElementType();
    auto intTy = getType(tok_int);
    auto i16 = builder.getIntegerType(16);
    auto idx = builder.create<ConstantOp>(
        loc, intTy,
        builder.getIntegerAttr(i16, arrTy.getDimensions()[0].first + index));
    return builder.create<GetIndexOp>(loc, AddressType::get(elemTy), arrAddr,
                                      mlir::ValueRange{idx}, attr);
  }

  /// String/array comparison (s1 = s2, s1 <> 'lit', </<=/>/>=): unrolled
  /// byte-wise compare of the char arrays; a string literal side compares
  /// against its zero-padded bytes. Returns nullptr when neither side is a
  /// string/array/literal (not handled here).
  mlir::Value mlirGenStringCompare(const BinaryExpressionAST *expr) {
    auto location = loc(expr->loc());
    auto op = expr->getOp();
    ArrayType lhsArr = arrayTypeOfAST(expr->getLhs());
    ArrayType rhsArr = arrayTypeOfAST(expr->getRhs());
    const StringConstantAST *lhsLit =
        llvm::dyn_cast<StringConstantAST>(expr->getLhs());
    const StringConstantAST *rhsLit =
        llvm::dyn_cast<StringConstantAST>(expr->getRhs());
    if (!lhsArr && !rhsArr && !lhsLit && !rhsLit)
      return nullptr;
    ArrayType arrTy = lhsArr ? lhsArr : rhsArr;
    if (!arrTy) {
      emitError(location) << "string literal compared with a non-string";
      return nullptr;
    }
    auto elemTy = arrTy.getElementType();
    if (!elemTy.isa<IntegerType>() ||
        elemTy.cast<IntegerType>().getWidth() != 8) {
      emitError(location) << "only string (char array) comparison is supported";
      return nullptr;
    }
    auto dims = arrTy.getDimensions();
    int64_t count = dims[0].second - dims[0].first + 1;
    mlir::Value lhsAddr = lhsLit ? nullptr : mlirGen(expr->getLhs());
    mlir::Value rhsAddr = rhsLit ? nullptr : mlirGen(expr->getRhs());
    if (!lhsLit && !lhsAddr)
      return nullptr;
    if (!rhsLit && !rhsAddr)
      return nullptr;
    if ((!lhsLit && !arrayTypeOfAddress(lhsAddr)) ||
        (!rhsLit && !arrayTypeOfAddress(rhsAddr))) {
      emitError(location) << "cannot compare a string with a non-string";
      return nullptr;
    }
    mlir::TypeAttr attr = mlir::TypeAttr::get(arrTy);
    auto i8 = builder.getIntegerType(8);
    auto i1 = builder.getI1Type();
    auto constBool = [&](bool v) {
      return builder.create<ConstantOp>(
          location, getType(tok_bool),
          builder.getIntegerAttr(i1, v ? 1 : 0));
    };
    // byteAt: i8 value of element i of side (addr or literal)
    auto byteAt = [&](mlir::Value addr, llvm::StringRef lit,
                      int64_t i) -> mlir::Value {
      if (addr)
        return builder.create<LoadOp>(
            location, i8, arrayElemAddr(location, addr, arrTy, i, attr));
      uint8_t byte = i < (int64_t)lit.size() ? (uint8_t)lit[i] : 0;
      return builder.create<ConstantOp>(location, i8,
                                        builder.getIntegerAttr(i8, byte));
    };
    mlir::Value eqAcc = constBool(true), ltAcc = constBool(false),
                gtAcc = constBool(false), prefix = constBool(true);
    for (int64_t i = 0; i < count; ++i) {
      mlir::Value a = byteAt(lhsAddr, lhsLit ? lhsLit->getValue() : "", i);
      mlir::Value b = byteAt(rhsAddr, rhsLit ? rhsLit->getValue() : "", i);
      mlir::Value eq = builder.create<EqualOp>(location, a, b);
      mlir::Value lt = builder.create<LessThanOp>(location, a, b);
      mlir::Value gt = builder.create<GreaterThanOp>(location, a, b);
      eqAcc = builder.create<AndOp>(location, eqAcc, eq);
      ltAcc = builder.create<OrOp>(
          location, ltAcc, builder.create<AndOp>(location, prefix, lt));
      gtAcc = builder.create<OrOp>(
          location, gtAcc, builder.create<AndOp>(location, prefix, gt));
      prefix = builder.create<AndOp>(location, prefix, eq);
    }
    switch (op) {
    case tok_cmp_eq: return eqAcc;
    case tok_cmp_ne: return builder.create<UnaryNotOp>(location, eqAcc);
    case tok_cmp_lt: return ltAcc;
    case tok_cmp_gt: return gtAcc;
    case tok_cmp_le: return builder.create<OrOp>(location, eqAcc, ltAcc);
    case tok_cmp_ge: return builder.create<OrOp>(location, eqAcc, gtAcc);
    default: return nullptr;
    }
  }

  /// Store the bytes of a string literal into a char-array buffer address
  /// (zero-padded to the array length), byte by byte.
  mlir::LogicalResult mlirGenStoreStringBytes(mlir::Location location,
                                              mlir::Value addr,
                                              ArrayType arrTy,
                                              llvm::StringRef value) {
    auto elemTy = arrTy.getElementType();
    if (!elemTy.isa<IntegerType>() ||
        elemTy.cast<IntegerType>().getWidth() != 8) {
      emitError(location) << "string target is not an array of char";
      return mlir::failure();
    }
    auto dims = arrTy.getDimensions();
    if (dims.size() != 1) {
      emitError(location) << "only 1-d strings are supported";
      return mlir::failure();
    }
    int64_t count = dims[0].second - dims[0].first + 1;
    auto intTy = getType(tok_int);
    auto i16 = builder.getIntegerType(16);
    auto charAddrTy = AddressType::get(elemTy);
    auto i8 = builder.getIntegerType(8);
    auto arrayTypeAttr = mlir::TypeAttr::get(arrTy);
    for (int64_t i = 0; i < count; ++i) {
      uint8_t byte = i < (int64_t)value.size() ? (uint8_t)value[i] : 0;
      auto idx = builder.create<ConstantOp>(
          location, intTy, builder.getIntegerAttr(i16, dims[0].first + i));
      auto elem = builder.create<GetIndexOp>(location, charAddrTy, addr,
                                             mlir::ValueRange{idx}, arrayTypeAttr);
      auto byteVal = builder.create<ConstantOp>(
          location, elemTy, builder.getIntegerAttr(i8, byte));
      builder.create<StoreOp>(location, elem, byteVal);
    }
    return mlir::success();
  }

  /// Materialize a string literal as a value of the declared parameter type:
  /// allocate a temp char-array buffer, store the literal bytes (zero-padded)
  /// into it and load the whole array value.
  mlir::Value mlirGenStringLiteralValue(
      mlir::Location location, const llvm::StringMap<mlir::Type> *paramTypes,
      StringRef name, const StringConstantAST *str) {
    if (!paramTypes) {
      emitError(location) << "string literal argument '" << name
                          << "': parameter type unknown";
      return nullptr;
    }
    auto it = paramTypes->find(name);
    if (it == paramTypes->end()) {
      emitError(location) << "string literal argument: unknown parameter '"
                          << name << "'";
      return nullptr;
    }
    auto arrTy = it->second.dyn_cast<ArrayType>();
    if (!arrTy || !arrTy.getElementType().isa<IntegerType>() ||
        arrTy.getElementType().cast<IntegerType>().getWidth() != 8) {
      emitError(location) << "string literal argument to non-string parameter '"
                          << name << "'";
      return nullptr;
    }
    auto addrTy = AddressType::get(arrTy);
    auto storage = builder.create<TempVariableOp>(location, addrTy,
                                                  builder.getStringAttr("strlit"));
    if (failed(mlirGenStoreStringBytes(location, storage, arrTy,
                                       str->getValue())))
      return nullptr;
    return builder.create<LoadOp>(location, arrTy, storage);
  }

  /// Assign a string literal to a string variable: store the literal bytes
  /// (zero-padded to the target length) into the char buffer byte by byte.
  mlir::LogicalResult mlirGenStringAssignment(mlir::Location location,
                                              const ExpressionAST *lvalue,
                                              const StringConstantAST *str) {
    mlir::Value lhs = mlirGenLValue(lvalue);
    if (!lhs)
      return mlir::failure();
    auto baseElem = lhs.getType().cast<AddressType>().getElementType();
    // char := 'A': a string literal assigns its first byte to a char
    if (auto it8 = baseElem.dyn_cast<IntegerType>()) {
      if (it8.getWidth() == 8 && !str->getValue().empty()) {
        auto byteVal = builder.create<ConstantOp>(
            location, baseElem,
            builder.getIntegerAttr(builder.getIntegerType(8),
                                   (uint8_t)str->getValue()[0]));
        builder.create<StoreOp>(location, lhs, byteVal);
        return mlir::success();
      }
    }
    ArrayType arrTy;
    if (auto at = baseElem.dyn_cast<ArrayType>()) {
      arrTy = at;
    } else {
      auto it = byteViewTypes.find(lhs);
      if (it == byteViewTypes.end() || !it->second.isa<ArrayType>()) {
        emitError(location) << "string literal assigned to non-string target";
        return mlir::failure();
      }
      arrTy = it->second.cast<ArrayType>();
    }
    return mlirGenStoreStringBytes(location, lhs, arrTy, str->getValue());
  }

  /// Codegen a variable assignment
  mlir::LogicalResult mlirGen(const ValueAssignmentAST *instr) {
    auto location = loc(instr->loc());

    const auto expr = llvm::cast<BinaryExpressionAST>(instr->getExpression());
    assert(expr->getOp() == tok_assignment);

    if (auto *str = llvm::dyn_cast<StringConstantAST>(expr->getRhs()))
      return mlirGenStringAssignment(location, expr->getLhs(), str);

    mlir::Value lhs = mlirGenLValue(expr->getLhs());
    if (!lhs)
      return mlir::failure();

    // Whole-array (incl. string) var-to-var assignment `a := b`: copy the
    // elements one by one (the buffers are flat byte/element buffers).
    if (ArrayType lhsArr = arrayTypeOfAddress(lhs)) {
      ArrayType rhsArr = arrayTypeOfAST(expr->getRhs());
      if (rhsArr && lhsArr.getElementType() == rhsArr.getElementType() &&
          lhsArr.getDimensions() == rhsArr.getDimensions()) {
        mlir::Value rhsAddr = mlirGen(expr->getRhs());
        if (!rhsAddr)
          return mlir::failure();
        auto dims = lhsArr.getDimensions();
        int64_t count = dims[0].second - dims[0].first + 1;
        // Composite elements (struct/array): copy the flat byte buffer
        // through a char-array view of the same size.
        mlir::Type elemTy = lhsArr.getElementType();
        ArrayType copyTy = lhsArr;
        int64_t elemBytes = byteSizeOf(elemTy);
        if (elemBytes <= 0)
          return mlir::failure();
        if (!elemTy.isa<IntegerType>()) {
          auto i8 = IntegerType::get(builder.getContext(), 8);
          copyTy = ArrayType::get({{0, count * elemBytes - 1}}, i8);
          elemTy = i8;
          count *= elemBytes;
        }
        mlir::TypeAttr attr = mlir::TypeAttr::get(copyTy);
        for (int64_t i = 0; i < count; ++i) {
          mlir::Value dst = arrayElemAddr(location, lhs, copyTy, i, attr);
          mlir::Value src = arrayElemAddr(location, rhsAddr, copyTy, i, attr);
          mlir::Value v = builder.create<LoadOp>(location, elemTy, src);
          builder.create<StoreOp>(location, dst, v);
        }
        return mlir::success();
      }
    }

    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return mlir::failure();

    // Implicit integer conversion on assignment (e.g. char -> int). For
    // byte-view member addresses use the declared member type, not the i8
    // address element type.
    auto lhsElem = lhs.getType().cast<AddressType>().getElementType();
    auto bvIt = byteViewTypes.find(lhs);
    mlir::Type declared = bvIt != byteViewTypes.end() ? bvIt->second : lhsElem;
    // Implicit integer conversion on width mismatch. byte/word/dword are
    // LogicalType (not IntegerType), so cast to an IntegerType of the same
    // width; storing a wider value into a byte view would otherwise write
    // past the view and read back the high byte.
    auto widthOfTy = [](mlir::Type t) -> int {
      if (auto it = t.dyn_cast<IntegerType>())
        return it.getWidth();
      if (auto lt = t.dyn_cast<LogicalType>())
        return lt.getWidth();
      return -1;
    };
    int dw = widthOfTy(declared), rw = widthOfTy(rhs.getType());
    if (dw > 0 && rw > 0 && dw != rw && rhs.getType().isa<IntegerType>()) {
      mlir::Type castTy = declared.isa<IntegerType>()
                              ? declared
                              : IntegerType::get(builder.getContext(), dw);
      rhs = builder.create<IntegerCastOp>(location, castTy, rhs);
    }

    builder.create<StoreOp>(location, lhs, rhs);

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const SubroutineProcessingAST *call) {
    return mlirGenVoid(call->getCall());
  }

  mlir::Value mlirGenLValue(const ExpressionAST *expr) {
    auto location = loc(expr->loc());
    mlir::Value address = mlirGen(expr);

    if (!address.getType().isa<AddressType>()) {
      emitError(location) << "not a lvalue, type " << address.getType();
      return nullptr;
    }

    return address;
  }

  mlir::Value mlirGen(const ExpressionAST *expr) {
    return TypeSwitch<const ExpressionAST *, mlir::Value>(expr)
        .Case<IntegerConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<RealConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<StringConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<TimeConstantAST>([&](auto expr) { return mlirGen(expr); })
        .Case<SimpleVariableAST>([&](auto expr) { return mlirGen(expr); })
        .Case<DirectVariableAST>([&](auto expr) { return mlirGen(expr); })
        .Case<IndexedVariableAST>([&](auto expr) { return mlirGen(expr); })
        .Case<BinaryExpressionAST>([&](auto expr) { return mlirGen(expr); })
        .Case<UnaryExpressionAST>([&](auto expr) { return mlirGen(expr); })
        .Case<FunctionCallAST>([&](auto expr) { return mlirGen(expr); })
        .Default([&](auto expr) {
          emitError(loc(expr->loc())) << "expression kind not implemented";
          return nullptr; // TODO: TBD
        });
  }

  mlir::Value mlirGenRValue(const ExpressionAST *expr) {
    // A direct address (memory area / DB) loads with the value type of the
    // access width (bool for bit access, else an integer of that width).
    if (auto *dir = llvm::dyn_cast<DirectVariableAST>(expr)) {
      auto addr = mlirGen(dir);
      if (!addr)
        return nullptr;
      auto directTy = addr.getType().cast<AddressType>()
                          .getElementType().cast<DirectType>();
      return builder.create<LoadOp>(loc(expr->loc()), directTy.getValueType(),
                                    addr);
    }
    // FB/struct members are addressed as i8 views; load with the declared
    // member type so the value has the right type.
    if (auto *var = llvm::dyn_cast<SimpleVariableAST>(expr)) {
      if (!var->isSymbol()) {
        auto sym = symbolTable.lookup(var->getName());
        if (sym.isElement()) {
          auto addr = mlirGen(var);
          if (!addr)
            return nullptr;
          return builder.create<LoadOp>(loc(expr->loc()), sym.getType(), addr);
        }
      }
    }
    if (auto *bin = llvm::dyn_cast<BinaryExpressionAST>(expr)) {
      if (bin->getOp() == tok_dot) {
        auto addr = mlirGen(expr);
        if (!addr)
          return nullptr;
        // Declared member type: tracked for byte-view member addresses,
        // otherwise derived from the base type.
        mlir::Type ty = nullptr;
        auto it = byteViewTypes.find(addr);
        if (it != byteViewTypes.end())
          ty = it->second;
        else {
          auto base = mlirGen(bin->getLhs());
          auto name = llvm::cast<SimpleVariableAST>(bin->getRhs())->getName();
          ty = base ? memberDeclaredType(base, name) : nullptr;
        }
        if (!ty)
          return nullptr;
        return builder.create<LoadOp>(loc(expr->loc()), ty, addr);
      }
    }
    if (llvm::isa<IndexedVariableAST>(expr)) {
      // Array member of a FB/struct: the element address is a byte view;
      // load with the declared element type.
      auto addr = mlirGen(expr);
      if (!addr)
        return nullptr;
      auto it = byteViewTypes.find(addr);
      if (it != byteViewTypes.end())
        return builder.create<LoadOp>(loc(expr->loc()), it->second, addr);
    }
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
    // Boolean literals are typed bool (i1); other integer literals are int.
    mlir::Type type = getType(tok_int);
    unsigned width = 16;
    if (expr->getType() == tok_bool) {
      type = getType(tok_bool);
      width = 1;
    }
    auto attrType = builder.getIntegerType(width);
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

  unsigned int getTimeMS(const TimeConstantAST *expr) {
    unsigned int time = 0;
    time += expr->getDay();
    time *= 24;
    time += expr->getHour();
    time *= 60;
    time += expr->getMinute();
    time *= 60;
    time += expr->getSec();
    time *= 1000;
    time += expr->getMSec();

    return time;
  }

  mlir::Value mlirGen(const TimeConstantAST *expr) {
    auto location = loc(expr->loc());
    int year = expr->getYear();
    int month = expr->getMonth();
    int day = expr->getDay();

    unsigned int time = 0;
    time += expr->getDay();
    time *= 24;
    time += expr->getHour();
    time *= 60;
    time += expr->getMinute();
    time *= 60;
    time += expr->getSec();
    time *= 1000;
    time += expr->getMSec();

    switch (expr->getType()) {
    default:
      assert(false);
    case tok_date:
      if (year >= 1990 && year <= 2168 &&
          month >= 1 && month <= 12 &&
          day >= 1 && day <= 31)
        return builder.create<ConstantDateOp>(location, year, month, day);
      emitError(location) << "invalid date " <<
                             year << '-' << month << '-' << day;
      return nullptr;

    case tok_date_and_time:
      return builder.create<ConstantDateAndTimeOp>(location, year, month, day,
           expr->getHour(), expr->getMinute(), expr->getSec(), expr->getMSec());
    case tok_s5time:
      return builder.create<ConstantS5TimeOp>(location, time);
    case tok_time:
      return builder.create<ConstantTimeOp>(location, time);
    case tok_time_of_day:
      return builder.create<ConstantTimeOfDayOp>(location, time);    }
  }

  mlir::Value mlirGenVariable(mlir::Location location, StringRef name) {
    auto variable = symbolTable.lookup(name);
    if (variable.isElement()) {
      // FB members live in the instance DB byte buffer; address them as i8.
      auto i8 = IntegerType::get(builder.getContext(), 8);
      auto type = AddressType::get(i8);
      auto self = symbolTable.lookup("$self").getValue();
      assert(self);
      auto result = builder.create<GetVariableOp>(location, type, self, name);
      byteViewTypes[result] = variable.getType();
      return result;
    }
    if (variable.isAddress())
      return variable.getValue();
    if (variable.isDirect())
      return variable.getValue();
    emitError(location) << "unknown variable '" << name << "'";
    return nullptr;
  }

  mlir::Value mlirGenSymbol(mlir::Location location, StringRef name) {
    auto nameAttr = builder.getStringAttr(name);
    DataBlockOp db =
        mlir::SymbolTable::lookupNearestSymbolFrom<DataBlockOp>(theModule, nameAttr);
    if (!db) {
      emitError(location) << "unknown symbol '" << name << "'";
      return nullptr;
    }
    mlir::Type type = db.getType();
    mlir::Type address = AddressType::get(type);
    return builder.create<GetGlobalOp>(location, address, name);
  }

  StructType mlirGenIdbStruct(mlir::Location location, FunctionBlockOp fb) {
    SmallVector<mlir::StringAttr, 4> names;
    SmallVector<mlir::Type, 4> types;

    return StructType::get(names, types);
  }

  /// The declared type of a struct/IDB member (used to type loads).
  mlir::Type memberDeclaredType(mlir::Value base, StringRef name) {
    auto baseTy = base.getType().cast<AddressType>().getElementType();
    if (auto idbTy = baseTy.dyn_cast<InstanceDbType>()) {
      auto fbName = mlir::StringAttr::get(theModule->getContext(),
                                          idbTy.getFbSymbol());
      auto fb = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
          theModule, fbName);
      if (!fb)
        return nullptr;
      auto var = mlir::dyn_cast_or_null<VariableOp>(
          mlir::SymbolTable::lookupSymbolIn(fb, name));
      return var ? var.getType() : nullptr;
    }
    if (auto structTy = baseTy.dyn_cast<StructType>())
      return structTy.getElementType(name);
    return nullptr;
  }

  mlir::Value mlirGenElement(mlir::Location location,
                             mlir::Value address, StringRef name) {
    mlir::Type baseType = address.getType().cast<AddressType>().getElementType();
    auto i8 = IntegerType::get(builder.getContext(), 8);
    mlir::Type resultType = AddressType::get(i8);

    return mlir::TypeSwitch<mlir::Type, mlir::Value>(baseType)
    .Case<InstanceDbType>([&](auto idbType) -> mlir::Value {
      auto fbName = builder.getStringAttr(idbType.getFbSymbol());
      auto fb = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
        theModule, fbName);
      // Builtin timer FBs have no FunctionBlockOp; their member types are
      // fixed (IN/Q: bool, PT/ET: time).
      std::string fbUp = std::string(fbName.getValue().upper());
      if (!fb && (fbUp == "TON" || fbUp == "TOF" || fbUp == "TP")) {
        mlir::Type t;
        std::string mUp = std::string(llvm::StringRef(name).upper());
        if (mUp == "IN" || mUp == "Q")
          t = getType(tok_bool);
        else if (mUp == "PT" || mUp == "ET")
          t = getType(tok_time);
        if (!t) {
          emitError(location) << "unknown variable '" << name
                              << "' in builtin FB '" << fbName << "'";
          return nullptr;
        }
        auto result = builder.create<GetVariableOp>(location, resultType,
                                                    address, name);
        byteViewTypes[result] = t;
        return result;
      }
      if (!fb) {
        emitError(location) << "unknown FB '" << fbName << "'";
        return nullptr;
      }
      auto var = mlir::dyn_cast_or_null<VariableOp>(mlir::SymbolTable::lookupSymbolIn(fb, name));
      if (!var) {
        emitError(location) << "unknown variable '" << name << "' in FB '" << fbName << "'";
        return nullptr;
      }
      // Address the member as an i8 view into the instance byte buffer.
      auto result = builder.create<GetVariableOp>(location, resultType, address, name);
      byteViewTypes[result] = var.getType();
      return result;
    })
    .Case<StructType>([&](auto structType) -> mlir::Value {
      auto result = builder.create<GetElementOp>(location, resultType, address,
                                                 name, mlir::TypeAttr());
      byteViewTypes[result] = structType.getElementType(name);
      return result;
    })
    .Case<IntegerType>([&](auto charType) -> mlir::Value {
      // Nested struct member: the base is itself a byte view; the declared
      // struct type is tracked in byteViewTypes.
      if (charType.getWidth() != 8) {
        emitError(location) << "unsupported base type " << baseType
                            << " for element '" << name << "'";
        return nullptr;
      }
      auto it = byteViewTypes.find(address);
      if (it == byteViewTypes.end() || !it->second.isa<StructType>()) {
        emitError(location) << "unsupported base type " << baseType
                            << " for element '" << name << "'";
        return nullptr;
      }
      auto structType = it->second.cast<StructType>();
      auto memberTy = structType.getElementType(name);
      if (!memberTy) {
        emitError(location) << "unknown element '" << name << "' in struct";
        return nullptr;
      }
      auto baseAttr = mlir::TypeAttr::get(structType);
      auto result = builder.create<GetElementOp>(location, resultType, address,
                                                 name, baseAttr);
      byteViewTypes[result] = memberTy;
      return result;
    })
    .Default([&](auto baseType) {
      emitError(location) << "unsupported base type " << baseType << " for element '" << name << "'";
      return nullptr;
    });
  }

  mlir::Value mlirGen(const SimpleVariableAST *expr) {
    auto location = loc(expr->loc());
    auto name = expr->getName();

    if (expr->isSymbol()) {
      return mlirGenSymbol(location, name);
    } else {
      return mlirGenVariable(location, name);
    }
  }

  /// Codegen a directly-represented variable (memory area / DB operand,
  /// e.g. `%I0.0`, `%MW10`, `%DB1.DBW4`) to a `scl.getdirect` address.
  mlir::Value mlirGen(const DirectVariableAST *expr) {
    auto location = loc(expr->loc());
    auto area = DirectType::areaFromName(expr->getArea());
    if (!area) {
      emitError(location) << "unknown direct address area '" << expr->getArea()
                          << "'";
      return nullptr;
    }
    // %T/%C go through the runtime like the other areas; the sim treats them
    // as plain byte areas and the virtual clock advances via scl_rt_cycle
    // (full IEC timer semantics are a known gap).
    return builder.create<GetDirectOp>(location, *area, expr->getDb(),
                                       expr->getByte(), expr->getBit(),
                                       expr->getWidth());
  }

  mlir::Value mlirGen(const IndexedVariableAST *expr) {
    auto location = loc(expr->loc());

    mlir::Value base = mlirGen(expr->getBase());
    if (!base)
      return nullptr;
    auto baseElem = base.getType().cast<AddressType>().getElementType();
    mlir::Type resultElem;
    mlir::TypeAttr arrayTypeAttr;
    ArrayType arrayTy;
    if (auto at = baseElem.dyn_cast<ArrayType>()) {
      // Plain array variable: element address has the element type.
      arrayTy = at;
      resultElem = at.getElementType();
    } else if (baseElem.isa<IntegerType>() &&
               baseElem.cast<IntegerType>().getWidth() == 8) {
      // Array member of a FB/struct: the base is a byte view; keep the
      // result a byte view and record the declared array type on the op.
      auto it = byteViewTypes.find(base);
      if (it == byteViewTypes.end() || !it->second.isa<ArrayType>()) {
        emitError(location) << "cannot index into '" << baseElem << "'";
        return nullptr;
      }
      arrayTy = it->second.cast<ArrayType>();
      resultElem = baseElem;
      arrayTypeAttr = mlir::TypeAttr::get(arrayTy);
    } else {
      emitError(location) << "cannot index into '" << baseElem << "'";
      return nullptr;
    }
    mlir::Type resultType = AddressType::get(resultElem);
    SmallVector<mlir::Value, 1> indices;
    for (const auto & index : expr->getIndices()) {
      indices.push_back(mlirGenRValue(index.get()));
    }
    auto result = builder.create<GetIndexOp>(location, resultType, base, indices,
                                             arrayTypeAttr);
    if (arrayTypeAttr)
      byteViewTypes[result] = arrayTy.getElementType();
    return result;
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
    if (expr->getOp() == tok_dot) {
      // this BinaryExpression is a STRUCT element access:
      // the right hand side is special, it's not a normal expression but a name
      mlir::Value lhs = mlirGen(expr->getLhs());
      if (!lhs)
        return nullptr;
      if (!isa<SimpleVariableAST>(expr->getRhs())) {
        emitError(loc(expr->getRhs()->loc()), "expected element name");
        return nullptr;
      }
      auto name = llvm::cast<SimpleVariableAST>(expr->getRhs())->getName();
      return mlirGenElement(location, lhs, name);
    }
    // String/array comparisons are intercepted before the generic rvalue
    // conversion, which cannot load a whole array.
    switch (expr->getOp()) {
    case tok_cmp_eq:
    case tok_cmp_ne:
    case tok_cmp_lt:
    case tok_cmp_le:
    case tok_cmp_gt:
    case tok_cmp_ge:
      if (mlir::Value strCmp = mlirGenStringCompare(expr))
        return strCmp;
      break;
    default:
      break;
    }
    // Go through the AST overload so that FB/DB members (addressed as i8
    // views) are loaded with their declared member type.
    mlir::Value lhs = mlirGenRValue(expr->getLhs());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return nullptr;

    // Implicit integer widening for mixed-width operands (e.g. char + int,
    // byte + word): normalize both operands to an IntegerType of the wider
    // width, like SCL. LogicalType (byte/word/dword/bool) values are
    // bitcast to the same-width IntegerType first (IntegerCastOp only
    // accepts IntegerType).
    auto widthOfTy = [](mlir::Type t) -> int {
      if (auto it = t.dyn_cast<IntegerType>())
        return it.getWidth();
      if (auto lt = t.dyn_cast<LogicalType>())
        return lt.getWidth();
      return -1;
    };
    auto toIntOfWidth = [&](mlir::Value v, int w) -> mlir::Value {
      int vw = widthOfTy(v.getType());
      if (vw < 0 || (vw == w && v.getType().isa<IntegerType>()))
        return v;
      // LogicalType (byte/word/dword/bool) is unsigned: zero-extend.
      bool zext = v.getType().isa<LogicalType>();
      auto op = builder.create<IntegerCastOp>(
          location, IntegerType::get(builder.getContext(), w), v);
      if (zext)
        op.setZextAttr(builder.getUnitAttr());
      return op.getResult();
    };
    int lw = widthOfTy(lhs.getType()), rw = widthOfTy(rhs.getType());
    if (lw > 0 && rw > 0 && lw != rw) {
      int w = std::max(lw, rw);
      lhs = toIntOfWidth(lhs, w);
      rhs = toIntOfWidth(rhs, w);
    }

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

  mlir::LogicalResult mlirGenVoid(const FunctionCallAST *expr) {
    auto location = loc(expr->loc());

    return TypeSwitch<const ExpressionAST *, mlir::LogicalResult>(expr->getFunction())
    .Case<SimpleVariableAST>([&](auto callee) {
      mlirGenFcCall(expr, callee->getName());
      return mlir::success();
    })
    .Case<BinaryExpressionAST>([&](auto callee) {
      if (callee->getOp() != tok_dot)
        return mlir::failure();
      auto fbVar = dyn_cast<SimpleVariableAST>(callee->getLhs());
      if (!fbVar) {
        emitError(location, "invalid fb in function call");
        return mlir::failure();
      }
      auto idbVar = dyn_cast<SimpleVariableAST>(callee->getRhs());
      if (!idbVar) {
        emitError(location, "invalid instance db in function call");
        return mlir::failure();
      }
      if (failed(mlirGenFbCall(expr, idbVar->getName(), fbVar->getName()))) {
        emitError(location, "invalid FB call");
        return mlir::failure();
      }
      return mlir::success();
    })
    .Default([&](auto callee) {
      emitError(location, "invalid callee in function or FB call");
      return mlir::failure();
    });
  }

  mlir::Value mlirGen(const FunctionCallAST *expr) {
    auto location = loc(expr->loc());

    // The function name is parsed as a variable
    auto callee = dyn_cast<SimpleVariableAST>(expr->getFunction());
    if (!callee) {
      emitError(location, "invalid callee in function call");
      return nullptr;
    }
    return mlirGenFcCall(expr, callee->getName());
   }

  mlir::LogicalResult mlirGenCallParameters(
      const FunctionCallAST *fc,
      std::function<void (StringRef, mlir::Value)> fN,
      std::function<void (mlir::Value)> fNN,
      const llvm::StringMap<mlir::Type> *paramTypes = nullptr) {
    for (auto &arg : fc->getParameters()) {
      auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
      if (binary && binary->getOp() == tok_assignment) {
        auto lhs = binary->getLhs();
        auto name = dyn_cast<SimpleVariableAST>(lhs);
        if (!name || name->isSymbol()) {
          emitError(loc(lhs->loc()), "invalid parameter name");
          return mlir::failure();
        }
        mlir::Value value;
        if (auto *str = dyn_cast<StringConstantAST>(binary->getRhs()))
          value = mlirGenStringLiteralValue(loc(binary->getRhs()->loc()),
                                            paramTypes, name->getName(), str);
        else
          value = mlirGenRValue(binary->getRhs());
        if (!value)
          return mlir::failure();
        fN(name->getName(), value);
      } else {
        if (!fNN) {
          emitError(loc(arg->loc()), "parameter without name");
          return mlir::failure();
        }
        if (fc->getParameters().size() != 1) {
          emitError(loc(arg->loc()), "parameter without name");
          return mlir::failure();
        }
        fNN(mlirGenRValue(arg.get()));
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGenCallParameters(
      const FunctionCallAST *fc,
      std::function<void (StringRef, mlir::Value)> f,
      const llvm::StringMap<mlir::Type> *paramTypes = nullptr) {
    return mlirGenCallParameters(fc, f, nullptr, paramTypes);
  }

  mlir::Value mlirGenFcCall(const FunctionCallAST *expr, StringRef callee) {
    auto location = loc(expr->loc());
    auto context = builder.getContext();

    // get arguments
    SmallVector<mlir::Value, 4> arguments;
    SmallVector<mlir::Attribute, 4> argNames;
    const llvm::StringMap<mlir::Type> *paramTypes = nullptr;
    auto ptIt = callableVarTypes.find(callee);
    if (ptIt != callableVarTypes.end())
      paramTypes = &ptIt->second;
    if (failed(mlirGenCallParameters(expr, [&] (StringRef name, mlir::Value value) {
      argNames.push_back(mlir::StringAttr::get(context, name));
      arguments.push_back(value);
      return mlir::success();
    }, [&] (mlir::Value value) {
      arguments.push_back(value);
    }, paramTypes)))
      return nullptr;

    auto rtIt = functionResultTypes.find(callee);
    if (rtIt == functionResultTypes.end()) {
      emitError(location) << "unknown function '" << callee << "'";
      return nullptr;
    }
    mlir::Type resultType = rtIt->second;
    if (resultType.isa<mlir::NoneType>()) {
      emitError(location) << "void function '" << callee << "' used as a value";
      return nullptr;
    }

    auto argNamesAttr = mlir::ArrayAttr::get(context, argNames);
    return builder.create<CallFcOp>(location, resultType, callee, arguments, argNamesAttr);
  }

  mlir::LogicalResult mlirGenFbCall(const FunctionCallAST *expr, StringRef idb, StringRef fb) {
    auto location = loc(expr->loc());

    auto idbType = InstanceDbType::get(builder.getContext(), fb);
    auto argType = AddressType::get(idbType);
    auto idbRef = builder.create<GetGlobalOp>(location, argType, idb);
    if (!idbRef) {
      emitError(location, "invalid IDB reference");
      return mlir::failure();
    }

    const llvm::StringMap<mlir::Type> *paramTypes = nullptr;
    auto ptIt = callableVarTypes.find(fb);
    if (ptIt != callableVarTypes.end())
      paramTypes = &ptIt->second;
    if (failed(mlirGenCallParameters(expr, [&] (StringRef name, mlir::Value value) {
      auto elem = mlirGenElement(location, idbRef, name);
      builder.create<StoreOp>(location, elem, value);
      return mlir::success();
    }, nullptr, paramTypes)))
        return mlir::failure();

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
    case tok_date:
      return DateType::get(builder.getContext());
    case tok_date_and_time:
      return DateAndTimeType::get(builder.getContext());
    case tok_s5time:
      return S5TimeType::get(builder.getContext());
    case tok_time:
      return TimeType::get(builder.getContext());
    case tok_time_of_day:
      return TimeOfDayType::get(builder.getContext());
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
    case ElementaryDataTypeAST::Type_Date:
      return getType(tok_date);
    case ElementaryDataTypeAST::Type_DateAndTime:
      return getType(tok_date_and_time);
    case ElementaryDataTypeAST::Type_S5Time:
      return getType(tok_s5time);
    case ElementaryDataTypeAST::Type_Time:
      return getType(tok_time);
    case ElementaryDataTypeAST::Type_TimeOfDay:
      return getType(tok_time_of_day);
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
    .Case<SimpleVariableAST>([&](auto expr) {
      auto location = loc(expr->loc());
      auto constName = expr->getName();
      auto constSymbol = symbolTable.lookup(constName);
      if (!constSymbol.isConst()) {
        emitError(location) << "not a constant: " << constName;
        return mlir::failure();
      }

      return getConstantInteger(constSymbol.getConstValue(), value);
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
    llvm::SmallVector<mlir::StringAttr, 1> names;
    llvm::SmallVector<mlir::Type, 1> types;

    auto context = builder.getContext();
    auto components = type->getComponents();
    names.reserve(components.size());
    types.reserve(components.size());

    for (auto &component : components) {
      auto name = mlir::StringAttr::get(context, component.get()->getName());
      names.push_back(name);

      mlir::Type type = getType(component.get()->getDataType());
      if (!type)
        return nullptr;
      types.push_back(type);
    }
    return StructType::get(names, types);
  }

  mlir::Type getType(const UserDefinedTypeIdentifierAST *type) {
    auto it = udtTypes.find(type->getName());
    if (it != udtTypes.end())
      return it->second;
    // A reference to a function block type is an instance DB.
    if (fbNames.contains(type->getName()))
      return InstanceDbType::get(builder.getContext(), type->getName());
    // Builtin timer FBs (lowered to @scl_rt_timer_fb, no source definition)
    std::string up = llvm::StringRef(type->getName()).upper();
    if (up == "TON" || up == "TOF" || up == "TP")
      return InstanceDbType::get(builder.getContext(), up);
    emitError(loc(type->loc())) << "unknown type '" << type->getName() << "'";
    return nullptr;
  }

  mlir::Type getType(const StringDataTypeSpecAST *type) {
    // A string is a fixed-size buffer of chars (maxLen bytes).
    auto charTy = IntegerType::get(builder.getContext(), 8);
    std::vector<ArrayType::DimTy> dims;
    dims.push_back(std::make_pair(0, (int)type->getMaxLen() - 1));
    return ArrayType::get(dims, charTy);
  }

  mlir::Type getType(const DataTypeSpecAST *type) {
    return TypeSwitch<const DataTypeSpecAST *, mlir::Type>(type)
        .Case<ElementaryDataTypeAST>([&](auto type) { return getType(type); })
        .Case<StringDataTypeSpecAST>([&](auto type) { return getType(type); })
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

  mlir::LogicalResult mlirGen(const ForDoAST *forDo) {
    auto location = loc(forDo->loc());

    auto assign = llvm::cast<BinaryExpressionAST>(forDo->getAssignment());
    if (!assign || assign->getOp() != tok_assignment) {
      emitError(location) << "expected assignment";
      return mlir::failure();
    }

    mlir::Value var = mlirGen(assign->getLhs());
    mlir::Value start = mlirGenRValue(assign->getRhs());
    mlir::Value last = mlirGenRValue(forDo->getLast());
    mlir::Value incr = nullptr;
    if (forDo->getIncrement().has_value())
      incr = mlirGenRValue(forDo->getIncrement().value());

    auto forDoOp = builder.create<ForDoOp>(location, var, start, last, incr);

    auto old = builder.saveInsertionPoint();

    builder.createBlock(&forDoOp.getDoBody());
    if (failed(mlirGen(forDo->getCode())))
      return mlir::failure();

    createEndOp(location);
    builder.restoreInsertionPoint(old);

    return mlir::success();
  }

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
        createEndOp(location);
      }
      builder.createBlock(&cond.getThenBody());
      if (failed(mlirGen(ifThen->getCodeBlock())))
        return mlir::failure();
      createEndOp(location);
      builder.createBlock(&cond.getElseBody());
    }
    if (ifThenElse->getElseBlock()) {
      if (failed(mlirGen(ifThenElse->getElseBlock().value())))
        return mlir::failure();
    }
    createEndOp(loc(ifThenElse->loc()));

    builder.restoreInsertionPoint(old);

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const WhileDoAST *whileDo) {
    auto location = loc(whileDo->loc());

    auto whileDoOp = builder.create<WhileDoOp>(location);

    auto old = builder.saveInsertionPoint();

    builder.createBlock(&whileDoOp.getWhileBody());
    auto condition = mlirGenRValue(whileDo->getCondition());
    if (!condition)
      return mlir::failure();
    // Successor placeholders; the loop lowering sets the real targets.
    auto *condBlock = &whileDoOp.getWhileBody().front();
    auto *dummy = builder.createBlock(&whileDoOp.getWhileBody());
    builder.setInsertionPointToEnd(condBlock);
    builder.create<ConditionOp>(location, condition, dummy, dummy);
    builder.setInsertionPointToEnd(dummy);
    createEndOp(location);

    builder.createBlock(&whileDoOp.getDoBody());
    if (failed(mlirGen(whileDo->getCode())))
      return mlir::failure();
    createEndOp(location);

    builder.restoreInsertionPoint(old);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const RepeatUntilAST *repeatUntil) {
    auto location = loc(repeatUntil->loc());

    auto repeatOp = builder.create<RepeatOp>(location);
    auto old = builder.saveInsertionPoint();

    builder.createBlock(&repeatOp.getRepeatBody());
    if (failed(mlirGen(repeatUntil->getCode())))
      return mlir::failure();

    auto until = mlirGenRValue(repeatUntil->getCondition());
    if (!until)
      return mlir::failure();
    // Successor placeholders; the loop lowering sets the real targets.
    auto *bodyBlock = &repeatOp.getRepeatBody().front();
    auto *dummy = builder.createBlock(&repeatOp.getRepeatBody());
    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<UntilOp>(location, until, dummy, dummy);
    builder.setInsertionPointToEnd(dummy);
    createEndOp(location);

    builder.restoreInsertionPoint(old);
    return mlir::success();
  }

  mlir::LogicalResult getConstantReal(const ExpressionAST *expr,
                                       double &value) {
    return TypeSwitch<const ExpressionAST *, mlir::LogicalResult>(expr)
        .Case<RealConstantAST>([&](auto expr) {
          value = expr->getValue();
          return mlir::success();
        })
        .Case<IntegerConstantAST>([&](auto expr) {
          value = expr->getValue();
          return mlir::success();
        })
        .Case<SimpleVariableAST>([&](auto expr) {
          auto constSymbol = symbolTable.lookup(expr->getName());
          if (!constSymbol.isConst()) {
            emitError(loc(expr->loc())) << "not a constant: " << expr->getName();
            return mlir::failure();
          }
          return getConstantReal(constSymbol.getConstValue(), value);
        })
        .Default([&](auto expr) {
          emitError(loc(expr->loc())) << "constant real expected";
          return mlir::failure();
        });
  }

  /// CASE on a real selector: cf.switch needs integers, so emit a nested
  /// if-chain comparing the selector with cmpf-oeq against each case value.
  mlir::LogicalResult mlirGenRealCase(mlir::Location location,
                                      const CaseOfAST *caseOf,
                                      mlir::Value selector) {
    struct CaseGroup {
      std::vector<double> values;
      std::vector<const InstructionAST *> body;
    };
    std::vector<CaseGroup> groups;
    for (const auto &instr : caseOf->getCode()->getInstructions()) {
      if (auto *label = llvm::dyn_cast<CaseLabelAST>(instr.get())) {
        CaseGroup group;
        for (const auto &value : label->getValues()) {
          double v;
          if (failed(getConstantReal(value.get(), v)))
            return mlir::failure();
          group.values.push_back(v);
        }
        groups.push_back(std::move(group));
      } else {
        if (groups.empty()) {
          emitError(loc(instr->loc()))
              << "instruction before the first case label";
          return mlir::failure();
        }
        groups.back().body.push_back(instr.get());
      }
    }
    auto old = builder.saveInsertionPoint();
    auto f32 = builder.getF32Type();
    std::function<mlir::LogicalResult(unsigned)> emitChain =
        [&](unsigned i) -> mlir::LogicalResult {
      mlir::Value cond;
      for (double v : groups[i].values) {
        auto c = builder.create<ConstantOp>(
            location, selector.getType(), builder.getFloatAttr(f32, v));
        auto eq = builder.create<EqualOp>(location, selector, c);
        cond = cond ? mlir::Value(builder.create<OrOp>(location, cond, eq))
                    : mlir::Value(eq);
      }
      if (!cond)
        cond = builder.create<ConstantOp>(
            location, getType(tok_bool),
            builder.getIntegerAttr(builder.getI1Type(), 1));
      auto ifOp = builder.create<IfThenElseOp>(location, cond);
      builder.createBlock(&ifOp.getThenBody());
      builder.setInsertionPointToStart(&ifOp.getThenBody().front());
      for (auto *instr : groups[i].body)
        if (failed(mlirGen(instr)))
          return mlir::failure();
      createEndOp(location);
      mlir::Block *elseB = builder.createBlock(&ifOp.getElseBody());
      builder.setInsertionPointToStart(elseB);
      mlir::LogicalResult r = mlir::success();
      if (i + 1 < groups.size()) {
        r = emitChain(i + 1);
      } else if (caseOf->getElseBlock().has_value()) {
        r = mlirGen(caseOf->getElseBlock().value());
      }
      // the recursion left the insertion point inside the nested if; go back
      // to this else block to terminate it
      builder.setInsertionPointToEnd(elseB);
      createEndOp(location);
      return r;
    };
    mlir::LogicalResult r = mlir::success();
    if (!groups.empty())
      r = emitChain(0);
    else if (caseOf->getElseBlock().has_value())
      r = mlirGen(caseOf->getElseBlock().value());
    builder.restoreInsertionPoint(old);
    return r;
  }

  /// Codegen a case-of selection into a scl.case op with one region per
  /// case label plus an else region.
  mlir::LogicalResult mlirGen(const CaseOfAST *caseOf) {
    auto location = loc(caseOf->loc());
    mlir::Value selector = mlirGenRValue(caseOf->getExpr());
    if (!selector)
      return mlir::failure();
    if (selector.getType().isa<RealType>())
      return mlirGenRealCase(location, caseOf, selector);
    if (!selector.getType().isa<IntegerType>() &&
        !selector.getType().isa<LogicalType>()) {
      emitError(location) << "case selector must be an integer, bool or real";
      return mlir::failure();
    }

    // Group the flat instruction list by case labels.
    struct CaseGroup {
      std::vector<int32_t> values;
      std::vector<const InstructionAST *> body;
    };
    std::vector<CaseGroup> groups;
    for (const auto &instr : caseOf->getCode()->getInstructions()) {
      if (auto *label = llvm::dyn_cast<CaseLabelAST>(instr.get())) {
        CaseGroup group;
        for (const auto &value : label->getValues()) {
          int v;
          if (failed(getConstantInteger(value.get(), v)))
            return mlir::failure();
          group.values.push_back(v);
        }
        groups.push_back(std::move(group));
      } else {
        if (groups.empty()) {
          emitError(loc(instr->loc()))
              << "instruction before the first case label";
          return mlir::failure();
        }
        groups.back().body.push_back(instr.get());
      }
    }

    SmallVector<mlir::Attribute, 4> caseValues;
    for (auto &group : groups)
      caseValues.push_back(
          mlir::DenseI32ArrayAttr::get(builder.getContext(), group.values));
    auto caseValuesAttr = mlir::ArrayAttr::get(builder.getContext(), caseValues);

    auto caseOp =
        builder.create<CaseOp>(location, selector, caseValuesAttr,
                              (unsigned)groups.size() + 1);
    auto old = builder.saveInsertionPoint();

    for (size_t i = 0; i < groups.size(); ++i) {
      auto &group = groups[i];
      builder.createBlock(&caseOp.getBody()[i]);
      builder.setInsertionPointToStart(&caseOp.getBody()[i].front());
      for (auto *instr : group.body) {
        if (failed(mlirGen(instr)))
          return mlir::failure();
      }
      createEndOp(location);
    }

    // The last region is the else block.
    builder.createBlock(&caseOp.getBody().back());
    builder.setInsertionPointToStart(&caseOp.getBody().back().front());
    (void)0;
    if (caseOf->getElseBlock().has_value()) {
      if (failed(mlirGen(caseOf->getElseBlock().value())))
        return mlir::failure();
    }
    createEndOp(location);

    builder.restoreInsertionPoint(old);
    return mlir::success();
  }

  /// Codegen a jump label: split the current block, branch to the label block.
  mlir::LogicalResult mlirGen(const JumpLabelAST *label) {
    auto location = loc(label->loc());
    auto name = label->getLabel();
    mlir::Block *cur = builder.getInsertionBlock();
    mlir::Block *labelBlock;
    auto it = labelBlocks.find(name);
    if (it != labelBlocks.end()) {
      labelBlock = it->second;
    } else {
      labelBlock = builder.createBlock(cur->getParent());
      labelBlocks[name] = labelBlock;
    }
    if (!blockHasTerminator()) {
      builder.setInsertionPointToEnd(cur);
      builder.create<mlir::cf::BranchOp>(location, labelBlock);
    }
    builder.setInsertionPointToStart(labelBlock);
    return mlir::success();
  }

  /// Codegen a goto: branch to the label block (created on demand for
  /// forward references) and continue in a fresh unreachable block.
  mlir::LogicalResult mlirGen(const GotoAST *goto_) {
    auto location = loc(goto_->loc());
    auto name = goto_->getLabel();
    mlir::Block *cur = builder.getInsertionBlock();
    mlir::Block *target;
    auto it = labelBlocks.find(name);
    if (it != labelBlocks.end()) {
      target = it->second;
    } else {
      target = builder.createBlock(cur->getParent());
      labelBlocks[name] = target;
    }
    if (blockHasTerminator()) {
      emitError(location) << "goto must not follow a terminator";
      return mlir::failure();
    }
    builder.setInsertionPointToEnd(cur);
    builder.create<mlir::cf::BranchOp>(location, target);
    mlir::Block *dead = builder.createBlock(cur->getParent());
    builder.setInsertionPointToStart(dead);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const DebugPrintAST *debugPrint) {
    auto location = loc(debugPrint->loc());

    builder.create<DebugPrintOp>(location, debugPrint->getMsg());
    return mlir::success();
  }

};

} // namespace

namespace sclang {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(&moduleAST);
}

} // namespace sclang
