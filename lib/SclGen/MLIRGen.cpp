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
#include "sclang/AwlDialect/AwlOps.h"
#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclGen/AST.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
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

  /// Set when any error diagnostic was emitted. Lets mlirGen fail cleanly
  /// (return nullptr) instead of crashing on null types/values when
  /// assertions are disabled. Member overloads shadow mlir::emitError for
  /// unqualified calls inside this class.
  bool errorOccurred = false;
  mlir::InFlightDiagnostic emitError(mlir::Location location) {
    errorOccurred = true;
    return mlir::emitError(location);
  }
  mlir::InFlightDiagnostic emitError(mlir::Location location,
                                     const llvm::Twine &message) {
    errorOccurred = true;
    return mlir::emitError(location, message);
  }

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
                        callableVarTypes[fn->getIdentifier()],
                        &callableVarNames[fn->getIdentifier()]);
      } else if (auto *ob = llvm::dyn_cast<OrganizationBlockAST>(unit.get())) {
        functionResultTypes[ob->getIdentifier()] = builder.getNoneType();
        collectVarTypes(ob->getDeclarations(),
                        callableVarTypes[ob->getIdentifier()],
                        &callableVarNames[ob->getIdentifier()]);
      } else if (auto *fb = llvm::dyn_cast<FunctionBlockAST>(unit.get())) {
        collectVarTypes(fb->getDeclarations(),
                        callableVarTypes[fb->getIdentifier()],
                        &callableVarNames[fb->getIdentifier()]);
      }
    }
    if (errorOccurred)
      return nullptr;

    for (const auto &unit : *moduleAST) {
      builder.clearInsertionPoint();
      if (auto *udt = llvm::dyn_cast<UserDefinedTypeAST>(unit.get())) {
        if (failed(mlirGen(udt)) || errorOccurred)
          return nullptr;
        continue;
      }
      auto func = mlirGen(unit.get());
      if (!func || errorOccurred)
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
  /// For `base.%Xn` bit accesses: bit number per byte address.
  llvm::DenseMap<mlir::Value, int> bitViewBits;
  /// Function/OB name -> result type, collected in a pre-pass over the
  /// module so calls can be typed even before the callee is generated.
  llvm::StringMap<mlir::Type> functionResultTypes;
  /// Callable name (function/OB/FB) -> variable name -> declared type,
  /// collected in a pre-pass so call arguments can be typed (and string
  /// literals materialized) even before the callee is generated.
  llvm::StringMap<llvm::StringMap<mlir::Type>> callableVarTypes;
  /// Callable name -> declared parameter names IN DECLARATION ORDER (for
  /// positional call arguments: the i-th positional argument gets the i-th
  /// parameter's name, so the canonicalizer can reorder/fill defaults).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> callableVarNames;

  /// Collect the declared types of all variables of a declaration section
  /// into `map`, keyed by variable name.
  void collectVarTypes(const DeclarationSectionAST *declarations,
                       llvm::StringMap<mlir::Type> &map) {
    collectVarTypes(declarations, map, nullptr);
  }
  void collectVarTypes(const DeclarationSectionAST *declarations,
                       llvm::StringMap<mlir::Type> &map,
                       llvm::SmallVector<std::string, 8> *order) {
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
        for (const auto &var : value->getVars()) {
          map[var->getIdentifier()] = type;
          if (order)
            order->push_back(std::string(var->getIdentifier()));
        }
      }
    }
  }
  /// Names of all function blocks in the module (for type resolution).
  llvm::StringMap<bool> fbNames;
  // strip the surrounding quotes of a quoted TIA symbol name ("TON" -> TON)
  /// Normalized (uppercase, unquoted, _TIME-stripped) name if `name` is a
  /// builtin standard FB (TON/TOF/TP, CTU/CTD/CTUD, R_TRIG/F_TRIG/SR/RS);
  /// empty otherwise. These lower to runtime calls, not FunctionBlockOps.
  static std::string builtinStandardFb(llvm::StringRef name) {
    std::string up = llvm::StringRef(unquote(name).c_str()).upper();
    if (llvm::StringRef(up).ends_with("_TIME"))
      up = up.substr(0, up.size() - 5);
    static const std::set<std::string> names = {
        "TON", "TOF", "TP", "CTU", "CTD", "CTUD",
        "R_TRIG", "F_TRIG", "SR", "RS"};
    return names.count(up) ? up : std::string();
  }

  static std::string unquote(llvm::StringRef s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
      return s.substr(1, s.size() - 2).str();
    return s.str();
  }
  // owns synthesized AST nodes (e.g. default constants for initializer-less
  // VAR CONSTANT entries)
  llvm::SmallVector<std::unique_ptr<ExpressionAST>, 8> astPool;
  // FB names for which we created an empty stub (TIA library FBs): member
  // access on them is approximated by a temp cell instead of an error
  std::set<std::string> stubFbNames;

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
    mlir::Value value;
    mlir::Type type;
    if (llvm::isa<StringConstantAST>(constValue)) {
      // a string literal is not a standalone SSA value; register the AST so
      // comparisons/assignments can read it via getConstValue()
      auto i8 = IntegerType::get(builder.getContext(), 8);
      type = ArrayType::get({{0, 253}}, i8); // String[254]
    } else {
      value = mlirGen(constValue);
      if (!value)
        return mlir::failure();
      type = value.getType();
    }

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
      if (!type)
        return mlir::failure();
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
        auto varOp =
            builder.create<VariableOp>(location, type, isInput, isOutput, name,
                                       init);
        // SPEC[SCL] IEC 61131-3 ST VAR string initializer (VAR s : STRING :=
        // 'abc'): recorded as an attribute so the cold-start value is baked
        // into the instance DB global (DataBlockOpLowering), like integer
        // initializers via initBits.
        if (decl->getInitializer().has_value())
          if (auto *sc = llvm::dyn_cast<StringConstantAST>(
                  decl->getInitializer().value()))
            varOp->setAttr("scl.init_str", builder.getStringAttr(sc->getValue()));
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGenTempVar(const VariableDeclarationSubsectionAST *decls) {
    for (const auto &decl : decls->getValues()) {
      auto type = getType(decl->getDataType());
      if (!type)
        return mlir::failure();
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

  /// TIA `VAR CONSTANT` subsection: register each declared name as a
  /// constant symbol with its initializer as the constant value.
  mlir::LogicalResult mlirGenConstantVars(
      const VariableDeclarationSubsectionAST *decls) {
    for (const auto &decl : decls->getValues()) {
      auto init = decl->getInitializer();
      if (!init) {
        // TIA allows VAR CONSTANT without an initializer (e.g. EMPTYSTRING):
        // default to an empty string for string types, zero otherwise
        auto type = getType(decl->getDataType());
        if (!type)
          return mlir::failure();
        emitWarning(loc(decl->loc()))
            << "constant declaration without initializer, using default";
        if (type.isa<ArrayType>())
          astPool.push_back(std::make_unique<StringConstantAST>(
              decl->loc(), "", tok_string_literal));
        else
          astPool.push_back(
              std::make_unique<IntegerConstantAST>(decl->loc(), 0, tok_int));
        init = astPool.back().get();
      }
      for (const auto &var : decl->getVars()) {
        auto location = loc(var->loc());
        if (failed(declare(location, var->getIdentifier(), *init)))
          return mlir::failure();
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
        if (decl->isConstant())
          return mlirGenConstantVars(decl);
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

  // SPEC[SCL] IEC 61131-3 ST organization block (OB) -> void function with scan-cycle hooks (ADR 002)
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
          // FC outputs are not propagated to the caller yet (no `=>`
          // write-back): keep them as function-local statics so the block
          // compiles. FB outputs are IDB members and unaffected.
          tempvar.push_back(&vardecls);
          break;
        case VariableDeclarationSubsectionAST::VarInOut:
          // In/out passed as input; write-back to the caller is a
          // documented gap (see AGENTS.md).
          addVariables(vardecls, input_types, input_names);
          break;
        case VariableDeclarationSubsectionAST::Var:
        case VariableDeclarationSubsectionAST::VarTemp:
          tempvar.push_back(&vardecls);
          break;
        }
      }
    }
    // Create an MLIR function. SCL projects may contain forward
    // declarations (FUNCTION x ... BEGIN END_FUNCTION in a separate file)
    // that are later defined with a real body: if a FunctionOp with this
    // name already exists, the new definition replaces it (and a forward
    // declaration arriving after the definition is skipped).
    auto func_type = builder.getFunctionType(input_types, output_types);
    if (auto prev = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionOp>(
            theModule, builder.getStringAttr(name))) {
      bool newIsFwdDecl = code->getInstructions().empty();
      if (!newIsFwdDecl)
        prev.erase(); // real definition replaces the forward declaration
      else
        return prev; // forward declaration after the definition: keep it
    }
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
    mlir::Type type;
    if (db->getType()) {
      type = getType(db->getType());
    } else if (auto *decls = db->getDeclarations()) {
      // TIA non-optimized DB: the type is the implicit struct of the
      // declared members.
      llvm::SmallVector<mlir::StringAttr, 4> names;
      llvm::SmallVector<mlir::Type, 4> types;
      for (auto &sub : decls->getDecls()) {
        if (sub->getKind() !=
            DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable)
          continue;
        auto &vardecls = llvm::cast<VariableDeclarationSubsectionAST>(*sub);
        for (auto &value : vardecls.getValues()) {
          auto t = getType(value->getDataType());
          if (!t)
            return nullptr;
          for (auto &var : value->getVars())
            names.push_back(builder.getStringAttr(var->getIdentifier()));
          for (auto &var : value->getVars()) {
            (void)var;
            types.push_back(t);
          }
        }
      }
      type = StructType::get(names, types);
    }
    if (!type) return nullptr;
    mlir::TypeAttr typeAttr = mlir::TypeAttr::get(type);
    mlir::StringAttr nameAttr = mlir::StringAttr::get(context, name);

    // DB begin-section assignments `member := <const>` become the `init`
    // dict attr (member name -> i64 bit pattern), baked into the IDB
    // global's cold start by DataBlockOpLowering.
    mlir::DictionaryAttr initAttr;
    if (auto *code = db->getAssignments()) {
      SmallVector<mlir::NamedAttribute, 4> entries;
      for (auto &instr : code->getInstructions()) {
        auto *assign = llvm::dyn_cast<ValueAssignmentAST>(instr.get());
        if (!assign)
          continue;
        auto *bin = llvm::dyn_cast<BinaryExpressionAST>(assign->getExpression());
        if (!bin || bin->getOp() != tok_assignment)
          continue;
        auto *lhs = llvm::dyn_cast<SimpleVariableAST>(bin->getLhs());
        if (!lhs || lhs->isSymbol()) {
          // not a plain start-value assignment (e.g. member/array init in a
          // DB code section): not executed by the sim, skip with a warning
          emitWarning(loc(bin->getLhs()->loc()))
              << "non-trivial DB initializer in '" << name << "' is ignored";
          continue;
        }
        int64_t value = 0;
        if (mlir::failed(foldConstIntAST(bin->getRhs(), value))) {
          emitWarning(loc(bin->getRhs()->loc()))
              << "non-constant DB initializer for '" << lhs->getName()
              << "' in DB '" << name << "' is ignored";
          continue;
        }
        entries.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(context, lhs->getName()),
            builder.getIntegerAttr(builder.getIntegerType(64), value)));
      }
      if (!entries.empty())
        initAttr = mlir::DictionaryAttr::get(context, entries);
    }

    return builder.create<DataBlockOp>(location, typeAttr, nameAttr,
                                       (mlir::Attribute)initAttr);
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

    // SPEC[SCL] IEC 61131-3 ST statement list: assignment, call, IF/ELSIF/ELSE,
    // FOR/WHILE/REPEAT-UNTIL, CASE, GOTO/labels, RETURN/EXIT/CONTINUE
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
  // A string literal, possibly behind a constant symbol (#EMPTYSTRING).
  const StringConstantAST *asStringLit(const ExpressionAST *e) {
    if (auto *s = llvm::dyn_cast<StringConstantAST>(e))
      return s;
    if (auto *v = llvm::dyn_cast<SimpleVariableAST>(e)) {
      auto sym = symbolTable.lookup(v->getName());
      if (sym.isConst())
        return llvm::dyn_cast<StringConstantAST>(sym.getConstValue());
    }
    return nullptr;
  }

  mlir::Value mlirGenStringCompare(const BinaryExpressionAST *expr) {
    auto location = loc(expr->loc());
    auto op = expr->getOp();
    ArrayType lhsArr = arrayTypeOfAST(expr->getLhs());
    ArrayType rhsArr = arrayTypeOfAST(expr->getRhs());
    const StringConstantAST *lhsLit = asStringLit(expr->getLhs());
    const StringConstantAST *rhsLit = asStringLit(expr->getRhs());
    if (!lhsArr && !rhsArr && !lhsLit && !rhsLit)
      return nullptr;
    // member access (db.member) has no AST-level array type: take it from
    // the addressed value instead
    mlir::Value lhsAddr0 = lhsLit ? nullptr : mlirGen(expr->getLhs());
    mlir::Value rhsAddr0 = rhsLit ? nullptr : mlirGen(expr->getRhs());
    if (!lhsLit && !lhsAddr0)
      return nullptr;
    if (!rhsLit && !rhsAddr0)
      return nullptr;
    if (!lhsArr && lhsAddr0)
      lhsArr = arrayTypeOfAddress(lhsAddr0);
    if (!rhsArr && rhsAddr0)
      rhsArr = arrayTypeOfAddress(rhsAddr0);
    ArrayType arrTy = lhsArr ? lhsArr : rhsArr;
    if (!arrTy) {
      // a single-char literal against a char scalar: not a string compare,
      // fall through to the normal path (char constant)
      const StringConstantAST *lit = lhsLit ? lhsLit : rhsLit;
      if (lit && lit->getValue().size() <= 1)
        return nullptr;
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
    mlir::Value lhsAddr = lhsAddr0;
    mlir::Value rhsAddr = rhsAddr0;
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
      // unknown callee (e.g. a library string builtin): use String[254]
      auto i8 = IntegerType::get(builder.getContext(), 8);
      ArrayType arrTy = ArrayType::get({{0, 253}}, i8);
      auto addrTy = AddressType::get(arrTy);
      auto storage = builder.create<TempVariableOp>(
          location, addrTy, builder.getStringAttr("strlit"));
      if (failed(mlirGenStoreStringBytes(location, storage, arrTy,
                                         str->getValue())))
        return nullptr;
      return builder.create<LoadOp>(location, arrTy, storage);
    }
    auto it = paramTypes->find(name);
    if (it == paramTypes->end())
      for (auto &kv : *paramTypes)
        if (kv.getKey().equals_insensitive(name)) {
          it = paramTypes->find(kv.getKey());
          break;
        }
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
  // SPEC[SCL] IEC 61131-3 ST assignment (:=), incl. string/array copies -> see mlirGenStringAssignment and the array-copy path below
  mlir::LogicalResult mlirGen(const ValueAssignmentAST *instr) {
    auto location = loc(instr->loc());

    const auto expr = llvm::cast<BinaryExpressionAST>(instr->getExpression());
    assert(expr->getOp() == tok_assignment);

    if (auto *str = llvm::dyn_cast<StringConstantAST>(expr->getRhs()))
      return mlirGenStringAssignment(location, expr->getLhs(), str);
    // `s := #CONST` with a string constant symbol
    if (auto *v = llvm::dyn_cast<SimpleVariableAST>(expr->getRhs())) {
      auto sym = symbolTable.lookup(v->getName());
      if (sym.isConst())
        if (auto *str = llvm::dyn_cast<StringConstantAST>(sym.getConstValue()))
          return mlirGenStringAssignment(location, expr->getLhs(), str);
    }

    mlir::Value lhs = mlirGenLValue(expr->getLhs());
    if (!lhs)
      return mlir::failure();

    // Whole-array (incl. string) var-to-var assignment `a := b`: copy the
    // elements one by one (the buffers are flat byte/element buffers).
    if (ArrayType lhsArr = arrayTypeOfAddress(lhs)) {
      mlir::Value rhsAddr = mlirGen(expr->getRhs());
      ArrayType rhsArr = arrayTypeOfAST(expr->getRhs());
      // the rhs may also be an address-valued expression (member access, or
      // a string builtin call returning a temp buffer address)
      if (!rhsArr && rhsAddr)
        rhsArr = arrayTypeOfAddress(rhsAddr);
      if (!rhsArr) {
        // non-array rhs into an array/string lhs (e.g. a scalar): zero-fill,
        // with a warning
        emitWarning(location) << "assigning a non-string value to a string, "
                                 "storing an empty string";
        if (failed(mlirGenStoreStringBytes(location, lhs, lhsArr, "")))
          return mlir::failure();
        return mlir::success();
      }
      // char arrays (strings) of different declared sizes: copy the common
      // prefix and zero-fill the rest (string assignment truncates/pads)
      bool lhsIsChar = lhsArr.getElementType().isa<IntegerType>() &&
                       lhsArr.getElementType().cast<IntegerType>().getWidth() == 8;
      if (rhsArr.getElementType() == lhsArr.getElementType() &&
          (lhsArr.getDimensions() == rhsArr.getDimensions() || lhsIsChar)) {
        if (!rhsAddr)
          return mlir::failure();
        auto dims = lhsArr.getDimensions();
        int64_t count = dims[0].second - dims[0].first + 1;
        auto rdims = rhsArr.getDimensions();
        int64_t srcCount = rdims[0].second - rdims[0].first + 1;
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
          srcCount *= elemBytes;
        }
        mlir::TypeAttr attr = mlir::TypeAttr::get(copyTy);
        int64_t ncopy = std::min(count, srcCount);
        for (int64_t i = 0; i < ncopy; ++i) {
          mlir::Value dst = arrayElemAddr(location, lhs, copyTy, i, attr);
          mlir::Value src = arrayElemAddr(location, rhsAddr, copyTy, i, attr);
          mlir::Value v = builder.create<LoadOp>(location, elemTy, src);
          builder.create<StoreOp>(location, dst, v);
        }
        // zero-fill the rest of the lhs (string truncation/padding)
        for (int64_t i = ncopy; i < count; ++i) {
          mlir::Value dst = arrayElemAddr(location, lhs, copyTy, i, attr);
          builder.create<StoreOp>(
              location, dst,
              builder.create<ConstantOp>(
                  location, elemTy,
                  builder.getIntegerAttr(
                      builder.getIntegerType(
                          elemTy.isa<IntegerType>()
                              ? elemTy.cast<IntegerType>().getWidth()
                              : 8),
                      0)));
        }
        return mlir::success();
      }
    }

    mlir::Value rhs = mlirGenRValue(expr->getRhs());
    if (!rhs)
      return mlir::failure();

    // `base.%Xn := v`: read-modify-write of a single bit in the base byte.
    if (auto *bin = llvm::dyn_cast<BinaryExpressionAST>(expr->getLhs())) {
      if (bin->getOp() == tok_dot) {
        if (auto *dir = llvm::dyn_cast<DirectVariableAST>(bin->getRhs())) {
          if (std::string(dir->getArea()) == "X") {
            auto *ctx = builder.getContext();
            auto byteTy = LogicalType::get(ctx, 8);
            auto i8 = IntegerType::get(ctx, 8);
            int bit = dir->getBit();
            mlir::Value bitVal = rhs;
            if (bitVal.getType() != i8) {
              auto cast = builder.create<IntegerCastOp>(location, i8, bitVal);
              if (bitVal.getType().isa<LogicalType>())
                cast.setZextAttr(builder.getUnitAttr());
              bitVal = cast.getResult();
            }
            auto scaleC = builder.create<ConstantOp>(
                location, i8, builder.getIntegerAttr(i8, (int64_t)(1 << bit)));
            auto set = builder.create<MulOp>(location, bitVal, scaleC.getResult());
            auto byteVal = builder.create<LoadOp>(location, byteTy, lhs).getResult();
            auto byteI8 = builder.create<IntegerCastOp>(location, i8, byteVal);
            byteI8.setZextAttr(builder.getUnitAttr());
            auto maskC = builder.create<ConstantOp>(
                location, i8,
                builder.getIntegerAttr(i8, (int64_t)(uint8_t)~(1 << bit)));
            auto cleared = builder.create<AndOp>(location, byteI8.getResult(), maskC.getResult());
            auto newByte = builder.create<OrOp>(location, cleared.getResult(), set.getResult());
            builder.create<StoreOp>(location, lhs, newByte.getResult());
            return mlir::success();
          }
        }
      }
    }

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
    if (dw > 0 && rhs.getType().isa<ArrayType>()) {
      // assigning a string/array value to a scalar (e.g. a string builtin
      // stub result): store zero, with a warning
      emitWarning(location) << "assigning a string value to a scalar, storing 0";
      rhs = builder.create<ConstantOp>(
          location, declared.isa<IntegerType>()
                        ? declared
                        : IntegerType::get(builder.getContext(), dw),
          builder.getIntegerAttr(
              builder.getIntegerType(declared.isa<IntegerType>()
                                         ? declared.cast<IntegerType>().getWidth()
                                         : dw),
              0));
    } else if (dw > 0 && rw > 0 && dw != rw && rhs.getType().isa<IntegerType>()) {
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
    if (!address)
      return nullptr;

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
        auto location = loc(expr->loc());
        auto addr = mlirGen(expr);
        if (!addr)
          return nullptr;
        if (auto *dir = llvm::dyn_cast<DirectVariableAST>(bin->getRhs())) {
          // relative access base.%Xn: load the byte and extract the bit
          // (byte / 2^n) % 2 as a byte-typed 0/1 value
          if (std::string(dir->getArea()) == "X") {
            auto byteTy = LogicalType::get(builder.getContext(), 8);
            auto i8 = IntegerType::get(builder.getContext(), 8);
            auto byteVal = builder.create<LoadOp>(location, byteTy, addr).getResult();
            auto divC = builder.create<ConstantOp>(
                location, byteTy,
                builder.getIntegerAttr(i8, (int64_t)(1 << dir->getBit())));
            auto shifted = builder.create<DivOp>(location, byteVal, divC.getResult());
            auto modC = builder.create<ConstantOp>(
                location, byteTy, builder.getIntegerAttr(i8, (int64_t)2));
            return builder.create<ModOp>(location, shifted.getResult(), modC.getResult()).getResult();
          }
          // base.%Bn/.%Wn/.%Dn: load with the recorded access width
          auto it = byteViewTypes.find(addr);
          if (it == byteViewTypes.end())
            return nullptr;
          return builder.create<LoadOp>(location, it->second, addr);
        }
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
        return builder.create<LoadOp>(location, ty, addr);
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
    // The literal's declared/typed token decides its type (IEC 61131-3:
    // unsuffixed literals take the smallest type that fits, see
    // ParseIntegerLiteralExpression); default is int (i16).
    mlir::Type type = getType(tok_int);
    unsigned width = 16;
    switch (expr->getType()) {
    case tok_bool:
      type = getType(tok_bool);
      width = 1;
      break;
    case tok_byte:
    case tok_usint:
      type = getType(tok_byte);
      width = 8;
      break;
    case tok_char:
      type = getType(tok_char);
      width = 8;
      break;
    case tok_word:
    case tok_uint:
      type = getType(tok_word);
      width = 16;
      break;
    case tok_dword:
    case tok_udint:
      type = getType(tok_dword);
      width = 32;
      break;
    case tok_dint:
      type = getType(tok_dint);
      width = 32;
      break;
    default:
      break;
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
    // A single-character string literal used as a value is a char constant
    // (e.g. comparing a CHAR with 'x').
    if (expr->getValue().size() == 1) {
      auto i8 = IntegerType::get(builder.getContext(), 8);
      return builder.create<ConstantOp>(
          loc(expr->loc()), i8,
          builder.getIntegerAttr(i8, (uint8_t)expr->getValue()[0]));
    }
    // Multi-char literal in a plain value position (e.g. via a constant
    // symbol): materialize a temp String[254] and return the array value.
    auto location = loc(expr->loc());
    auto i8 = IntegerType::get(builder.getContext(), 8);
    ArrayType arrTy = ArrayType::get({{0, 253}}, i8);
    auto storage = builder.create<TempVariableOp>(
        location, AddressType::get(arrTy), builder.getStringAttr("strlit"));
    if (failed(mlirGenStoreStringBytes(location, storage, arrTy,
                                       expr->getValue())))
      return nullptr;
    return builder.create<LoadOp>(location, arrTy, storage);
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

  /// Fold a constant symbol's initializer into a value (rvalue position).
  mlir::Value mlirGenConstSymbol(mlir::Location location, StringRef name,
                                 const VariableSymbol &variable) {
    const ExpressionAST *ce = variable.getConstValue();
    mlir::Type resTy = variable.getType() ? variable.getType() : getType(tok_int);
    unsigned width = 16;
    if (auto it = resTy.dyn_cast<IntegerType>())
      width = it.getWidth();
    else if (auto lt = resTy.dyn_cast<LogicalType>())
      width = lt.getWidth();
    if (auto *ic = llvm::dyn_cast<IntegerConstantAST>(ce))
      return builder.create<ConstantOp>(
          location, resTy,
          builder.getIntegerAttr(builder.getIntegerType(width), ic->getValue()));
    if (llvm::isa<RealConstantAST>(ce) || llvm::isa<TimeConstantAST>(ce))
      return mlirGen(ce);
    if (llvm::isa<StringConstantAST>(ce)) {
      emitWarning(location) << "string constant '" << name
                            << "' used as a value, assuming 0";
      return builder.create<ConstantOp>(
          location, getType(tok_int), builder.getIntegerAttr(builder.getIntegerType(16), 0));
    }
    // constant expression (e.g. A + B): fold to an integer
    int64_t v = 0;
    if (mlir::failed(foldConstIntAST(ce, v)))
      v = 0;
    return builder.create<ConstantOp>(
        location, resTy, builder.getIntegerAttr(builder.getIntegerType(width), v));
  }

  mlir::Value mlirGenVariable(mlir::Location location, StringRef name) {
    auto variable = symbolTable.lookup(name);
    if (variable.isConst())
      return mlirGenConstSymbol(location, name, variable);
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
    // A global data block referenced without quotes, e.g. `MyDB.member`.
    if (DataBlockOp db = mlir::SymbolTable::lookupNearestSymbolFrom<DataBlockOp>(
            theModule, builder.getStringAttr(name))) {
      mlir::Type address = AddressType::get(db.getType());
      return builder.create<GetGlobalOp>(location, address, name);
    }
    emitError(location) << "unknown variable '" << name << "'";
    return nullptr;
  }

  mlir::Value mlirGenSymbol(mlir::Location location, StringRef name) {
    auto sym0 = symbolTable.lookup(name);
    if (sym0.isConst())
      return mlirGenConstSymbol(location, name, sym0);
    auto nameAttr = builder.getStringAttr(name);
    DataBlockOp db =
        mlir::SymbolTable::lookupNearestSymbolFrom<DataBlockOp>(theModule, nameAttr);
    if (!db) {
      emitWarning(location) << "unknown symbol '" << name << "', assuming 1";
      return builder.create<ConstantOp>(
          location, getType(tok_int),
          builder.getIntegerAttr(builder.getIntegerType(16), 1));
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
      auto var = lookupVariableCI(fb, name);
      return var ? var.getType() : nullptr;
    }
    if (auto structTy = baseTy.dyn_cast<StructType>())
      return structTy.getElementType(name);
    return nullptr;
  }

  /// Relative access into a composite variable: `base.%B<n>` / `.%W<n>` /
  /// `.%D<n>` select a byte/word/dword at absolute byte offset <n> within
  /// the base (returned as an i8 view address with the access width recorded
  /// in byteViewTypes); `base.%X<n>` selects bit <n> of the base byte (the
  /// byte address is returned and the bit number recorded in bitViewBits).
  mlir::Value mlirGenRelativeAccess(mlir::Location location, mlir::Value base,
                                    const DirectVariableAST *dir) {
    if (!base)
      return nullptr;
    auto *ctx = builder.getContext();
    auto area = std::string(dir->getArea());
    auto i8 = IntegerType::get(ctx, 8);
    // byte-offset address via an i8-array view with a constant index
    // (element size 1, so the index is the byte offset)
    mlir::Value addr = base;
    if (dir->getByte() != 0) {
      auto intTy = getType(tok_int);
      auto i16 = builder.getIntegerType(16);
      auto idx = builder.create<ConstantOp>(
          location, intTy, builder.getIntegerAttr(i16, dir->getByte()));
      auto arrTy = ArrayType::get({{0, 0}}, i8);
      addr = builder.create<GetIndexOp>(location, AddressType::get(i8), base,
                                        mlir::ValueRange{idx},
                                        mlir::TypeAttr::get(arrTy));
    }
    if (area == "X") {
      bitViewBits[addr] = dir->getBit();
      byteViewTypes[addr] = LogicalType::get(ctx, 8);
      return addr;
    }
    // B/W/D: record the access width on the (possibly offset) byte address
    byteViewTypes[addr] = LogicalType::get(ctx, dir->getWidth());
    return addr;
  }

  mlir::Value mlirGenElement(mlir::Location location,
                             mlir::Value address, StringRef name) {
    mlir::Type baseType = address.getType().cast<AddressType>().getElementType();
    auto i8 = IntegerType::get(builder.getContext(), 8);
    mlir::Type resultType = AddressType::get(i8);

    // Member access into an instance DB (FB variable), possibly reached
    // through a byte view (FB-typed member of another FB/struct).
    auto genIdbMember = [&](InstanceDbType idbType) -> mlir::Value {
      auto fbName = builder.getStringAttr(idbType.getFbSymbol());
      // Builtin standard FBs (TON/TOF/TP, CTU/CTD/CTUD, R_TRIG/F_TRIG/SR/RS)
      // are recognized by name first — the frontend may have emitted an
      // empty stub FunctionBlockOp for them, which must not shadow the
      // builtin. Their fixed layouts are known to the lowering
      // (FbLayoutCache), so address members via scl.getelement with the fb
      // attr (a GetVariableOp on the byte view cannot be lowered).
      std::string fbUp = builtinStandardFb(fbName.getValue());
      auto fb = fbUp.empty()
                    ? mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
                          theModule, fbName)
                    : nullptr;
      if (!fbUp.empty()) {
        std::string mUp = std::string(llvm::StringRef(name).upper());
        mlir::Type t;
        if (mUp == "IN" || mUp == "Q" || mUp == "Q1" || mUp == "CU" ||
            mUp == "CD" || mUp == "R" || mUp == "R1" || mUp == "LD" ||
            mUp == "S" || mUp == "S1" || mUp == "CLK" ||
            mUp == "QU" || mUp == "QD")
          t = getType(tok_bool);
        else if (mUp == "PV" || mUp == "CV")
          t = getType(tok_int);
        else if (mUp == "PT" || mUp == "ET")
          t = getType(tok_time);
        if (!t) {
          emitError(location) << "unknown variable '" << name
                              << "' in builtin FB '" << fbName << "'";
          return nullptr;
        }
        auto addrElem = address.getType().cast<AddressType>().getElementType();
        mlir::Value result;
        if (addrElem == idbType) {
          // Direct instance-DB address: GetVariableOp can be lowered.
          result = builder.create<GetVariableOp>(location, resultType, address,
                                                 name);
        } else {
          result = builder.create<GetElementOp>(
              location, resultType, address, name, mlir::TypeAttr(),
              mlir::FlatSymbolRefAttr::get(fbName));
        }
        byteViewTypes[result] = t;
        return result;
      }
      if (!fb) {
        // Unknown FB (TIA library FB): approximate the member with a fresh
        // temp cell so parameter stores/reads still type-check.
        emitWarning(location) << "unknown FB '" << fbName << "', member '"
                              << name << "' approximated by a temp cell";
        auto cell = builder.create<TempVariableOp>(
            location, resultType, builder.getStringAttr(name));
        byteViewTypes[cell] = ArrayType::get({{0, 255}}, i8);
        return cell;
      }
      auto var = lookupVariableCI(fb, name);
      if (!var && stubFbNames.count(fbName.getValue().str())) {
        emitWarning(location) << "member '" << name << "' of stub FB '"
                              << fbName << "' approximated by a temp cell";
        auto cell = builder.create<TempVariableOp>(
            location, resultType, builder.getStringAttr(name));
        byteViewTypes[cell] = ArrayType::get({{0, 255}}, i8);
        return cell;
      }
      if (!var) {
        emitError(location) << "unknown variable '" << name << "' in FB '" << fbName << "'";
        return nullptr;
      }
      // Address the member as an i8 view into the instance byte buffer.
      auto addrElem = address.getType().cast<AddressType>().getElementType();
      if (addrElem == idbType) {
        // Direct access through the FB's own instance-DB argument ($self).
        auto result = builder.create<GetVariableOp>(location, resultType, address, name);
        byteViewTypes[result] = var.getType();
        return result;
      }
      // The instance is reached through a byte view (e.g. a local FB-typed
      // member `g.y` where g is a member of the enclosing FB): address it as
      // a struct element so GetElementOpLowering can compute the offset from
      // the FB's member layout (a GetVariableOp on a byte view has no IDB
      // type and cannot be lowered).
      SmallVector<mlir::StringAttr, 8> names;
      SmallVector<mlir::Type, 8> types;
      for (auto v : fb.getBody().getOps<VariableOp>()) {
        names.push_back(builder.getStringAttr(v.getSymName()));
        types.push_back(v.getType());
      }
      auto st = StructType::get(names, types);
      auto result = builder.create<GetElementOp>(
          location, resultType, address, var.getSymName(),
          mlir::TypeAttr::get(st), mlir::FlatSymbolRefAttr::get(fbName));
      byteViewTypes[result] = var.getType();
      return result;
    };

    return mlir::TypeSwitch<mlir::Type, mlir::Value>(baseType)
    .Case<InstanceDbType>([&](auto idbType) -> mlir::Value {
      return genIdbMember(idbType);
    })
    .Case<StructType>([&](auto structType) -> mlir::Value {
      auto memberTy = structType.getElementType(name);
      if (!memberTy) {
        emitError(location) << "unknown element '" << name << "' in struct";
        return nullptr;
      }
      auto result = builder.create<GetElementOp>(location, resultType, address,
                                                 name, mlir::TypeAttr(),
                                                 mlir::FlatSymbolRefAttr());
      byteViewTypes[result] = memberTy;
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
      if (it == byteViewTypes.end()) {
        emitError(location) << "unsupported base type " << baseType
                            << " for element '" << name << "'";
        return nullptr;
      }
      // FB-typed member (e.g. a TON_TIME timer inside an FB): the declared
      // instance-DB type is tracked in byteViewTypes.
      if (auto idbTy = it->second.dyn_cast<InstanceDbType>())
        return genIdbMember(idbTy);
      if (!it->second.isa<StructType>()) {
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
                                                 name, baseAttr,
                                                 mlir::FlatSymbolRefAttr());
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
    auto intTy = getType(tok_int);
    for (const auto & index : expr->getIndices()) {
      mlir::Value idx = mlirGenRValue(index.get());
      if (idx && idx.getType() != intTy)
        idx = builder.create<DialectCastOp>(location, intTy, idx);
      indices.push_back(idx);
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
      if (auto *dir = llvm::dyn_cast<DirectVariableAST>(expr->getRhs())) {
        // relative access: base.%Bn/.%Wn/.%Dn (byte offset) or base.%Xn (bit)
        return mlirGenRelativeAccess(location, lhs, dir);
      }
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
      // time/date types compare and mix arithmetically with integers as
      // plain integers of their storage width
      if (t.isa<S5TimeType>() || t.isa<DateType>())
        return 16;
      if (t.isa<TimeType>() || t.isa<TimeOfDayType>())
        return 32;
      if (t.isa<DateAndTimeType>())
        return 64;
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
    // Normalize when widths differ OR when the types differ at the same
    // width (e.g. !scl.time vs i32): normalize to a plain IntegerType.
    if (lw > 0 && rw > 0 && (lw != rw || lhs.getType() != rhs.getType())) {
      int w = std::max(lw, rw);
      lhs = toIntOfWidth(lhs, w);
      rhs = toIntOfWidth(rhs, w);
    }

    // Mixed real/integer operands: convert the integer side to real
    // (sitofp at lowering) so comparisons/arithmetic have matching types.
    if (lhs.getType().isa<RealType>() != rhs.getType().isa<RealType>()) {
      auto realTy = lhs.getType().isa<RealType>() ? lhs.getType() : rhs.getType();
      auto toReal = [&](mlir::Value v) -> mlir::Value {
        if (v.getType() == realTy)
          return v;
        return builder.create<DialectCastOp>(location, realTy, v).getResult();
      };
      lhs = toReal(lhs);
      rhs = toReal(rhs);
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
      return builder.create<UnaryNotOp>(location, coerceToBool(location, rhs));
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
      // Call of an FB instance variable (e.g. #TIME(IN := ...) on a
      // TON_TIME member): a call.fb on that instance.
      if (auto fbSym = fbInstanceOf(callee->getName())) {
        if (failed(mlirGenFbCall(expr, callee->getName(), *fbSym)))
          return mlir::failure();
        return mlir::success();
      }
      mlirGenFcCall(expr, callee->getName(), /*allowVoid=*/true);
      return mlir::success();
    })
    .Case<BinaryExpressionAST>([&](auto callee) {
      if (callee->getOp() != tok_dot)
        return mlir::failure();
      // `#valve.timerClose.TON(...)`: explicit timer-FB suffix on an
      // instance member — call TON/TOF/TP on the member before the suffix.
      if (auto *rhsVar = dyn_cast<SimpleVariableAST>(callee->getRhs())) {
        std::string rhsUp = std::string(rhsVar->getName().upper());
        if ((rhsUp == "TON" || rhsUp == "TOF" || rhsUp == "TP") &&
            isa<BinaryExpressionAST>(callee->getLhs())) {
          mlir::Value instAddr = mlirGen(callee->getLhs());
          if (!instAddr)
            return mlir::failure();
          if (failed(mlirGenFbCallOn(expr, instAddr, rhsUp)))
            return mlir::failure();
          return mlir::success();
        }
      }
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
      // `FB1 "DB1"(...)`: lhs is an FB name (source-defined or builtin
      // timer), rhs a (global) instance DB.
      std::string fbUp = llvm::StringRef(unquote(fbVar->getName()).c_str()).upper();
      if (llvm::StringRef(fbUp).ends_with("_TIME"))
        fbUp = fbUp.substr(0, fbUp.size() - 5);
      if (fbNames.contains(fbVar->getName()) || fbUp == "TON" ||
          fbUp == "TOF" || fbUp == "TP") {
        if (failed(mlirGenFbCall(expr, idbVar->getName(), fbVar->getName()))) {
          emitError(location, "invalid FB call");
          return mlir::failure();
        }
        return mlir::success();
      }
      // `#valve.timerOpen(...)`: the callee is an FB-instance member access
      // (e.g. a TON_TIME timer inside a struct); call the member's FB on the
      // member address.
      mlir::Value instAddr = mlirGen(callee);
      if (!instAddr)
        return mlir::failure();
      auto it = byteViewTypes.find(instAddr);
      if (it == byteViewTypes.end() || !it->second.isa<InstanceDbType>()) {
        emitError(location, "call target '")
            << fbVar->getName() << "." << idbVar->getName()
            << "' is not a function block instance";
        return mlir::failure();
      }
      if (failed(mlirGenFbCallOn(expr, instAddr,
                                 it->second.cast<InstanceDbType>().getFbSymbol())))
        return mlir::failure();
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
    if (auto fbSym = fbInstanceOf(callee->getName())) {
      emitError(location) << "function block '" << callee->getName()
                          << "' call used as a value";
      return nullptr;
    }
    return mlirGenFcCall(expr, callee->getName());
   }

  mlir::LogicalResult mlirGenCallParameters(
      const FunctionCallAST *fc,
      std::function<void (StringRef, mlir::Value)> fN,
      std::function<void (mlir::Value)> fNN,
      const llvm::StringMap<mlir::Type> *paramTypes = nullptr,
      std::function<mlir::LogicalResult (StringRef, const StringConstantAST *)>
          fStr = nullptr) {
    // IEC 61131-3 ST allows all-positional argument lists (e.g. MAX(a, b),
    // LIMIT(MN, IN, MX)); mixing positional and named arguments is not
    // allowed.
    bool sawNamed = false, sawPositional = false;
    for (auto &arg : fc->getParameters()) {
      auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
      bool isNamed = binary && (binary->getOp() == tok_assignment ||
                                binary->getOp() == tok_assign_output);
      if (isNamed && sawPositional) {
        emitError(loc(arg->loc()), "named argument after positional argument");
        return mlir::failure();
      }
      if (!isNamed)
        sawPositional = true;
      else
        sawNamed = true;
      if (isNamed) {
        auto lhs = binary->getLhs();
        auto name = dyn_cast<SimpleVariableAST>(lhs);
        if (!name) {
          emitWarning(loc(lhs->loc()))
              << "invalid parameter name, argument skipped";
          continue;
        }
        if (auto *str = dyn_cast<StringConstantAST>(binary->getRhs())) {
          // FB calls can store the literal bytes directly into the IDB
          // member (per-byte stores); FC calls need a whole-array value.
          if (fStr && succeeded(fStr(name->getName(), str)))
            continue;
          mlir::Value value = mlirGenStringLiteralValue(
              loc(binary->getRhs()->loc()), paramTypes, name->getName(), str);
          if (!value)
            return mlir::failure();
          fN(name->getName(), value);
        } else {
          mlir::Value value = mlirGenRValue(binary->getRhs());
          if (!value)
            return mlir::failure();
          fN(name->getName(), value);
        }
      } else {
        if (!fNN) {
          emitError(loc(arg->loc()), "parameter without name");
          return mlir::failure();
        }
        if (sawNamed) {
          emitError(loc(arg->loc()),
                    "positional argument after named argument");
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

  /// IEC 61131-3 conversion builtin `SRC_TO_DST` (single argument): cast
  /// the argument to the destination type via scl.int.cast (zext for the
  /// unsigned byte/word/dword targets). Returns nullptr if not a builtin.
  /// TIA DATE_AND_TIME field extractor DT_YEAR..DT_MSEC / DT_WEEKDAY:
  /// the DT is stored as 8 big-endian BCD bytes in an i64 (ADR-001);
  /// byte k (0=year .. 5=second, 6=msec/weekday) is extracted with
  /// div/mod and BCD-decoded. Returns nullptr if `callee` is not a DT_*.
  mlir::Value mlirGenDtField(mlir::Location location,
                             const FunctionCallAST *expr, StringRef callee) {
    std::string up = std::string(callee.upper());
    if (!llvm::StringRef(up).starts_with("DT_"))
      return nullptr;
    // byte index within the 8-byte DT and whether to BCD-decode
    int byteIdx = -1;
    bool bcd = true;
    std::string field = up.substr(3);
    if (field == "YEAR") byteIdx = 0;
    else if (field == "MONTH") byteIdx = 1;
    else if (field == "DAY") byteIdx = 2;
    else if (field == "HOUR") byteIdx = 3;
    else if (field == "MINUTE") byteIdx = 4;
    else if (field == "SECOND") byteIdx = 5;
    else if (field == "MSEC") byteIdx = 6;      // high nibble pair of byte 6
    else if (field == "WEEKDAY") { byteIdx = 6; bcd = false; } // low nibble
    else if (field == "NANOSECOND") byteIdx = 7;
    else if (field == "TOD") byteIdx = -2; // special: time-of-day in ms
    else return nullptr;

    mlir::Value arg;
    if (failed(mlirGenCallParameters(
            expr, [&](StringRef, mlir::Value v) { arg = v; },
            [&](mlir::Value v) { arg = v; })))
      return nullptr;
    if (!arg)
      return nullptr;
    auto *ctx = builder.getContext();
    auto i64 = IntegerType::get(ctx, 64);
    auto i16 = IntegerType::get(ctx, 16);
    if (arg.getType() != i64) {
      auto cast = builder.create<IntegerCastOp>(location, i64, arg);
      if (arg.getType().isa<LogicalType>())
        cast.setZextAttr(builder.getUnitAttr());
      arg = cast.getResult();
    }
    auto i64Const = [&](int64_t v) {
      return builder.create<ConstantOp>(location, i64,
                                        builder.getIntegerAttr(i64, v)).getResult();
    };
    if (byteIdx == -2) {
      // DT_TOD: ms since midnight from the BCD hour/min/sec/msec bytes
      auto bcdByte = [&](int idx) -> mlir::Value {
        int64_t p = 1;
        for (int i = 0; i < 7 - idx; ++i)
          p *= 256;
        auto b = builder.create<ModOp>(
            location,
            builder.create<DivOp>(location, arg, i64Const(p)).getResult(),
            i64Const(256)).getResult();
        auto lo = builder.create<ModOp>(location, b, i64Const(16)).getResult();
        auto hi = builder.create<DivOp>(location, b, i64Const(16)).getResult();
        return builder.create<AddOp>(
            location, lo,
            builder.create<MulOp>(location, hi, i64Const(10)).getResult()).getResult();
      };
      mlir::Value ms = builder.create<MulOp>(location, bcdByte(6), i64Const(1)).getResult();
      ms = builder.create<AddOp>(
          location, ms,
          builder.create<MulOp>(location, bcdByte(5), i64Const(1000)).getResult()).getResult();
      ms = builder.create<AddOp>(
          location, ms,
          builder.create<MulOp>(location, bcdByte(4), i64Const(60000)).getResult()).getResult();
      ms = builder.create<AddOp>(
          location, ms,
          builder.create<MulOp>(location, bcdByte(3), i64Const(3600000)).getResult()).getResult();
      auto todTy = getType(tok_time_of_day);
      auto i32 = IntegerType::get(ctx, 32);
      if (ms.getType() != i32)
        ms = builder.create<IntegerCastOp>(location, i32, ms).getResult();
      if (ms.getType() != todTy)
        ms = builder.create<DialectCastOp>(location, todTy, ms).getResult();
      return ms;
    }
    // byte = (v / 256^(7-byteIdx)) % 256   (unsigned shifts via div/mod)
    int64_t pow = 1;
    for (int i = 0; i < 7 - byteIdx; ++i)
      pow *= 256;
    auto byteVal = builder.create<ModOp>(
        location, builder.create<DivOp>(location, arg, i64Const(pow)).getResult(),
        i64Const(256)).getResult();
    mlir::Value fieldVal = byteVal;
    if (bcd) {
      // BCD decode: (b % 16) + 10 * (b / 16)
      auto lo = builder.create<ModOp>(location, byteVal, i64Const(16)).getResult();
      auto hi = builder.create<DivOp>(location, byteVal, i64Const(16)).getResult();
      fieldVal = builder.create<AddOp>(
          location, lo, builder.create<MulOp>(location, hi, i64Const(10)).getResult()).getResult();
    } else {
      fieldVal = builder.create<ModOp>(location, byteVal, i64Const(16)).getResult();
    }
    // result type INT
    auto resTy = getType(tok_int);
    if (fieldVal.getType() != i16)
      fieldVal = builder.create<IntegerCastOp>(location, i16, fieldVal).getResult();
    (void)resTy;
    return fieldVal;
  }

  // SPEC[SCL] IEC 61131-3 ST type conversions (INT_TO_WORD, CHAR_TO_INT, ...) -> mlirGenBuiltinConversion; string functions (LEN/CONCAT/...) -> mlirGenStringBuiltin
  mlir::Value mlirGenBuiltinConversion(mlir::Location location,
                                        const FunctionCallAST *expr,
                                        StringRef callee) {
    auto context = builder.getContext();
    // destination type from the _TO_<DST> suffix
    auto pos = callee.rfind("_TO_");
    if (pos == StringRef::npos || pos + 4 >= callee.size())
      return nullptr;
    std::string dst = std::string(callee.substr(pos + 4).upper());
    Token dstTok;
    if (dst == "INT") dstTok = tok_int;
    else if (dst == "DINT") dstTok = tok_dint;
    else if (dst == "BYTE" || dst == "USINT") dstTok = tok_byte;
    else if (dst == "WORD" || dst == "UINT") dstTok = tok_word;
    else if (dst == "DWORD" || dst == "UDINT" || dst == "ULINT")
      dstTok = tok_dword;
    else if (dst == "CHAR" || dst == "SINT") dstTok = tok_char;
    else if (dst == "BOOL") dstTok = tok_bool;
    else if (dst == "TIME") dstTok = tok_time;
    else if (dst == "S5TIME") dstTok = tok_s5time;
    else if (dst == "DATE") dstTok = tok_date;
    else if (dst == "TOD" || dst == "TIME_OF_DAY") dstTok = tok_time_of_day;
    else if (dst == "DT" || dst == "DATE_AND_TIME" || dst == "DTL")
      dstTok = tok_date_and_time;
    else if (dst == "REAL" || dst == "LREAL")
      dstTok = tok_real;
    else
      return nullptr; // string targets not supported yet
    mlir::Type resTy = getType(dstTok);
    if (!resTy)
      return nullptr;

    mlir::Value arg;
    if (failed(mlirGenCallParameters(
            expr, [&](StringRef, mlir::Value v) { arg = v; },
            [&](mlir::Value v) { arg = v; })))
      return nullptr;
    if (!arg)
      return nullptr;
    // string/array source (e.g. CHAR_TO_INT(s) on a string): approximate as 0
    if (arg.getType().isa<ArrayType>()) {
      emitWarning(location) << "conversion of a string value, assuming 0";
      return builder.create<ConstantOp>(
          location, resTy,
          builder.getIntegerAttr(builder.getIntegerType(16), 0));
    }
    // REAL source: dialect-cast straight to the target (fptosi/… at lowering)
    if (arg.getType().isa<RealType>() && !resTy.isa<RealType>()) {
      if (arg.getType() != resTy)
        arg = builder.create<DialectCastOp>(location, resTy, arg);
      return arg;
    }
    // cast to the destination width as an IntegerType first (int.cast only
    // has integer results), then dialect-cast to the scl target type
    int width = 16;
    if (auto it = resTy.dyn_cast<IntegerType>())
      width = it.getWidth();
    else if (auto lt = resTy.dyn_cast<LogicalType>())
      width = lt.getWidth();
    else if (resTy.isa<RealType>())
      width = 32;
    else if (resTy.isa<S5TimeType>() || resTy.isa<DateType>())
      width = 16;
    else if (resTy.isa<TimeType>() || resTy.isa<TimeOfDayType>())
      width = 32;
    else if (resTy.isa<DateAndTimeType>())
      width = 64;
    if (resTy.isa<RealType>()) {
      // int -> real: sitofp at lowering, via mlir.cast i32 -> !scl.real
      auto intTy = IntegerType::get(context, 32);
      if (arg.getType() != intTy) {
        auto cast = builder.create<IntegerCastOp>(location, intTy, arg);
        if (arg.getType().isa<LogicalType>())
          cast.setZextAttr(builder.getUnitAttr()); // byte/word/dword: unsigned
        arg = cast.getResult();
      }
      return builder.create<DialectCastOp>(location, resTy, arg).getResult();
    }
    auto intTy = IntegerType::get(context, width);
    if (arg.getType() != intTy) {
      auto cast = builder.create<IntegerCastOp>(location, intTy, arg);
      if (resTy.isa<LogicalType>())
        cast.setZextAttr(builder.getUnitAttr()); // byte/word/dword: unsigned
      arg = cast.getResult();
    }
    if (arg.getType() != resTy)
      arg = builder.create<DialectCastOp>(location, resTy, arg);
    return arg;
  }

  // ------------------------------------------------------------------
  // IEC 61131-3 ST string functions (LEN/LEFT/RIGHT/MID/FIND/CONCAT/
  // DELETE/INSERT/REPLACE and the numeric<->string conversions).
  //
  // Strings are flat zero-padded char buffers (ArrayType of !scl.char), so
  // the current length of a string is the index of its first zero byte
  // (C-string semantics) — see mlirGenStoreStringBytes/mlirGenStringCompare.
  // The functions are expanded inline as unrolled byte loops (same pattern
  // as the string compare above), so they lower to plain loads/stores/arith
  // in every backend (JIT and AWL alike). String results are returned as the
  // address of a fresh String[254] temp buffer.
  // SPEC[SCL] IEC 61131-3 ST string functions LEN/LEFT/RIGHT/MID/FIND/CONCAT/DELETE/INSERT/REPLACE and <type>_TO_STRING/STRING_TO_<type> -> mlirGenStringBuiltin (approx: flat zero-padded buffers with C-string length; STRING_TO_REAL/REAL_TO_STRING/VAL_STRG remain external stubs)
  struct StrArg {
    mlir::Value addr;   // char-array address (var/member/builtin result)
    ArrayType arrTy;
    std::string lit;    // string literal bytes
    mlir::Value scalar; // single char value
    int64_t maxLen() const {
      if (addr) {
        auto d = const_cast<ArrayType &>(arrTy).getDimensions();
        return d[0].second - d[0].first + 1;
      }
      return scalar ? 1 : (int64_t)lit.size();
    }
  };
  struct StrBuf {
    mlir::Value addr;
    ArrayType ty;
  };

  mlir::Type charTy() const { return IntegerType::get(builder.getContext(), 8); }
  mlir::Type dintTy() const { return IntegerType::get(builder.getContext(), 32); }

  mlir::Value sclConst(mlir::Location location, mlir::Type ty, int64_t v) {
    int width = 32;
    if (auto it = ty.dyn_cast<IntegerType>())
      width = it.getWidth();
    else if (auto lt = ty.dyn_cast<LogicalType>())
      width = lt.getWidth();
    return builder.create<ConstantOp>(
        location, ty, builder.getIntegerAttr(builder.getIntegerType(width), v));
  }
  mlir::Value boolConst(mlir::Location location, bool v) {
    return builder.create<ConstantOp>(
        location, getType(tok_bool),
        builder.getIntegerAttr(builder.getI1Type(), v ? 1 : 0));
  }

  /// widen any integer-ish value to scl dint (char/byte zero-extended)
  mlir::Value toDint(mlir::Location location, mlir::Value v) {
    auto d = dintTy();
    if (v.getType() == d)
      return v;
    if (auto it = v.getType().dyn_cast<IntegerType>()) {
      auto c = builder.create<IntegerCastOp>(location, d, v);
      if (it.getWidth() == 8)
        c.setZextAttr(builder.getUnitAttr()); // char: unsigned
      return c.getResult();
    }
    if (v.getType().isa<LogicalType>()) {
      auto c = builder.create<IntegerCastOp>(location, d, v);
      c.setZextAttr(builder.getUnitAttr()); // byte/word/dword/bool: unsigned
      return c.getResult();
    }
    return builder.create<DialectCastOp>(location, d, v).getResult();
  }
  /// narrow to scl char (!scl.char)
  mlir::Value toChar(mlir::Location location, mlir::Value v) {
    auto c8 = charTy();
    if (v.getType() == c8)
      return v;
    if (v.getType().isa<IntegerType>() || v.getType().isa<LogicalType>())
      return builder.create<IntegerCastOp>(location, c8, v).getResult();
    return builder.create<DialectCastOp>(location, c8, v).getResult();
  }
  /// cond ? a : b for same-typed scl integers a/b (cond: !scl.bool)
  mlir::Value mux(mlir::Location location, mlir::Value cond, mlir::Value a,
                  mlir::Value b) {
    auto ty = a.getType();
    auto c = builder.create<IntegerCastOp>(location, ty, cond);
    c.setZextAttr(builder.getUnitAttr());
    auto one = sclConst(location, ty, 1);
    auto nc = builder.create<SubOp>(location, one, c).getResult();
    return builder.create<AddOp>(
        location, builder.create<MulOp>(location, a, c).getResult(),
        builder.create<MulOp>(location, b, nc).getResult()).getResult();
  }
  /// cast a scl dint value to an arbitrary scl target type
  mlir::Value castToType(mlir::Location location, mlir::Value v,
                         mlir::Type target) {
    if (v.getType() == target)
      return v;
    if (target.isa<IntegerType>())
      return builder.create<IntegerCastOp>(location, target, v).getResult();
    return builder.create<DialectCastOp>(location, target, v).getResult();
  }

  mlir::Value elemAddr(mlir::Location location, mlir::Value addr,
                       ArrayType arrTy, mlir::Value idx) {
    return builder.create<GetIndexOp>(
        location, AddressType::get(arrTy.getElementType()), addr,
        mlir::ValueRange{idx}, mlir::TypeAttr::get(arrTy));
  }
  mlir::Value elemAddrAt(mlir::Location location, mlir::Value addr,
                         ArrayType arrTy, int64_t i) {
    return elemAddr(location, addr, arrTy,
                    sclConst(location, getType(tok_int),
                             arrTy.getDimensions()[0].first + i));
  }

  /// byte at compile-time index i of a string arg, as scl dint. The index
  /// is clamped into the source buffer (callers only use the byte when the
  /// logical index is in range, so the loaded value is discarded otherwise).
  mlir::Value byteD(mlir::Location location, const StrArg &s, int64_t i) {
    if (s.addr) {
      int64_t n = s.maxLen();
      if (i >= n)
        i = n - 1;
      return toDint(location,
                    builder.create<LoadOp>(location, charTy(),
                                           elemAddrAt(location, s.addr, s.arrTy, i)));
    }
    if (s.scalar)
      return toDint(location, i == 0 ? s.scalar : sclConst(location, charTy(), 0));
    return sclConst(location, dintTy(),
                    i < (int64_t)s.lit.size() ? (uint8_t)s.lit[(size_t)i] : 0);
  }
  /// clamp a runtime (scl dint) index into [0, n-1] so speculative loads
  /// stay in bounds (the value is only used when logically in range)
  mlir::Value clampIdx(mlir::Location location, mlir::Value idx, int64_t n) {
    auto d = dintTy();
    mlir::Value lo = builder.create<LessThanOp>(location, idx, sclConst(location, d, 0));
    idx = mux(location, lo, sclConst(location, d, 0), idx);
    mlir::Value hi = builder.create<GreaterThanOp>(
        location, idx, sclConst(location, d, n - 1));
    return mux(location, hi, sclConst(location, d, n - 1), idx);
  }
  /// byte at runtime (scl dint) index of a string arg, as scl dint
  mlir::Value byteVD(mlir::Location location, const StrArg &s,
                     mlir::Value idx) {
    if (s.addr)
      return toDint(location, builder.create<LoadOp>(
                                  location, charTy(),
                                  elemAddr(location, s.addr, s.arrTy, idx)));
    // literal/scalar: select over the known bytes
    mlir::Value r = sclConst(location, dintTy(), 0);
    int64_t n = s.maxLen();
    for (int64_t i = 0; i < n; ++i)
      r = mux(location,
              builder.create<EqualOp>(location, idx,
                                      sclConst(location, dintTy(), i)),
              byteD(location, s, i), r);
    return r;
  }

  /// current length (index of the first zero byte) of a string arg, as scl int
  mlir::Value strLenOf(mlir::Location location, const StrArg &s) {
    auto d = dintTy();
    mlir::Value acc = sclConst(location, d, 0);
    int64_t n = s.maxLen();
    for (int64_t i = 0; i < n; ++i) {
      mlir::Value nz = builder.create<NotEqualOp>(
          location, byteD(location, s, i), sclConst(location, d, 0));
      acc = mux(location, nz, sclConst(location, d, i + 1), acc);
    }
    return builder.create<IntegerCastOp>(location, getType(tok_int), acc)
        .getResult();
  }

  StrBuf newStrBuf(mlir::Location location, int64_t maxLen) {
    ArrayType arrTy = ArrayType::get({{0, (int)maxLen - 1}}, charTy());
    return {builder.create<TempVariableOp>(location, AddressType::get(arrTy),
                                           builder.getStringAttr("strres")),
            arrTy};
  }
  void storeStrByte(mlir::Location location, const StrBuf &buf, int64_t i,
                    mlir::Value byteDint) {
    builder.create<StoreOp>(
        location, elemAddrAt(location, buf.addr, buf.ty, i),
        toChar(location, byteDint));
  }

  /// Build a string argument (array address / literal / char scalar) from an
  /// AST expression.
  StrArg strArgOf(mlir::Location location, const ExpressionAST *e) {
    StrArg s;
    if (const StringConstantAST *lit = asStringLit(e)) {
      s.lit = lit->getValue();
      return s;
    }
    mlir::Value v = mlirGen(e);
    if (!v)
      return s;
    if (v.getType().isa<AddressType>()) {
      if (ArrayType at = arrayTypeOfAddress(v)) {
        s.addr = v;
        s.arrTy = at;
        return s;
      }
      // scalar address (char var / member): load with the declared type
      mlir::Type vt = v.getType().cast<AddressType>().getElementType();
      auto it = byteViewTypes.find(v);
      if (it != byteViewTypes.end())
        vt = it->second;
      s.scalar = toChar(location, builder.create<LoadOp>(location, vt, v));
      return s;
    }
    if (auto at = v.getType().dyn_cast<ArrayType>()) {
      // whole-array value (e.g. a user FC returning a string): spill to a temp
      auto buf = builder.create<TempVariableOp>(location, AddressType::get(at),
                                                builder.getStringAttr("strarg"));
      builder.create<StoreOp>(location, buf, v);
      s.addr = buf;
      s.arrTy = at;
      return s;
    }
    s.scalar = toChar(location, v);
    return s;
  }

  /// Numeric argument of a conversion builtin, as scl dint.
  mlir::Value numArgD(mlir::Location location, const ExpressionAST *e) {
    mlir::Value v = mlirGenRValue(e);
    return v ? toDint(location, v) : mlir::Value();
  }

  mlir::Value mlirGenStringBuiltin(mlir::Location location,
                                   const FunctionCallAST *expr,
                                   StringRef calleeRef) {
    std::string up = unquote(calleeRef);
    up = llvm::StringRef(up).upper();
    // collect arguments: named (IN1/IN2/IN/L/P/...) or positional
    SmallVector<std::pair<std::string, const ExpressionAST *>, 4> args;
    for (auto &p : expr->getParameters()) {
      if (auto *b = dyn_cast<BinaryExpressionAST>(p.get())) {
        if (b->getOp() == tok_assignment || b->getOp() == tok_assign_output) {
          if (auto *n = dyn_cast<SimpleVariableAST>(b->getLhs())) {
            args.push_back({std::string(n->getName().upper()), b->getRhs()});
            continue;
          }
        }
      }
      args.push_back({"", p.get()});
    }
    bool anyNamed = llvm::any_of(args,
                                 [](auto &a) { return !a.first.empty(); });
    auto arg = [&](const char *name, int pos) -> const ExpressionAST * {
      if (anyNamed) {
        for (auto &a : args)
          if (a.first == name)
            return a.second;
        return nullptr;
      }
      return pos < (int)args.size() ? args[(size_t)pos].second : nullptr;
    };

    auto d = dintTy();
    const int64_t RESULT_MAX = 254; // String[254]: max SCL string size

    if (up == "LEN") {
      auto *e = arg("IN", 0);
      if (!e)
        return nullptr;
      return strLenOf(location, strArgOf(location, e));
    }

    if (up == "CONCAT" || up == "LEFT" || up == "RIGHT" || up == "MID" ||
        up == "DELETE" || up == "INSERT" || up == "REPLACE" || up == "FIND") {
      auto *e1 = arg("IN1", 0);
      auto *e2 = arg("IN2", 1);
      // single-input functions take IN
      if (!e1)
        e1 = arg("IN", 0);
      StrArg a = strArgOf(location, e1);
      if (!a.addr && a.lit.empty() && !a.scalar)
        return nullptr;
      int64_t nA = a.maxLen();
      mlir::Value lenA = toDint(location, strLenOf(location, a));
      StrBuf buf = newStrBuf(location, RESULT_MAX);

      if (up == "CONCAT") {
        if (!e2)
          return nullptr;
        StrArg b = strArgOf(location, e2);
        mlir::Value lenB = toDint(location, strLenOf(location, b));
        int64_t nB = b.maxLen();
        for (int64_t i = 0; i < RESULT_MAX; ++i) {
          mlir::Value ic = sclConst(location, d, i);
          mlir::Value inA = builder.create<LessThanOp>(location, ic, lenA);
          mlir::Value idxB = clampIdx(
              location, builder.create<SubOp>(location, ic, lenA).getResult(), nB);
          mlir::Value byteB = byteVD(location, b, idxB);
          // zero beyond b's current length
          byteB = mux(location,
                      builder.create<LessThanOp>(location, idxB, lenB), byteB,
                      sclConst(location, d, 0));
          mlir::Value inRange = builder.create<LessThanOp>(
              location, ic, builder.create<AddOp>(location, lenA, lenB).getResult());
          mlir::Value byte = mux(location, inA, byteD(location, a, i), byteB);
          byte = mux(location, inRange, byte, sclConst(location, d, 0));
          storeStrByte(location, buf, i, byte);
        }
        return buf.addr;
      }
      if (up == "LEFT" || up == "RIGHT" || up == "MID" || up == "DELETE" ||
          up == "INSERT" || up == "REPLACE" || up == "FIND") {
        // integer parameters L / P (1-based position), as scl dint
        auto numArg = [&](const char *name, int pos) -> mlir::Value {
          auto *e = arg(name, pos);
          return e ? numArgD(location, e) : mlir::Value();
        };
        mlir::Value L = numArg("L", 1), P = numArg("P", 2);
        if (up == "LEFT" || up == "RIGHT") {
          if (!L)
            return nullptr;
        } else if (up == "MID" || up == "DELETE" || up == "REPLACE") {
          if (!L || !P)
            return nullptr;
        }
        if (up == "FIND") {
          if (!e2)
            return nullptr;
          StrArg b = strArgOf(location, e2);
          int64_t nB = b.maxLen();
          mlir::Value lenB = toDint(location, strLenOf(location, b));
          mlir::Value pos = sclConst(location, d, 0);
          for (int64_t s = 0; s < nA; ++s) {
            mlir::Value match = boolConst(location, true);
            for (int64_t k = 0; k < nB; ++k) {
              if (s + k >= nA)
                continue; // would read past a's buffer: no match
              mlir::Value eq = builder.create<EqualOp>(
                  location, byteD(location, b, k), byteD(location, a, s + k));
              mlir::Value ge = builder.create<GreaterEqualOp>(
                  location, sclConst(location, d, k), lenB); // k >= lenB
              match = builder.create<AndOp>(
                  location, match,
                  builder.create<OrOp>(location, ge, eq).getResult());
            }
            mlir::Value first = builder.create<EqualOp>(
                location, pos, sclConst(location, d, 0));
            match = builder.create<AndOp>(location, match, first);
            pos = mux(location, match, sclConst(location, d, s + 1), pos);
          }
          return builder.create<IntegerCastOp>(location, getType(tok_int), pos)
              .getResult();
        }
        // string-producing functions: byte i of the result
        StrArg b;      // second string arg (INSERT/REPLACE), else unused
        mlir::Value lenB, nB;
        if (up == "INSERT" || up == "REPLACE") {
          if (!e2)
            return nullptr;
          b = strArgOf(location, e2);
          lenB = toDint(location, strLenOf(location, b));
          nB = sclConst(location, d, b.maxLen());
        }
        // pm1 = P-1 (0-based insertion/deletion point); P is only present
        // for MID/DELETE/INSERT/REPLACE (null otherwise)
        mlir::Value pm1 = P
            ? builder.create<SubOp>(location, P, sclConst(location, d, 1)).getResult()
            : sclConst(location, d, 0);
        for (int64_t i = 0; i < RESULT_MAX; ++i) {
          mlir::Value ic = sclConst(location, d, i);
          mlir::Value byte = sclConst(location, d, 0);
          if (up == "LEFT") {
            byte = mux(location,
                       builder.create<LessThanOp>(location, ic, L),
                       byteD(location, a, i), sclConst(location, d, 0));
          } else if (up == "RIGHT") {
            // last L chars, left-aligned in the result: byte[i] = a[start+i]
            mlir::Value start = mux(
                location,
                builder.create<GreaterThanOp>(location, lenA, L),
                builder.create<SubOp>(location, lenA, L).getResult(),
                sclConst(location, d, 0));
            mlir::Value src = clampIdx(
                location, builder.create<AddOp>(location, start, ic).getResult(), nA);
            mlir::Value in = builder.create<AndOp>(
                location, builder.create<LessThanOp>(location, ic, L),
                builder.create<LessThanOp>(
                    location, builder.create<AddOp>(location, start, ic).getResult(),
                    lenA));
            byte = mux(location, in, byteVD(location, a, src), sclConst(location, d, 0));
          } else if (up == "MID") {
            mlir::Value src = clampIdx(
                location, builder.create<AddOp>(location, pm1, ic).getResult(), nA);
            mlir::Value in = builder.create<AndOp>(
                location, builder.create<LessThanOp>(location, ic, L),
                builder.create<LessThanOp>(location,
                                           builder.create<AddOp>(location, pm1, ic).getResult(),
                                           lenA));
            mlir::Value bb = byteVD(location, a, src);
            byte = mux(location, in, bb, sclConst(location, d, 0));
          } else if (up == "DELETE") {
            // result[i] = a[i] for i < P-1, else a[i+L] while in range
            mlir::Value src = clampIdx(
                location, builder.create<AddOp>(location, ic, L).getResult(), nA);
            mlir::Value fromTail = builder.create<AndOp>(
                location,
                builder.create<GreaterEqualOp>(location, ic, pm1),
                builder.create<LessThanOp>(
                    location, builder.create<AddOp>(location, ic, L).getResult(), lenA));
            mlir::Value byteHead = mux(
                location, builder.create<LessThanOp>(location, ic, pm1),
                byteD(location, a, i), sclConst(location, d, 0));
            mlir::Value byteTail = mux(location, fromTail,
                                       byteVD(location, a, src), sclConst(location, d, 0));
            byte = mux(location,
                       builder.create<LessThanOp>(location, ic, pm1),
                       byteHead, byteTail);
          } else if (up == "INSERT" || up == "REPLACE") {
            // result = a[0..P-2] + b + a[P-1(+L)..]  (REPLACE skips L chars of a)
            mlir::Value skip = up == "REPLACE" ? L : sclConst(location, d, 0);
            int64_t nBInt = b.maxLen();
            mlir::Value srcB = clampIdx(
                location, builder.create<SubOp>(location, ic, pm1).getResult(), nBInt);
            mlir::Value byteB = byteVD(location, b, srcB);
            byteB = mux(location,
                        builder.create<LessThanOp>(location, srcB, lenB), byteB,
                        sclConst(location, d, 0));
            // tail of a after the replaced/inserted region: a[i - lenB + skip]
            mlir::Value srcA = clampIdx(
                location,
                builder.create<AddOp>(
                    location,
                    builder.create<SubOp>(location, ic, lenB).getResult(),
                    skip).getResult(),
                nA);
            mlir::Value inMid = builder.create<AndOp>(
                location,
                builder.create<GreaterEqualOp>(location, ic, pm1),
                builder.create<LessThanOp>(
                    location, ic,
                    builder.create<AddOp>(location, pm1, lenB).getResult()));
            mlir::Value inTail = builder.create<AndOp>(
                location,
                builder.create<GreaterEqualOp>(
                    location, ic,
                    builder.create<AddOp>(location, pm1, lenB).getResult()),
                builder.create<LessThanOp>(
                    location, srcA, lenA));
            mlir::Value byteHead = mux(
                location, builder.create<LessThanOp>(location, ic, pm1),
                byteD(location, a, i), sclConst(location, d, 0));
            mlir::Value byteTail = mux(location, inTail,
                                       byteVD(location, a, srcA), sclConst(location, d, 0));
            byte = mux(location, builder.create<LessThanOp>(location, ic, pm1),
                       byteHead, mux(location, inMid, byteB, byteTail));
          }
          storeStrByte(location, buf, i, byte);
        }
        return buf.addr;
      }
    }

    // <type>_TO_STRING(IN): decimal representation of an integer value
    if (llvm::StringRef(up).ends_with("_TO_STRING") && up != "STRING_TO_STRING") {
      auto *e = arg("IN", 0);
      if (!e)
        return nullptr;
      mlir::Value v = numArgD(location, e);
      if (!v)
        return nullptr;
      mlir::Value neg = builder.create<LessThanOp>(location, v, sclConst(location, d, 0));
      mlir::Value a = mux(location, neg,
                         builder.create<SubOp>(location, sclConst(location, d, 0), v).getResult(),
                         v);
      // digits d[k] = (a / 10^k) % 10, least significant first
      SmallVector<mlir::Value, 10> dig;
      mlir::Value t = a;
      for (int k = 0; k < 10; ++k) {
        dig.push_back(builder.create<ModOp>(location, t, sclConst(location, d, 10)).getResult());
        t = builder.create<DivOp>(location, t, sclConst(location, d, 10)).getResult();
      }
      // number of digits of a (1..10)
      mlir::Value len = sclConst(location, d, 1);
      int64_t p10 = 10;
      for (int k = 1; k < 10; ++k, p10 *= 10)
        len = mux(location,
                  builder.create<GreaterEqualOp>(location, a, sclConst(location, d, p10)),
                  sclConst(location, d, k + 1), len);
      mlir::Value sign = mux(location, neg, sclConst(location, d, 1),
                             sclConst(location, d, 0));
      mlir::Value total = builder.create<AddOp>(location, len, sign).getResult();
      StrBuf buf = newStrBuf(location, RESULT_MAX);
      for (int64_t p = 0; p < RESULT_MAX; ++p) {
        mlir::Value byte;
        if (p > 10) {
          byte = sclConst(location, d, 0);
        } else {
          mlir::Value pc = sclConst(location, d, p);
          mlir::Value inL = builder.create<LessThanOp>(location, pc, total);
          mlir::Value j = builder.create<SubOp>(location, pc, sign).getResult();
          mlir::Value di = builder.create<SubOp>(
              location, builder.create<SubOp>(location, len, sclConst(location, d, 1)).getResult(),
              j).getResult(); // digit index from the right: len-1-j
          mlir::Value dsel = sclConst(location, d, 0);
          for (int k = 0; k < 10; ++k)
            dsel = mux(location,
                       builder.create<EqualOp>(location, di, sclConst(location, d, k)),
                       dig[(size_t)k], dsel);
          mlir::Value ch = builder.create<AddOp>(location, dsel, sclConst(location, d, 48)).getResult();
          byte = mux(location, inL, ch, sclConst(location, d, 0));
          if (p == 0)
            byte = mux(location, neg, sclConst(location, d, 45), byte); // '-'
        }
        storeStrByte(location, buf, p, byte);
      }
      return buf.addr;
    }

    // STRING_TO_<inttype>(IN): decimal parse (optional leading '-')
    if (llvm::StringRef(up).starts_with("STRING_TO_")) {
      std::string dst = up.substr(std::string("STRING_TO_").size());
      mlir::Type target = nullptr;
      if (dst == "INT" || dst == "SINT" || dst == "CHAR")
        target = dst == "CHAR" ? getType(tok_char) : getType(tok_int);
      else if (dst == "DINT" || dst == "UDINT" || dst == "LINT" || dst == "ULINT")
        target = getType(tok_dint);
      else if (dst == "WORD" || dst == "UINT")
        target = getType(tok_word);
      else if (dst == "DWORD")
        target = getType(tok_dword);
      else if (dst == "BYTE" || dst == "USINT")
        target = getType(tok_byte);
      else
        return nullptr; // STRING_TO_REAL/BOOL/...: not supported (stub)
      auto *e = arg("IN", 0);
      if (!e)
        return nullptr;
      StrArg s = strArgOf(location, e);
      mlir::Value acc = sclConst(location, d, 0);
      mlir::Value sign = sclConst(location, d, 1);
      mlir::Value cont = boolConst(location, true);
      int64_t n = s.maxLen();
      for (int64_t i = 0; i < n; ++i) {
        mlir::Value c = byteD(location, s, i);
        mlir::Value isDig = builder.create<AndOp>(
            location,
            builder.create<GreaterEqualOp>(location, c, sclConst(location, d, 48)),
            builder.create<LessEqualOp>(location, c, sclConst(location, d, 57)));
        mlir::Value isMinus = builder.create<EqualOp>(location, c, sclConst(location, d, 45));
        mlir::Value minusAt0 = i == 0 ? isMinus : mlir::Value(boolConst(location, false));
        mlir::Value take = builder.create<AndOp>(
            location, cont,
            builder.create<OrOp>(location, isDig, minusAt0).getResult());
        mlir::Value dval = mux(location, isDig,
                               builder.create<SubOp>(location, c, sclConst(location, d, 48)).getResult(),
                               sclConst(location, d, 0));
        acc = mux(location, take,
                  builder.create<AddOp>(
                      location,
                      builder.create<MulOp>(location, acc, sclConst(location, d, 10)).getResult(),
                      dval).getResult(),
                  acc);
        cont = builder.create<AndOp>(
            location, cont,
            builder.create<OrOp>(location, isDig, minusAt0).getResult());
        sign = mux(location, minusAt0, sclConst(location, d, -1), sign);
      }
      mlir::Value r = builder.create<MulOp>(location, sign, acc).getResult();
      return castToType(location, r, target);
    }

    return nullptr;
  }

  /// SPEC[SCL] IEC 61131-3 ST standard functions (ed.3 table 32/33):
  /// ABS/MAX/MIN/SEL/MUX/LIMIT on numeric types and the unary real functions
  /// SIN/COS/TAN/ASIN/ACOS/ATAN/SQRT/EXP/LN/ABS on REAL. Real functions lower
  /// through awl.real_math (scl_rt_realmath_f, awlsim-exact); the numeric
  /// selection functions lower to arith ops. Returns nullptr when `callee`
  /// is not a standard function.
  mlir::Value mlirGenStdFunction(mlir::Location location,
                                 const FunctionCallAST *expr, StringRef callee) {
    std::string up = std::string(callee.upper());
    // real math function kind -> awl.real_math kind
    static const std::map<std::string, std::string> realFns = {
        {"SIN", "sin"}, {"COS", "cos"}, {"TAN", "tan"}, {"ASIN", "asin"},
        {"ACOS", "acos"}, {"ATAN", "atan"}, {"SQRT", "sqr"}, {"EXP", "exp"},
        {"LN", "ln"}, {"LOG", "ln"}, {"ABS", "abs"}};
    static const std::map<std::string, bool> selFns = {
        {"MAX", true}, {"MIN", false}}; // max=true, min=false

    SmallVector<mlir::Value, 4> args;
    if (failed(mlirGenCallParameters(
            expr, [&](StringRef, mlir::Value v) { args.push_back(v); },
            [&](mlir::Value v) { args.push_back(v); })))
      return nullptr;
    if (args.empty() || !args[0])
      return nullptr;
    auto ctx = builder.getContext();
    mlir::Type t0 = args[0].getType();

    // Unary real functions (operand REAL/LREAL -> awl.real_math, f32).
    if (auto it = realFns.find(up); it != realFns.end() &&
                                      args.size() == 1 &&
                                      t0.isa<RealType>()) {
      auto f32 = mlir::FloatType::getF32(ctx);
      mlir::Value v = args[0];
      if (v.getType() != f32)
        v = builder.create<DialectCastOp>(location, f32, v);
      auto r = builder.create<mlir::awl::RealMathOp>(
          location, f32, builder.getStringAttr(it->second), v);
      if (r.getResult().getType() != t0)
        return builder.create<DialectCastOp>(location, t0, r.getResult());
      return r.getResult();
    }

    // Numeric selection functions on a common integer/float type.
    auto toStd = [&](mlir::Value v, mlir::Type stdTy) -> mlir::Value {
      if (v.getType() == stdTy)
        return v;
      return builder.create<DialectCastOp>(location, stdTy, v);
    };
    auto fromStd = [&](mlir::Value v, mlir::Type sclTy) -> mlir::Value {
      if (v.getType() == sclTy)
        return v;
      return builder.create<DialectCastOp>(location, sclTy, v);
    };

    if (up == "ABS" && args.size() == 1 && t0.isa<IntegerType>()) {
      auto it = t0.cast<IntegerType>();
      auto stdTy = mlir::IntegerType::get(ctx, it.getWidth());
      mlir::Value v = toStd(args[0], stdTy);
      auto zero = builder.create<ConstantOp>(
          location, t0, builder.getIntegerAttr(stdTy, 0));
      mlir::Value z = toStd(zero, stdTy);
      auto lt = builder.create<mlir::arith::CmpIOp>(
          location, mlir::arith::CmpIPredicate::slt, v, z);
      auto neg = builder.create<mlir::arith::SubIOp>(location, z, v);
      auto res = builder.create<mlir::arith::SelectOp>(location, lt.getResult(), neg.getResult(), v).getResult();
      return fromStd(res, t0);
    }
    if (auto it = selFns.find(up); it != selFns.end() && args.size() >= 2) {
      bool isFloat = llvm::any_of(args, [](mlir::Value v) {
        return v.getType().isa<RealType>();
      });
      mlir::Type resTy = t0;
      mlir::Type stdTy;
      if (isFloat)
        stdTy = mlir::FloatType::getF32(ctx);
      else {
        unsigned w = 32;
        for (auto a : args)
          if (auto it2 = a.getType().dyn_cast<IntegerType>())
            w = std::max<unsigned>(w, it2.getWidth());
        stdTy = mlir::IntegerType::get(ctx, w);
      }
      mlir::Value acc = toStd(args[0], stdTy);
      for (size_t i = 1; i < args.size(); ++i) {
        mlir::Value v = toStd(args[i], stdTy);
        // max/min are order-insensitive for values, so arg order is fine.
        if (isFloat)
          acc = it->second
                    ? builder.create<mlir::arith::MaxFOp>(location, acc, v).getResult()
                    : builder.create<mlir::arith::MinFOp>(location, acc, v).getResult();
        else
          acc = it->second
                    ? builder.create<mlir::arith::MaxSIOp>(location, acc, v).getResult()
                    : builder.create<mlir::arith::MinSIOp>(location, acc, v).getResult();
      }
      return fromStd(acc, resTy);
    }
    if (up == "SEL" && args.size() == 3) {
      // SEL(G, IN0, IN1): G ? IN1 : IN0
      mlir::Value g = toStd(args[0], mlir::IntegerType::get(ctx, 1));
      mlir::Type stdTy = mlir::IntegerType::get(ctx, 32);
      if (t0.isa<RealType>())
        stdTy = mlir::FloatType::getF32(ctx);
      else if (auto it2 = args[1].getType().dyn_cast<IntegerType>())
        stdTy = mlir::IntegerType::get(ctx, it2.getWidth());
      mlir::Value in0 = toStd(args[1], stdTy), in1 = toStd(args[2], stdTy);
      auto res = builder.create<mlir::arith::SelectOp>(location, g, in1, in0).getResult();
      return fromStd(res, t0.isa<RealType>() ? t0 : args[1].getType());
    }
    if (up == "LIMIT" && args.size() == 3) {
      // LIMIT(MN, IN, MX) = MIN(MAX(IN, MN), MX)
      bool isFloat = llvm::any_of(args, [](mlir::Value v) {
        return v.getType().isa<RealType>();
      });
      mlir::Type stdTy = isFloat ? (mlir::Type)mlir::FloatType::getF32(ctx)
                                 : (mlir::Type)mlir::IntegerType::get(ctx, 32);
      if (!isFloat)
        for (auto a : args)
          if (auto it2 = a.getType().dyn_cast<IntegerType>())
            stdTy = mlir::IntegerType::get(
                ctx, std::max<unsigned>(
                         it2.getWidth(),
                         stdTy.cast<mlir::IntegerType>().getWidth()));
      mlir::Value mn = toStd(args[0], stdTy), in = toStd(args[1], stdTy),
                  mx = toStd(args[2], stdTy);
      mlir::Value r;
      if (isFloat)
        r = builder.create<mlir::arith::MaxFOp>(location, in, mn).getResult();
      else
        r = builder.create<mlir::arith::MaxSIOp>(location, in, mn).getResult();
      if (isFloat)
        r = builder.create<mlir::arith::MinFOp>(location, r, mx).getResult();
      else
        r = builder.create<mlir::arith::MinSIOp>(location, r, mx).getResult();
      return fromStd(r, args[1].getType());
    }
    if (up == "MUX" && args.size() >= 2) {
      // MUX(K, IN0..INn): nested selects, INn is the ELSE
      mlir::Value k = toStd(args[0], mlir::IntegerType::get(ctx, 32));
      mlir::Type stdTy = mlir::FloatType::getF32(ctx);
      bool isFloat = false;
      for (size_t i = 1; i < args.size(); ++i)
        if (args[i].getType().isa<RealType>())
          isFloat = true;
      if (!isFloat) {
        unsigned w = 32;
        for (size_t i = 1; i < args.size(); ++i)
          if (auto it2 = args[i].getType().dyn_cast<IntegerType>())
            w = std::max<unsigned>(w, it2.getWidth());
        stdTy = mlir::IntegerType::get(ctx, w);
      }
      mlir::Value res = toStd(args.back(), stdTy);
      for (size_t i = args.size() - 1; i >= 1; --i) {
        auto ki = builder.create<ConstantOp>(
            location, args[0].getType(),
            builder.getIntegerAttr(mlir::IntegerType::get(ctx, 32),
                                  (int64_t)(i - 1)));
        auto kiv = toStd(ki, mlir::IntegerType::get(ctx, 32));
        auto eq = builder.create<mlir::arith::CmpIOp>(
            location, mlir::arith::CmpIPredicate::eq, k, kiv);
        res = builder.create<mlir::arith::SelectOp>(location, eq.getResult(),
                                                    toStd(args[i], stdTy), res).getResult();
      }
      return fromStd(res, args[1].getType());
    }
    return nullptr;
  }

  mlir::Value mlirGenFcCall(const FunctionCallAST *expr, StringRef callee,
                            bool allowVoid = false) {
    auto location = loc(expr->loc());
    auto context = builder.getContext();

    // SPEC[SCL] Test-support builtins (awlsim-style oracles in SCL form,
    // used by test/conformance/run_st_traces.py and SCL oracle lit tests):
    //   __ASSERT(<bool expr>) -> awl.assert (id=line, op ==, actual=i64(expr),
    //                            expected=1) -> scl_rt_assert_cmp
    //   __BITS(<value>)       -> bit.cast to the same-width integer (lets the
    //                            trace driver observe REAL bit patterns)
    std::string upTest = std::string(callee.upper());
    if (upTest == "__ASSERT" || upTest == "__BITS") {
      mlir::Value arg;
      if (failed(mlirGenCallParameters(
              expr, [&](StringRef, mlir::Value v) { arg = v; },
              [&](mlir::Value v) { arg = v; })))
        return nullptr;
      if (!arg) {
        emitError(location) << upTest << " needs exactly one argument";
        return nullptr;
      }
      if (upTest == "__BITS") {
        unsigned width;
        if (auto it = arg.getType().dyn_cast<IntegerType>())
          width = it.getWidth();
        else if (auto lt = arg.getType().dyn_cast<LogicalType>())
          width = lt.getWidth();
        else if (arg.getType().isa<RealType>() || arg.getType().isa<TimeType>())
          width = 32;
        else {
          emitError(location) << "__BITS: unsupported argument type "
                              << arg.getType();
          return nullptr;
        }
        return builder
            .create<BitCastOp>(location, IntegerType::get(context, width), arg)
            .getResult();
      }
      auto i64 = mlir::IntegerType::get(context, 64);
      if (arg.getType() != i64)
        arg = builder.create<DialectCastOp>(location, i64, arg);
      auto one = builder.create<ConstantOp>(
          location, i64, builder.getIntegerAttr(i64, 1));
      builder.create<mlir::awl::AssertOp>(
          location,
          builder.getIntegerAttr(builder.getI32Type(),
                                 (int32_t)expr->loc().line),
          builder.getIntegerAttr(builder.getI8Type(), 0), arg,
          one.getResult());
      return nullptr; // statement, no value
    }

    // IEC 61131-3 string function (LEN/CONCAT/...)? Handled before the
    // generic argument collection: these accept positional arguments, and
    // STRING_TO_INT must not be swallowed by the generic _TO_ conversion.
    if (functionResultTypes.find(callee) == functionResultTypes.end())
      if (mlir::Value v = mlirGenStringBuiltin(location, expr, callee))
        return v;

    // get arguments
    SmallVector<mlir::Value, 4> arguments;
    SmallVector<mlir::Attribute, 4> argNames;
    const llvm::StringMap<mlir::Type> *paramTypes = nullptr;
    auto ptIt = callableVarTypes.find(callee);
    if (ptIt == callableVarTypes.end()) {
      // case-insensitive callee lookup (IEC identifiers are case-insensitive)
      for (auto &kv : callableVarTypes)
        if (kv.getKey().equals_insensitive(callee)) {
          ptIt = callableVarTypes.find(kv.getKey());
          break;
        }
    }
    if (ptIt != callableVarTypes.end())
      paramTypes = &ptIt->second;
    auto orderIt = callableVarNames.find(callee);
    size_t posIdx = 0;
    if (failed(mlirGenCallParameters(expr, [&] (StringRef name, mlir::Value value) {
      argNames.push_back(mlir::StringAttr::get(context, name));
      arguments.push_back(value);
      return mlir::success();
    }, [&] (mlir::Value value) {
      arguments.push_back(value);
      // positional argument: name it after the i-th declared parameter so
      // the canonicalizer can reorder/fill omitted defaults by name
      if (orderIt != callableVarNames.end() && posIdx < orderIt->second.size())
        argNames.push_back(mlir::StringAttr::get(context, orderIt->second[posIdx]));
      ++posIdx;
    }, paramTypes)))
      return nullptr;

    auto rtIt = functionResultTypes.find(callee);
    if (rtIt == functionResultTypes.end()) {
      // IEC conversion builtin (INT_TO_WORD, TIME_TO_DINT, ...)?
      if (mlir::Value v = mlirGenBuiltinConversion(location, expr, callee))
        return v;
      // IEC 61131-3 standard functions (ABS/MAX/MIN/SEL/MUX/LIMIT and the
      // real math functions)?
      if (mlir::Value v = mlirGenStdFunction(location, expr, callee))
        return v;
      // TIA DATE_AND_TIME field extractor (DT_DAY(...), ...)?
      if (mlir::Value v = mlirGenDtField(location, expr, callee))
        return v;
      // TIA system functions reading the clock: return the sim epoch
      // (1990-01-01 00:00:00); the virtual clock advances via scl_rt_cycle.
      std::string upCallee = std::string(callee.upper());
      if (upCallee == "RD_SYS_T" || upCallee == "RD_LOC_T" ||
          upCallee == "READ_CLK")
        return builder.create<ConstantDateAndTimeOp>(location, 1990, 1, 1, 0,
                                                      0, 0, 0);
      // Unknown callee (TIA library/system function not in our table, e.g.
      // WRIT_DBL, DataLogCreate): warn and call an external stub declared
      // on the fly, returning i32 (0 at runtime unless the host binds it).
      emitWarning(location) << "unknown function '" << callee
                            << "' called as an external i32 stub";
      SmallVector<mlir::Type, 4> argTys;
      for (auto a : arguments)
        argTys.push_back(a.getType());
      auto fnTy = builder.getFunctionType(argTys, getType(tok_int));
      auto stub = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionOp>(
          theModule, builder.getStringAttr(callee));
      if (!stub) {
        mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
        stub = b.create<FunctionOp>(location, callee, fnTy);
        stub.setPrivate();
      }
      auto argNamesAttr = mlir::ArrayAttr::get(context, argNames);
      return builder.create<CallFcOp>(location, getType(tok_int), callee,
                                      arguments, argNamesAttr);
    }
    mlir::Type resultType = rtIt->second;
    if (resultType.isa<mlir::NoneType>()) {
      if (!allowVoid)
        emitError(location) << "void function '" << callee << "' used as a value";
      return nullptr;
    }

    auto argNamesAttr = mlir::ArrayAttr::get(context, argNames);
    return builder.create<CallFcOp>(location, resultType, callee, arguments, argNamesAttr);
  }

  /// Case-insensitive VariableOp lookup in a function block: IEC 61131-3
  /// (and TIA) identifiers are case-insensitive, so `counter(CU := ...)`
  /// must find the member declared as `cu`.
  VariableOp lookupVariableCI(FunctionBlockOp fb, StringRef name) {
    if (auto v = mlir::dyn_cast_or_null<VariableOp>(
            mlir::SymbolTable::lookupSymbolIn(fb, name)))
      return v;
    for (auto v : fb.getBody().getOps<VariableOp>())
      if (llvm::StringRef(v.getSymName()).equals_insensitive(name))
        return v;
    return nullptr;
  }

  /// Case-insensitive lookup in a callable's parameter type map.
  mlir::Type lookupParamTypeCI(const llvm::StringMap<mlir::Type> *map,
                               StringRef name) {
    if (!map)
      return nullptr;
    if (auto it = map->find(name); it != map->end())
      return it->second;
    for (auto &kv : *map)
      if (kv.getKey().equals_insensitive(name))
        return kv.getValue();
    return nullptr;
  }

  /// If `name` is a local variable of FB-instance type (e.g. a TON_TIME
  /// timer in VAR_TEMP), return the FB symbol it instantiates.
  // NB: returns std::string, not StringRef — unquote() produces a temporary
  // and a StringRef into it would dangle (this was the intermittent garbage
  // "unknown FB '<binary>'" warning in the LogMsg project)
  std::optional<std::string> fbInstanceOf(StringRef name) {
    auto sym = symbolTable.lookup(name);
    mlir::Type t = sym.getType();
    if (!t)
      return std::nullopt;
    if (t.isa<AddressType>())
      t = t.cast<AddressType>().getElementType();
    if (auto idbTy = t.dyn_cast<InstanceDbType>())
      return unquote(idbTy.getFbSymbol());
    return std::nullopt;
  }

  mlir::LogicalResult mlirGenFbCall(const FunctionCallAST *expr, StringRef idb, StringRef fb) {
    auto location = loc(expr->loc());

    auto idbType = InstanceDbType::get(builder.getContext(), fb);
    auto argType = AddressType::get(idbType);
    mlir::Value idbRef;
    auto sym = symbolTable.lookup(idb);
    if (sym.isAddress() || sym.isElement()) {
      // local instance variable (e.g. a timer in VAR_TEMP / an FB member)
      idbRef = sym.isElement() ? mlirGenVariable(location, idb) : sym.getValue();
    } else {
      idbRef = builder.create<GetGlobalOp>(location, argType, idb);
    }
    if (!idbRef) {
      emitError(location, "invalid IDB reference");
      return mlir::failure();
    }
    return mlirGenFbCallOn(expr, idbRef, fb);
  }

  /// Emit the parameter stores and the call.fb for an instance address that
  /// is already resolved (global DB, local variable, or struct member).
  mlir::LogicalResult mlirGenFbCallOn(const FunctionCallAST *expr,
                                      mlir::Value idbRef, StringRef fb) {
    auto location = loc(expr->loc());

    const llvm::StringMap<mlir::Type> *paramTypes = nullptr;
    auto ptIt = callableVarTypes.find(fb);
    if (ptIt != callableVarTypes.end())
      paramTypes = &ptIt->second;
    if (failed(mlirGenCallParameters(expr, [&] (StringRef name, mlir::Value value) {
      auto elem = mlirGenElement(location, idbRef, name);
      if (!elem)
        return mlir::failure();
      builder.create<StoreOp>(location, elem, value);
      return mlir::success();
    }, nullptr, paramTypes,
      [&] (StringRef name, const StringConstantAST *str)
          -> mlir::LogicalResult {
        auto elem = mlirGenElement(location, idbRef, name);
        if (!elem)
          return mlir::failure();
        auto it = byteViewTypes.find(elem);
        if (it == byteViewTypes.end() || !it->second.isa<ArrayType>())
          return mlir::failure(); // not a string member: fall back to value
        return mlirGenStoreStringBytes(location, elem,
                                       it->second.cast<ArrayType>(),
                                       str->getValue());
      })))
        return mlir::failure();

    // Unknown FB (TIA library FB): declare an empty stub so the symbolic
    // call verifies and lowers to a no-op (like the awl.call stubs).
    // Builtin standard FBs are exempt: they lower to scl_rt_*_fb runtime
    // calls (no stub op, so the forced layout is not shadowed).
    bool builtin = !builtinStandardFb(fb).empty();
    if (!builtin &&
        !mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
            theModule, builder.getStringAttr(unquote(fb)))) {
      mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
      auto stub = b.create<FunctionBlockOp>(location, unquote(fb));
      auto *entry = stub.addEntryBlock();
      b.setInsertionPointToEnd(entry);
      b.create<ReturnOp>(location);
      stubFbNames.insert(unquote(fb));
      emitWarning(location) << "unknown FB '" << fb
                            << "' called as an empty stub";
    }
    builder.create<CallFbOp>(location, unquote(fb), idbRef);
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
    // IEC 61131-3 unsigned type names map to the same-width S7 types
    case tok_usint:
      return LogicalType::get(builder.getContext(), 8);
    case tok_uint:
      return LogicalType::get(builder.getContext(), 16);
    case tok_udint:
      return LogicalType::get(builder.getContext(), 32);
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

  /// Fold a constant integer expression (literals, const symbols, and
  /// + - * / over them) — used for DB initial values.
  mlir::LogicalResult foldConstIntAST(const ExpressionAST *expr,
                                      int64_t &value) {
    if (auto *c = llvm::dyn_cast<IntegerConstantAST>(expr)) {
      value = c->getValue();
      return mlir::success();
    }
    if (auto *v = llvm::dyn_cast<SimpleVariableAST>(expr)) {
      auto sym = symbolTable.lookup(v->getName());
      if (!sym.isConst()) {
        // e.g. a TIA library constant like "FC_ID_CHECK_ARRAY" defined in a
        // global DB we do not have: warn and use a benign default
        emitWarning(loc(v->loc())) << "not a constant: " << v->getName()
                                    << ", assuming 1";
        value = 1;
        return mlir::success();
      }
      return foldConstIntAST(sym.getConstValue(), value);
    }
    if (auto *bin = llvm::dyn_cast<BinaryExpressionAST>(expr)) {
      int64_t l = 0, r = 0;
      if (mlir::failed(foldConstIntAST(bin->getLhs(), l)) ||
          mlir::failed(foldConstIntAST(bin->getRhs(), r)))
        return mlir::failure();
      switch (bin->getOp()) {
      case tok_plus: value = l + r; return mlir::success();
      case tok_minus: value = l - r; return mlir::success();
      case tok_times: value = l * r; return mlir::success();
      case tok_divide: case tok_div:
        if (r == 0) { emitError(loc(bin->loc())) << "division by zero in constant"; return mlir::failure(); }
        value = l / r; return mlir::success();
      default: break;
      }
    }
    // unresolvable constant expression (e.g. referencing a library
    // constant): warn and use a benign default instead of failing the file
    emitWarning(loc(expr->loc())) << "constant integer expected, assuming 1";
    value = 1;
    return mlir::success();
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
        emitWarning(location) << "not a constant: " << constName
                              << ", assuming 1";
        value = 1;
        return mlir::success();
      }

      return getConstantInteger(constSymbol.getConstValue(), value);
    })
    .Default([&](auto expr) {
      emitWarning(loc(expr->loc())) << "constant integer expected, assuming 1";
      value = 1;
      return mlir::success();
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
    // Builtin timer FBs (lowered to @scl_rt_timer_fb, no source definition);
    // Builtin standard FBs (also TIA's *_TIME spellings) are instance DBs
    // with a fixed builtin layout (see the lowering's FbLayoutCache).
    if (std::string b = builtinStandardFb(type->getName()); !b.empty())
      return InstanceDbType::get(builder.getContext(), b);
    // Unknown type (TIA library type, or a type defined in another file of
    // a multi-file project): warn and model it as an instance DB, like the
    // builtin timer FBs — brace initializers and instance calls then work.
    emitWarning(loc(type->loc()))
        << "unknown type '" << type->getName()
        << "' treated as an opaque instance DB";
    return InstanceDbType::get(builder.getContext(), unquote(type->getName()));
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

  /// Coerce a condition value to bool (LogicalType width 1); nonzero = true.
  /// Makes e.g. `IF someInt THEN` and unknown-symbol fallbacks work.
  mlir::Value coerceToBool(mlir::Location location, mlir::Value v) {
    auto t = v.getType();
    if (auto lt = mlir::dyn_cast<LogicalType>(t))
      if (lt.getWidth() == 1)
        return v;
    mlir::Attribute zero;
    if (auto it = mlir::dyn_cast<IntegerType>(t))
      zero = builder.getIntegerAttr(builder.getIntegerType(it.getWidth()), 0);
    else if (auto lt = mlir::dyn_cast<LogicalType>(t))
      zero = builder.getIntegerAttr(builder.getIntegerType(lt.getWidth()), 0);
    else if (t.isa<ArrayType>()) {
      // string/array value as a condition: approximate as false
      emitWarning(location) << "string value used as a condition, assuming false";
      return builder.create<ConstantOp>(
          location, getType(tok_bool),
          builder.getIntegerAttr(builder.getI1Type(), 0));
    } else
      return v; // non-scalar: let the verifier report it
    auto zeroC = builder.create<ConstantOp>(location, t, zero);
    return builder.create<UnaryNotOp>(
        location, builder.create<EqualOp>(location, v, zeroC));
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
      condition = coerceToBool(location, condition);
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
    builder.create<ConditionOp>(location, coerceToBool(location, condition), dummy, dummy);
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
    builder.create<UntilOp>(location, coerceToBool(location, until), dummy, dummy);
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
