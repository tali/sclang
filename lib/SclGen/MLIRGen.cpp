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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include "llvm/Support/Process.h"

#include <functional>
#include <numeric>
#include <string>

using namespace mlir::scl;
using namespace sclang;

using llvm::ArrayRef;
using llvm::isa;
using llvm::dyn_cast;
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
  bool isAddress() const { return type && llvm::isa<AddressType>(type); }
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
  MLIRGenImpl(mlir::MLIRContext &context, LangMode mode = LangMode::S7)
      : builder(&context), mode(mode) {}

  /// Source-language mode (ADR 004), from the ModuleAST.
  LangMode mode;
  bool iecMode() const { return mode == LangMode::IEC; }

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
        // stored upper-cased: IEC identifiers are case-insensitive
        fbNames[llvm::StringRef(fb->getIdentifier()).upper()] = true;
        fbNameOrig[llvm::StringRef(fb->getIdentifier()).upper()] =
            std::string(fb->getIdentifier());
      }
    }
    // Second pre-pass: collect function result types and the declared types
    // of all variables per callable, so that call arguments (incl. string
    // literals) can be typed even before the callee is generated.
    for (const auto &unit : *moduleAST) {
      // IEC CONFIGURATION (ADR 004): register the VAR_GLOBAL names as
      // externals resolving into the configuration's global DB.
      if (auto *cfg = llvm::dyn_cast<ConfigurationAST>(unit.get())) {
        // Emit the configuration's global DB + scl.config op now (detached),
        // so VAR_EXTERNAL references resolve while POUs are codegen'd below.
        builder.clearInsertionPoint();
        if (mlir::Operation *cfgOp = mlirGen(cfg)) {
          theModule.push_back(cfgOp);
          for (auto *e : extraOps)
            theModule.push_back(e);
          extraOps.clear();
        }
        if (errorOccurred)
          return nullptr;
        if (cfg->getGlobals())
          for (const auto &sub : cfg->getGlobals()->getDecls()) {
            if (sub->getKind() !=
                DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable)
              continue;
            auto &vardecls =
                llvm::cast<VariableDeclarationSubsectionAST>(*sub);
            for (const auto &value : vardecls.getValues()) {
              auto t = getType(value->getDataType());
              if (!t)
                return nullptr;
              for (const auto &v : value->getVars())
                externalVars[llvm::StringRef(v->getIdentifier()).upper()] = {
                    std::string(cfg->getIdentifier()), t};
            }
          }
      }
      if (auto *fn = llvm::dyn_cast<FunctionAST>(unit.get())) {
        functionResultTypes[fn->getIdentifier()] = getType(fn->getType());
        collectVarTypes(fn->getDeclarations(),
                        callableVarTypes[fn->getIdentifier()],
                        &callableVarNames[fn->getIdentifier()],
                        &callableVarDefaults[fn->getIdentifier()],
                        &callableAllOrder[fn->getIdentifier()],
                        &callableInOut[fn->getIdentifier()],
                        &callableOutputs[fn->getIdentifier()]);
      } else if (auto *ob = llvm::dyn_cast<OrganizationBlockAST>(unit.get())) {
        functionResultTypes[ob->getIdentifier()] = builder.getNoneType();
        collectVarTypes(ob->getDeclarations(),
                        callableVarTypes[ob->getIdentifier()],
                        &callableVarNames[ob->getIdentifier()],
                        &callableVarDefaults[ob->getIdentifier()],
                        &callableAllOrder[ob->getIdentifier()],
                        &callableInOut[ob->getIdentifier()],
                        &callableOutputs[ob->getIdentifier()]);
      } else if (auto *fb = llvm::dyn_cast<FunctionBlockAST>(unit.get())) {
        collectVarTypes(fb->getDeclarations(),
                        callableVarTypes[fb->getIdentifier()],
                        &callableVarNames[fb->getIdentifier()],
                        &callableVarDefaults[fb->getIdentifier()],
                        &callableAllOrder[fb->getIdentifier()],
                        &callableInOut[fb->getIdentifier()],
                        &callableOutputs[fb->getIdentifier()]);
        // IEC 61131-3 ed.3 METHODs: registered as functions "<FB>.<METHOD>"
        // whose first parameter is the instance-DB address.
        for (auto &m : fb->getMethods()) {
          std::string mname =
              (fb->getIdentifier() + "." + m->getIdentifier()).str();
          functionResultTypes[mname] = getType(m->getType());
          collectVarTypes(m->getDeclarations(), callableVarTypes[mname],
                          &callableVarNames[mname], &callableVarDefaults[mname],
                          &callableAllOrder[mname], &callableInOut[mname],
                          &callableOutputs[mname]);
        }
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
      if (unit->getKind() == UnitAST::Unit_Configuration)
        continue; // emitted in the pre-pass (global DB must exist early)
      auto func = mlirGen(unit.get());
      if (!func || errorOccurred)
        return nullptr;
      theModule.push_back(func);
      for (auto *op : extraOps)
        theModule.push_back(op);
      extraOps.clear();
      // Emit queued FB methods at module level (fresh symbol scope).
      for (auto &pm : pendingMethods) {
        // NB: create the op detached (cleared insertion point, like the
        // main loop does for units) so the push_back below does not link it
        // into the module twice.
        builder.clearInsertionPoint();
        auto fn = mlirGenFunctionLike(pm.second->getIdentifier(),
                                      getType(pm.second->getType()),
                                      pm.second->getDeclarations(),
                                      pm.second->getCode(),
                                      loc(pm.second->loc()), pm.first);
        if (!fn || errorOccurred)
          return nullptr;
        theModule.push_back(fn);
      }
      pendingMethods.clear();
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (llvm::sys::Process::GetEnv("SCL_DUMP_PREVERIFY"))
      theModule.print(llvm::errs());
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
  /// Lower bound of IEC 61131-3 subrange types (TYPE SMALL : INT (5..10)),
  /// keyed by upper-cased type name: the default initial value of a variable
  /// of that type is the lower limit (ed.3 6.4.4.4 / table 10).
  llvm::StringMap<int64_t> subrangeMin;

  /// Enumerated-value constants (upper-case name -> value), registered when
  /// an enum TYPE is declared; resolved as a fallback in mlirGenVariable.
  llvm::StringMap<int64_t> enumConsts;

  /// Ops generated alongside a unit but pushed to the module separately
  /// (methods of an FB become their own FunctionOps "<FB>.<METHOD>").
  std::vector<mlir::Operation *> extraOps;

  /// Per-instance storage cells for the implicit EN/ENO pins of FB calls
  /// (keyed by instance identity: global-DB symbol name, else the address's
  /// defining op), so `inst.ENO` reads back what the last call wrote.
  llvm::StringMap<mlir::Value> enoCells;

  /// Methods of the FB currently being generated; emitted as module-level
  /// FunctionOps by the main loop after the FB's symbol scope is closed.
  std::vector<std::pair<FunctionBlockOp, const MethodAST *>> pendingMethods;
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
  /// Declared constant initializers (defaults) per callable parameter, used
  /// to pad omitted formal-call arguments (IEC ed.3 6.6.1.7.1).
  llvm::StringMap<llvm::StringMap<std::pair<bool, int64_t>>> callableVarDefaults;
  /// All declared parameters per callable in declaration order (inputs,
  /// in/outs, outputs) — the full argument list of an FC call.
  llvm::StringMap<llvm::SmallVector<std::string, 8>> callableAllOrder;
  /// VAR_IN_OUT / VAR_OUTPUT parameter names per callable (passed by address
  /// for FCs; written back after the call for FBs).
  llvm::StringMap<std::set<std::string>> callableInOut, callableOutputs;

  /// Collect the declared types of all variables of a declaration section
  /// into `map`, keyed by variable name.
  void collectVarTypes(const DeclarationSectionAST *declarations,
                       llvm::StringMap<mlir::Type> &map) {
    collectVarTypes(declarations, map, nullptr);
  }
  void collectVarTypes(const DeclarationSectionAST *declarations,
                       llvm::StringMap<mlir::Type> &map,
                       llvm::SmallVector<std::string, 8> *order,
                       llvm::StringMap<std::pair<bool, int64_t>> *defaults =
                           nullptr,
                       llvm::SmallVector<std::string, 8> *allOrder = nullptr,
                       std::set<std::string> *inoutNames = nullptr,
                       std::set<std::string> *outputNames = nullptr) {
    if (!declarations)
      return;
    for (const auto &decl : declarations->getDecls()) {
      if (decl->getKind() !=
          DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable)
        continue;
      const auto &vardecls =
          llvm::cast<VariableDeclarationSubsectionAST>(*decl);
      // Only inputs and in/outs are call arguments (outputs are written
      // back via `=>`, not passed as values), so the declared-order list
      // used for argument padding covers just those.
      if (vardecls.getKind() ==
              VariableDeclarationSubsectionAST::VarExternal ||
          vardecls.getKind() == VariableDeclarationSubsectionAST::VarGlobal)
        continue; // IEC externals/globals are not locals (ADR 004)
      bool isCallArg = vardecls.getKind() ==
                           VariableDeclarationSubsectionAST::VarInput ||
                       vardecls.getKind() ==
                           VariableDeclarationSubsectionAST::VarInOut;
      for (const auto &value : vardecls.getValues()) {
        auto type = getType(value->getDataType());
        if (!type)
          continue;
        for (const auto &var : value->getVars()) {
          map[var->getIdentifier()] = type;
          if (order && isCallArg)
            order->push_back(std::string(var->getIdentifier()));
          if (allOrder && (isCallArg || vardecls.getKind() ==
                               VariableDeclarationSubsectionAST::VarOutput))
            allOrder->push_back(std::string(var->getIdentifier()));
          if (inoutNames && vardecls.getKind() ==
                               VariableDeclarationSubsectionAST::VarInOut)
            inoutNames->insert(std::string(var->getIdentifier()));
          if (outputNames && vardecls.getKind() ==
                                VariableDeclarationSubsectionAST::VarOutput)
            outputNames->insert(std::string(var->getIdentifier()));
          if (defaults && value->getInitializer().has_value() &&
              !isCompositeInitAST(value->getInitializer().value())) {
            int64_t v = 0;
            if (mlir::succeeded(foldConstIntAST(value->getInitializer().value(), v)))
              (*defaults)[var->getIdentifier()] = {true, v};
          }
        }
      }
    }
  }
  /// Names of all function blocks in the module (for type resolution).
  llvm::StringMap<bool> fbNames;
  /// upper-cased FB name -> original spelling (MLIR symbols are
  /// case-sensitive; IEC identifiers are not).
  llvm::StringMap<std::string> fbNameOrig;
  /// IEC VAR_EXTERNAL names (uppercased) -> (configuration's global DB name,
  /// declared type). Resolved in mlirGenVariable to members of that DB
  /// (ADR 004).
  llvm::StringMap<std::pair<std::string, mlir::Type>> externalVars;
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
        .Case<ConfigurationAST>([&](auto cfg) { return mlirGen(cfg); })
        .Default([&](auto unit) {
          emitError(loc(unit->loc()))
              << "MLIR codegen encountered an unhandled unit kind '"
              << Twine(unit->getKind()) << "'";
          return nullptr;
        });
  }

  /// Append all the types and names of a variable declaration to an array
  /// Like addVariables, but for by-reference parameters (VAR_IN_OUT /
  /// VAR_OUTPUT of a function): the argument is the address of the caller's
  /// variable, and the callee reads/writes through it directly.
  void addAddrVariables(const VariableDeclarationSubsectionAST &decls,
                        std::vector<mlir::Type> &types,
                        std::vector<llvm::StringRef> &names,
                        std::vector<bool> &isAddr) {
    const auto &values = decls.getValues();
    for (const auto &decl : values) {
      auto type = getType(decl->getDataType());
      for (const auto &var : decl->getVars()) {
        types.push_back(AddressType::get(type));
        names.push_back(var->getIdentifier());
        isAddr.push_back(true);
      }
    }
  }

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

  /// Byte size of an SCL type (packed, big-endian layout — mirrors byteSize
  /// in LowerToStd.cpp, which computes the instance layouts).
  static int64_t sclByteSize(mlir::Type t) {
    if (auto it = llvm::dyn_cast<IntegerType>(t))
      return (it.getWidth() + 7) / 8;
    if (auto lt = llvm::dyn_cast<LogicalType>(t))
      return (lt.getWidth() + 7) / 8;
    if (llvm::isa<RealType>(t))
      return 4;
    if (auto at = llvm::dyn_cast<ArrayType>(t)) {
      int64_t count = 1;
      for (auto &dim : at.getDimensions())
        count *= dim.second - dim.first + 1;
      return count * sclByteSize(at.getElementType());
    }
    if (auto st = llvm::dyn_cast<StructType>(t)) {
      int64_t size = 0;
      for (auto elem : st.getElementTypes())
        size += sclByteSize(elem);
      return size;
    }
    if (llvm::isa<S5TimeType>(t) || llvm::isa<DateType>(t))
      return 2;
    if (llvm::isa<TimeType>(t) || llvm::isa<TimeOfDayType>(t))
      return 4;
    if (llvm::isa<DateAndTimeType>(t))
      return 8;
    return 0;
  }

  /// Is this initializer an array/structure literal (vs. a scalar value)?
  static bool isCompositeInitAST(const ExpressionAST *e) {
    return llvm::isa<ExpressionListAST>(e) || llvm::isa<RepeatedConstantAST>(e);
  }

  /// Flatten an array-literal expression into its element expressions,
  /// expanding `n(v)` repetition factors. Nested lists are flattened too
  /// (multi-dimensional literals are row-major flat) — used for folding.
  static void flattenInitList(const ExpressionAST *e,
                              std::vector<const ExpressionAST *> &out) {
    if (auto *list = llvm::dyn_cast<ExpressionListAST>(e)) {
      for (auto &v : list->getValues())
        flattenInitList(v.get(), out);
      return;
    }
    if (auto *rep = llvm::dyn_cast<RepeatedConstantAST>(e)) {
      for (int32_t i = 0; i < rep->getRepetitions(); ++i)
        out.push_back(rep->getValue());
      return;
    }
    out.push_back(e);
  }

  /// Direct elements of an array literal: nested lists stay intact (they
  /// are sub-array literals), only `n(v)` repetitions are expanded.
  static void initListElements(const ExpressionAST *e,
                               std::vector<const ExpressionAST *> &out) {
    if (auto *list = llvm::dyn_cast<ExpressionListAST>(e)) {
      for (auto &v : list->getValues()) {
        if (auto *rep = llvm::dyn_cast<RepeatedConstantAST>(v.get()))
          for (int32_t i = 0; i < rep->getRepetitions(); ++i)
            out.push_back(rep->getValue());
        else
          out.push_back(v.get());
      }
      return;
    }
    if (auto *rep = llvm::dyn_cast<RepeatedConstantAST>(e)) {
      for (int32_t i = 0; i < rep->getRepetitions(); ++i)
        out.push_back(rep->getValue());
      return;
    }
    out.push_back(e);
  }

  /// Fold a constant composite initializer (array/struct literal) to the
  /// big-endian bit pattern of the whole value, for cold-start baking into
  /// instance DBs (scl.init_bits on VariableOp, consumed by the lowering).
  std::optional<llvm::APInt> foldConstCompositeAST(const ExpressionAST *e,
                                                   mlir::Type t) {
    if (auto at = llvm::dyn_cast<ArrayType>(t)) {
      int64_t total = sclByteSize(t);
      llvm::APInt result(total * 8, 0);
      // Shallow elements: a multi-dim array carries all dimensions in one
      // ArrayType (flat row-major literal), while a nested list is a
      // sub-array/struct literal folded recursively.
      std::vector<const ExpressionAST *> elems;
      initListElements(e, elems);
      int64_t elemBytes = sclByteSize(at.getElementType());
      unsigned idx = 0;
      for (auto *el : elems) {
        auto bits = foldConstCompositeAST(el, at.getElementType());
        if (!bits || bits->getBitWidth() != (unsigned)(elemBytes * 8))
          return std::nullopt;
        unsigned shift = (unsigned)((total - (idx + 1) * elemBytes) * 8);
        result |= bits->zext(total * 8) << shift;
        ++idx;
      }
      return result;
    }
    if (auto st = llvm::dyn_cast<StructType>(t)) {
      // Structure literal: `(x := 1, y := 10)` (named) or `(1, 10)`
      // (positional), as an ExpressionListAST.
      std::vector<const ExpressionAST *> elems;
      initListElements(e, elems);
      auto names = st.getElementNames();
      auto types = st.getElementTypes();
      llvm::APInt result(sclByteSize(t) * 8, 0);
      int64_t off = 0;
      for (unsigned i = 0; i < types.size(); ++i) {
        int64_t msize = sclByteSize(types[i]);
        const ExpressionAST *me = nullptr;
        if (i < elems.size()) {
          me = elems[i];
          // named element `name := v`?
          if (auto *bin = llvm::dyn_cast<BinaryExpressionAST>(me))
            if (bin->getOp() == tok_assignment)
              me = bin->getRhs();
        }
        if (me) {
          auto bits = foldConstCompositeAST(me, types[i]);
          if (!bits || bits->getBitWidth() != (unsigned)(msize * 8))
            return std::nullopt;
          result |= bits->zext(result.getBitWidth()) << ((sclByteSize(t) - off - msize) * 8);
        }
        off += msize;
      }
      return result;
    }
    // scalar element
    if (auto *rc = llvm::dyn_cast<RealConstantAST>(e)) {
      float f = rc->getValue();
      uint32_t bits;
      std::memcpy(&bits, &f, 4);
      return llvm::APInt(32, bits);
    }
    if (auto *sc = llvm::dyn_cast<StringConstantAST>(e)) {
      // string literal: zero-padded bytes, big-endian order
      std::string v = sc->getValue().str();
      int64_t bytes = sclByteSize(t);
      llvm::APInt result(bytes * 8, 0);
      for (size_t i = 0; i < v.size() && i < (size_t)bytes; ++i)
        result |= llvm::APInt(bytes * 8, (uint8_t)v[i])
                  << ((bytes - 1 - (int64_t)i) * 8);
      return result;
    }
    int64_t iv = 0;
    if (mlir::failed(foldConstIntAST(e, iv)))
      return std::nullopt;
    int64_t bytes = sclByteSize(t);
    if (bytes <= 0)
      return std::nullopt;
    return llvm::APInt(bytes * 8, (uint64_t)iv, true);
  }

  /// Emit per-element stores for a composite initializer into a plain
  /// (non-byte-view) variable address — used for local/temp variables.
  mlir::LogicalResult emitCompositeInit(mlir::Location location,
                                        mlir::Value addr, mlir::Type t,
                                        const ExpressionAST *e) {
    if (auto at = llvm::dyn_cast<ArrayType>(t)) {
      std::vector<const ExpressionAST *> elems;
      initListElements(e, elems);
      auto dims = at.getDimensions();
      auto intTy = getType(tok_int);
      // innermost element type (a multi-dim ArrayType carries all dims and
      // the scalar element type)
      mlir::Type elemTy = at.getElementType();
      unsigned flat = 0;
      for (auto *el : elems) {
        // decompose the flat index into per-dimension indices (row-major:
        // the last dimension varies fastest)
        SmallVector<mlir::Value, 2> indices;
        int64_t rem = flat;
        for (int d = (int)dims.size() - 1; d >= 0; --d) {
          int64_t count = dims[d].second - dims[d].first + 1;
          int64_t idx = count > 0 ? rem % count : 0;
          rem = count > 0 ? rem / count : 0;
          indices.insert(indices.begin(),
                         intConst(location, intTy, 16, (int64_t)(dims[d].first + idx))
                             .getResult());
        }
        // Mirror mlirGen(IndexedVariableAST): a plain array base yields a
        // typed element address (no attr); a byte-view base (array member of
        // a FB/struct) yields an i8 view with the array type recorded.
        mlir::Type baseElem = llvm::cast<AddressType>(addr.getType()).getElementType();
        bool byteBase = llvm::isa<IntegerType>(baseElem) &&
                        llvm::cast<IntegerType>(baseElem).getWidth() == 8;
        mlir::TypeAttr attr;
        mlir::Type resultElem;
        if (byteBase) {
          resultElem = baseElem;
          attr = mlir::TypeAttr::get(at);
        } else {
          resultElem = elemTy;
        }
        auto elemAddr = builder.create<GetIndexOp>(
            location, AddressType::get(resultElem), addr, indices, attr);
        if (byteBase)
          byteViewTypes[elemAddr] = elemTy;
        if (mlir::failed(emitCompositeInit(location, elemAddr, elemTy, el)))
          return mlir::failure();
        ++flat;
      }
      return mlir::success();
    }
    if (llvm::dyn_cast<StructType>(t)) {
      std::vector<const ExpressionAST *> elems;
      initListElements(e, elems);
      auto st = llvm::cast<StructType>(t);
      auto names = st.getElementNames();
      unsigned pos = 0;
      for (auto *el : elems) {
        std::string mname;
        const ExpressionAST *rhs = el;
        if (auto *bin = llvm::dyn_cast<BinaryExpressionAST>(el))
          if (bin->getOp() == tok_assignment) {
            if (auto *sv = llvm::dyn_cast<SimpleVariableAST>(bin->getLhs()))
              mname = sv->getName().str();
            rhs = bin->getRhs();
          }
        if (mname.empty() && pos < names.size())
          mname = names[pos].getValue().str();
        ++pos;
        auto elemAddr = mlirGenElement(location, addr, mname);
        if (!elemAddr)
          return mlir::failure();
        auto it = byteViewTypes.find(elemAddr);
        mlir::Type mty = it != byteViewTypes.end()
                             ? it->second
                             : st.getElementType(mname);
        if (llvm::dyn_cast<ArrayType>(mty) || llvm::dyn_cast<StructType>(mty)) {
          if (mlir::failed(emitCompositeInit(location, elemAddr, mty, rhs)))
            return mlir::failure();
        } else {
          auto val = mlirGenRValue(rhs);
          if (!val)
            return mlir::failure();
          if (val.getType() != mty)
            val = builder.create<DialectCastOp>(location, mty, val);
          builder.create<StoreOp>(location, elemAddr, val);
        }
      }
      return mlir::success();
    }
    // scalar
    auto val = mlirGenRValue(e);
    if (!val)
      return mlir::failure();
    if (val.getType() != t)
      val = builder.create<DialectCastOp>(location, t, val);
    builder.create<StoreOp>(location, addr, val);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(const VariableDeclarationSubsectionAST *decls,
                              bool isInput, bool isOutput) {
    for (const auto &decl : decls->getValues()) {
      auto type = getType(decl->getDataType());
      if (!type)
        return mlir::failure();
      const ExpressionAST *initAST =
          decl->getInitializer().has_value()
              ? decl->getInitializer().value()
              : nullptr;
      mlir::Value init = nullptr;
      if (initAST && !isCompositeInitAST(initAST)) {
        init = mlirGen(initAST);
        if (!init)
          return mlir::failure();
      } else if (!initAST) {
        // no initializer: a subrange-typed member starts at the lower limit
        init = subrangeDefault(loc(decl->loc()), decl->getDataType(), type);
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
        if (initAST)
          if (auto *sc = llvm::dyn_cast<StringConstantAST>(initAST))
            varOp->setAttr("scl.init_str", builder.getStringAttr(sc->getValue()));
        // SPEC[SCL] IEC 61131-3 array/struct initializers
        // (VAR a : ARRAY[1..3] OF INT := [7, 8, 9];): folded to the
        // big-endian bit pattern of the whole value and baked into the
        // instance DB cold start via scl.init_bits (see FbLayoutCache).
        if (initAST && isCompositeInitAST(initAST)) {
          if (auto bits = foldConstCompositeAST(initAST, type)) {
            varOp->setAttr("scl.init_bits",
                           builder.getIntegerAttr(
                               builder.getIntegerType(bits->getBitWidth()),
                               *bits));
          } else {
            emitWarning(location)
                << "non-constant array/struct initializer of '" << name
                << "' is ignored (member starts zeroed)";
          }
        }
      }
    }
    return mlir::success();
  }

  /// The default initial value of a scalar type (IEC 61131-3 ed.3 table 12:
  /// numeric 0, BOOL FALSE, TIME 0 ms, DATE D#1990-01-01, TOD midnight,
  /// DT 1990-01-01-00:00:00). Returns nullptr for composite types.
  mlir::Value defaultInitValue(mlir::Location location, mlir::Type type) {
    auto *ctx = builder.getContext();
    auto zeroInt = [&](int width) {
      return intConst(location, type, width, 0).getResult();
    };
    if (auto it = llvm::dyn_cast<IntegerType>(type))
      return zeroInt(it.getWidth());
    if (auto lt = llvm::dyn_cast<LogicalType>(type))
      return zeroInt(lt.getWidth());
    if (llvm::isa<RealType>(type))
      return builder
          .create<ConstantOp>(location, type,
                              builder.getFloatAttr(builder.getF32Type(), 0.0))
          .getResult();
    if (llvm::isa<S5TimeType>(type) || llvm::isa<DateType>(type))
      return zeroInt(16);
    if (llvm::isa<TimeType>(type) || llvm::isa<TimeOfDayType>(type))
      return zeroInt(32);
    if (llvm::isa<DateAndTimeType>(type))
      return builder.create<ConstantDateAndTimeOp>(location, 1990, 1, 1, 0, 0,
                                                   0, 0)
          .getResult();
    return nullptr; // arrays/structs/strings: no scalar default
  }

  /// Allocate the temp/var storage of a function-like's local variables.
  /// `defaultInit` (VAR sections of FUNCTIONs/OBs/METHODs, ed.3 6.6.2.1:
  /// functions have no internal state, their VAR variables are initialised
  /// at each invocation) stores the type's default value when the declaration
  /// has no explicit initializer; VAR_TEMP stays uninitialised.
  mlir::LogicalResult mlirGenTempVar(const VariableDeclarationSubsectionAST *decls,
                                     bool defaultInit = false) {
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
          if (isCompositeInitAST(init.value())) {
            if (mlir::failed(emitCompositeInit(location, varStorage, type,
                                               init.value())))
              return mlir::failure();
          } else {
            auto value = mlirGen(init.value());
            if (!value)
              return mlir::failure();
            builder.create<StoreOp>(location, varStorage, value);
          }
        } else if (defaultInit) {
          mlir::Value zero = defaultInitValue(location, type);
          if (mlir::Value sr = subrangeDefault(location, decl->getDataType(), type))
            zero = sr;
          if (zero)
            builder.create<StoreOp>(location, varStorage, zero);
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
        if (llvm::isa<ArrayType>(type))
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
      case VariableDeclarationSubsectionAST::VarExternal:
      case VariableDeclarationSubsectionAST::VarGlobal:
        // IEC VAR_EXTERNAL/VAR_GLOBAL: no local symbol — names resolve into
        // the configuration's global DB via externalVars (ADR 004).
        return mlir::success();
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
                                 mlir::Location location,
                                 FunctionBlockOp fbCtx = nullptr) {
    functionName = identifier;
    enoCells.clear();
    functionHasReturnValue = !llvm::isa<mlir::NoneType>(retType);
    labelBlocks.clear();
    std::string name(identifier);
    // A METHOD of an FB becomes a function "<FB>.<METHOD>" with the
    // instance-DB address as its first ($self) parameter.
    mlir::Type selfAddrTy;
    if (fbCtx) {
      name = (fbCtx.getName() + "." + identifier).str();
      auto idb = InstanceDbType::get(builder.getContext(), fbCtx.getName());
      selfAddrTy = AddressType::get(idb);
    }

    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<StringRef, VariableSymbol> var_scope(
        symbolTable);

    std::vector<mlir::Type> input_types;
    std::vector<llvm::StringRef> input_names;
    std::vector<bool> input_is_addr;
    if (selfAddrTy) {
      input_types.push_back(selfAddrTy);
      input_names.push_back("$self");
      input_is_addr.push_back(true); // by-address: use the block arg directly
    }
    std::vector<mlir::Type> output_types;
    std::vector<llvm::StringRef> output_names;
    // (subsection, isVar): VAR sections of function-likes get default-initialised
    // per call (ed.3 6.6.2.1), VAR_TEMP stays uninitialised.
    std::vector<std::pair<const VariableDeclarationSubsectionAST *, bool>>
        tempvar;
    // register function result as output variable (under the plain,
    // unqualified method name)
    if (functionHasReturnValue) {
      output_types.push_back(retType);
      output_names.push_back(identifier);
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
        case VariableDeclarationSubsectionAST::VarInput: {
          size_t before = input_types.size();
          addVariables(vardecls, input_types, input_names);
          input_is_addr.resize(input_types.size(), false);
          (void)before;
          break;
        }
        case VariableDeclarationSubsectionAST::VarOutput:
        case VariableDeclarationSubsectionAST::VarInOut:
          // FC outputs and in/outs are passed by reference: the argument is
          // the address of the caller's variable (`out => var` binds the
          // output; `inout := var` binds both directions; an unbound output
          // gets a fresh temp cell at the call site). FB outputs are IDB
          // members and unaffected.
          addAddrVariables(vardecls, input_types, input_names, input_is_addr);
          break;
        case VariableDeclarationSubsectionAST::Var:
          tempvar.push_back({&vardecls, /*isVar=*/true});
          break;
        case VariableDeclarationSubsectionAST::VarTemp:
          tempvar.push_back({&vardecls, /*isVar=*/false});
          break;
        case VariableDeclarationSubsectionAST::VarExternal:
        case VariableDeclarationSubsectionAST::VarGlobal:
          break; // resolved via externalVars, no local symbol (ADR 004)
        }
      }
    }
    // Create an MLIR function. SCL projects may contain forward
    // declarations (FUNCTION x ... BEGIN END_FUNCTION in a separate file)
    // that are later defined with a real body: if a FunctionOp with this
    // name already exists, the new definition replaces it (and a forward
    // declaration arriving after the definition is skipped).
    auto func_type = builder.getFunctionType(input_types, output_types);
    FunctionOp prev;
    for (auto f : theModule.getOps<FunctionOp>())
      if (f.getName() == name) { prev = f; break; }
    if (prev) {
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
      bool isAddr = argIndex < (int)input_is_addr.size() && input_is_addr[argIndex];
      auto addressType = isAddr ? llvm::cast<AddressType>(type) : AddressType::get(type);
      if (isAddr) {
        // by-reference parameter: use the caller's address directly
        if (failed(declare(location, name, addressType, value)))
          return nullptr;
      } else {
        auto varStorage = builder.create<TempVariableOp>(
            location, addressType, builder.getStringAttr(name));
        if (failed(declare(location, name, addressType, varStorage)))
          return nullptr;
        builder.create<StoreOp>(location, varStorage, value);
      }
      auto nameAttr = builder.getStringAttr(name);
      function.setArgAttr(argIndex, nameId, nameAttr);
      argIndex++;
    }

    // prologue: stack space for temporary variables (VAR initialised per call)
    for (const auto &[subsection, isVar] : tempvar) {
      if (failed(mlirGenTempVar(subsection, isVar)))
        return nullptr;
    }

    // A METHOD sees the FB's members as instance elements (via $self).
    if (fbCtx)
      for (auto v : fbCtx.getBody().getOps<VariableOp>())
        if (failed(declare(location, v.getSymName(), v.getType(), nullptr)))
          return nullptr;

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
    enoCells.clear();
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

    // IEC 61131-3 ed.3 METHODs: queued for emission as module-level
    // FunctionOps "<FB>.<METHOD>" (taking the instance-DB address as $self)
    // by the main loop, once this FB's symbol scope is closed.
    for (auto &m : fb->getMethods())
      pendingMethods.push_back({function, m.get()});

    return function;
  }

  // SPEC[ST] IEC 61131-3 CONFIGURATION (ed.3 6.8): VAR_GLOBAL becomes a
  // global data block named after the configuration; RESOURCE/TASK/PROGRAM
  // bindings are kept as a scl.config metadata op (ADR 004). VAR_EXTERNAL
  // members of POUs resolve into that DB (see externalVars).
  mlir::Operation *mlirGen(const ConfigurationAST *cfg) {
    auto location = loc(cfg->loc());
    auto context = builder.getContext();
    StringRef name = cfg->getIdentifier();
    mlir::Type type;
    llvm::SmallVector<mlir::StringAttr, 4> names;
    llvm::SmallVector<mlir::Type, 4> types;
    llvm::SmallVector<mlir::NamedAttribute, 4> initEntries;
    if (auto *globals = cfg->getGlobals()) {
      for (const auto &sub : globals->getDecls()) {
        if (sub->getKind() !=
            DeclarationSubsectionAST::DeclarationASTKind::Decl_Variable)
          continue;
        auto &vardecls = llvm::cast<VariableDeclarationSubsectionAST>(*sub);
        for (const auto &value : vardecls.getValues()) {
          auto t = getType(value->getDataType());
          if (!t)
            return nullptr;
          for (const auto &var : value->getVars()) {
            names.push_back(builder.getStringAttr(var->getIdentifier()));
            types.push_back(t);
            if (value->getInitializer()) {
              int64_t iv = 0;
              if (mlir::succeeded(
                      foldConstIntAST(value->getInitializer().value(), iv)))
                initEntries.push_back(builder.getNamedAttr(
                    var->getIdentifier(),
                    builder.getIntegerAttr(builder.getI64Type(), iv)));
            }
          }
        }
      }
    }
    type = StructType::get(names, types);
    auto db = builder.create<DataBlockOp>(
        location, mlir::TypeAttr::get(type),
        mlir::StringAttr::get(context, name),
        initEntries.empty() ? mlir::DictionaryAttr()
                            : mlir::DictionaryAttr::get(context, initEntries));
    // Metadata op with the RESOURCE/TASK/PROGRAM bindings.
    llvm::SmallVector<mlir::Attribute, 4> taskAttrs, progAttrs;
    for (auto &t : cfg->getTasks())
      taskAttrs.push_back(builder.getStringAttr(t));
    for (auto &p : cfg->getPrograms())
      progAttrs.push_back(builder.getStringAttr(p));
    auto configOp = builder.create<ConfigOp>(
        location, mlir::StringAttr::get(context, (name + "_config").str()),
        mlir::ArrayAttr::get(context, taskAttrs),
        mlir::ArrayAttr::get(context, progAttrs));
    extraOps.push_back(configOp);
    return db;
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
    if (auto *sr = llvm::dyn_cast<SubrangeDataTypeSpecAST>(udt->getType())) {
      int64_t min = 0;
      if (mlir::succeeded(foldConstIntAST(sr->getMin(), min)))
        subrangeMin[llvm::StringRef(udt->getIdentifier()).upper()] = min;
    }
    return mlir::success();
  }

  /// If `spec` names a subrange type, the default initial value of a variable
  /// of that type: the lower limit as a constant of the (base) storage type.
  mlir::Value subrangeDefault(mlir::Location location,
                              const DataTypeSpecAST *spec, mlir::Type type) {
    auto *udt = llvm::dyn_cast<UserDefinedTypeIdentifierAST>(spec);
    if (!udt)
      return nullptr;
    auto it = subrangeMin.find(llvm::StringRef(udt->getName()).upper());
    if (it == subrangeMin.end())
      return nullptr;
    if (auto it2 = llvm::dyn_cast<IntegerType>(type))
      return builder
          .create<ConstantOp>(location, type,
                              builder.getIntegerAttr(
                                  builder.getIntegerType(it2.getWidth()),
                                  it->second))
          .getResult();
    if (auto lt = llvm::dyn_cast<LogicalType>(type))
      return builder
          .create<ConstantOp>(location, type,
                              builder.getIntegerAttr(
                                  builder.getIntegerType(lt.getWidth()),
                                  it->second))
          .getResult();
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
    if (auto it = llvm::dyn_cast<IntegerType>(t))
      return (it.getWidth() + 7) / 8;
    if (auto lt = llvm::dyn_cast<LogicalType>(t))
      return (lt.getWidth() + 7) / 8;
    if (llvm::isa<RealType>(t))
      return 4;
    if (llvm::isa<S5TimeType, DateType>(t))
      return 2;
    if (llvm::isa<TimeType, DateAndTimeType>(t))
      return llvm::isa<TimeType>(t) ? 4 : 8;
    if (auto at = llvm::dyn_cast<ArrayType>(t)) {
      int64_t n = 1;
      for (auto d : at.getDimensions())
        n *= d.second - d.first + 1;
      return n * byteSizeOf(at.getElementType());
    }
    if (auto st = llvm::dyn_cast<StructType>(t)) {
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
    if (auto at = llvm::dyn_cast<AddressType>(t))
      t = at.getElementType();
    return llvm::dyn_cast<ArrayType>(t);
  }

  /// Array type behind an address value (direct element type or byte view).
  ArrayType arrayTypeOfAddress(mlir::Value addr) {
    if (!addr || !llvm::isa<AddressType>(addr.getType()))
      return nullptr;
    auto elem = llvm::cast<AddressType>(addr.getType()).getElementType();
    if (auto at = llvm::dyn_cast<ArrayType>(elem))
      return at;
    auto it = byteViewTypes.find(addr);
    if (it != byteViewTypes.end())
      return llvm::dyn_cast<ArrayType>(it->second);
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
    if (!llvm::isa<IntegerType>(elemTy) ||
        llvm::cast<IntegerType>(elemTy).getWidth() != 8) {
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
    if (!llvm::isa<IntegerType>(elemTy) ||
        llvm::cast<IntegerType>(elemTy).getWidth() != 8) {
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
    auto arrTy = llvm::dyn_cast<ArrayType>(it->second);
    if (!arrTy || !llvm::isa<IntegerType>(arrTy.getElementType()) ||
        llvm::cast<IntegerType>(arrTy.getElementType()).getWidth() != 8) {
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
    auto baseElem = llvm::cast<AddressType>(lhs.getType()).getElementType();
    // A string member (byte view of an array type) takes the whole literal;
    // only a plain char target gets just the first byte (char := 'A').
    ArrayType arrTy;
    if (auto at = llvm::dyn_cast<ArrayType>(baseElem)) {
      arrTy = at;
    } else {
      auto it = byteViewTypes.find(lhs);
      if (it != byteViewTypes.end() && llvm::isa<ArrayType>(it->second))
        arrTy = llvm::cast<ArrayType>(it->second);
    }
    if (!arrTy) {
      // char := 'A': a string literal assigns its first byte to a char
      if (auto it8 = llvm::dyn_cast<IntegerType>(baseElem)) {
        if (it8.getWidth() == 8 && !str->getValue().empty()) {
          auto byteVal = intConst(location, baseElem, 8, (uint8_t)str->getValue()[0]);
          builder.create<StoreOp>(location, lhs, byteVal);
          return mlir::success();
        }
      }
      emitError(location) << "string literal assigned to non-string target";
      return mlir::failure();
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
      bool lhsIsChar = llvm::isa<IntegerType>(lhsArr.getElementType()) &&
                       llvm::cast<IntegerType>(lhsArr.getElementType()).getWidth() == 8;
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
        if (!llvm::isa<IntegerType>(elemTy)) {
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
                          llvm::isa<IntegerType>(elemTy)
                              ? llvm::cast<IntegerType>(elemTy).getWidth()
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
              if (llvm::isa<LogicalType>(bitVal.getType()))
                cast.setZextAttr(builder.getUnitAttr());
              bitVal = cast.getResult();
            }
            auto scaleC = intConst(location, i8, 8, (int64_t)(1 << bit));
            auto set = builder.create<MulOp>(location, bitVal, scaleC.getResult());
            auto byteVal = builder.create<LoadOp>(location, byteTy, lhs).getResult();
            auto byteI8 = builder.create<IntegerCastOp>(location, i8, byteVal);
            byteI8.setZextAttr(builder.getUnitAttr());
            auto maskC = intConst(location, i8, 8, (int64_t)(uint8_t)~(1 << bit));
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
    auto lhsElem = llvm::cast<AddressType>(lhs.getType()).getElementType();
    auto bvIt = byteViewTypes.find(lhs);
    mlir::Type declared = bvIt != byteViewTypes.end() ? bvIt->second : lhsElem;
    // Implicit integer conversion on width mismatch. byte/word/dword are
    // LogicalType (not IntegerType), so cast to an IntegerType of the same
    // width; storing a wider value into a byte view would otherwise write
    // past the view and read back the high byte.
    auto widthOfTy = [](mlir::Type t) -> int {
      if (auto it = llvm::dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto it = llvm::dyn_cast<mlir::IntegerType>(t))
        return it.getWidth();
      if (auto lt = llvm::dyn_cast<LogicalType>(t))
        return lt.getWidth();
      return -1;
    };
    int dw = widthOfTy(declared), rw = widthOfTy(rhs.getType());
    if (dw > 0 && llvm::isa<ArrayType>(rhs.getType())) {
      // assigning a string/array value to a scalar (e.g. a string builtin
      // stub result): store zero, with a warning
      emitWarning(location) << "assigning a string value to a scalar, storing 0";
      rhs = builder.create<ConstantOp>(
          location, llvm::isa<IntegerType>(declared)
                        ? declared
                        : IntegerType::get(builder.getContext(), dw),
          builder.getIntegerAttr(
              builder.getIntegerType(llvm::isa<IntegerType>(declared)
                                         ? llvm::cast<IntegerType>(declared).getWidth()
                                         : dw),
              0));
    } else if (dw > 0 && rw > 0 && dw != rw &&
               (llvm::isa<IntegerType>(rhs.getType()) ||
                llvm::isa<mlir::IntegerType>(rhs.getType()))) {
      mlir::Type castTy = llvm::isa<IntegerType>(declared)
                              ? declared
                              : IntegerType::get(builder.getContext(), dw);
      rhs = builder.create<IntegerCastOp>(location, castTy, rhs);
    } else if (dw > 0 && rw > 0 && dw != rw &&
               (llvm::isa<LogicalType>(rhs.getType()) ||
                llvm::isa<LogicalType>(declared))) {
      // bool <-> int assignment (e.g. `r := c.CLEAR()`): zero-extend or
      // truncate through an integer of the declared width.
      mlir::Type castTy = llvm::isa<IntegerType>(declared)
                              ? declared
                              : IntegerType::get(builder.getContext(), dw);
      auto cast = builder.create<IntegerCastOp>(location, castTy, rhs);
      if (llvm::isa<LogicalType>(rhs.getType()))
        cast.setZextAttr(builder.getUnitAttr());
      rhs = cast.getResult();
    }

    // REAL <-> LREAL assignment coercion (IEC 61131-3 LREAL, ADR 004):
    // fpext/fptrunc at lowering via scl.mlir.cast.
    auto isRealTy = [](mlir::Type t) {
      return llvm::isa<RealType>(t) || llvm::isa<LrealType>(t);
    };
    if (isRealTy(declared) && isRealTy(rhs.getType()) &&
        declared != rhs.getType())
      rhs = builder.create<DialectCastOp>(location, declared, rhs);

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

    if (!llvm::isa<AddressType>(address.getType())) {
      emitError(location) << "not a lvalue, type " << address.getType();
      return nullptr;
    }

    return address;
  }

  /// Function calls can have side effects (VAR_IN_OUT write-back, counter
  /// FBs, __ASSERT); cache their result per AST node so the several handlers
  /// in mlirGenFcCall (string builtins, conversions, std functions) do not
  /// re-evaluate the same call expression.
  llvm::DenseMap<const ExpressionAST *, mlir::Value> callResultCache;

  mlir::Value mlirGen(const ExpressionAST *expr) {
    if (auto *call = llvm::dyn_cast<FunctionCallAST>(expr)) {
      auto it = callResultCache.find(call);
      if (it != callResultCache.end())
        return it->second;
      mlir::Value v = mlirGen(call);
      if (v)
        callResultCache[call] = v;
      return v;
    }
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
      auto directTy = llvm::cast<DirectType>(
          llvm::cast<AddressType>(addr.getType()).getElementType());
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
        // IEC VAR_EXTERNAL: load with the declared type (ADR 004).
        auto eit = externalVars.find(llvm::StringRef(var->getName()).upper());
        if (eit != externalVars.end()) {
          auto addr = mlirGen(var);
          if (!addr)
            return nullptr;
          return builder.create<LoadOp>(loc(expr->loc()), eit->second.second,
                                         addr);
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
            auto byteVal = builder.create<LoadOp>(location, byteTy, addr).getResult();
            auto divC = intConst(location, byteTy, 8, (int64_t)(1 << dir->getBit()));
            auto shifted = builder.create<DivOp>(location, byteVal, divC.getResult());
            auto modC = intConst(location, byteTy, 8, (int64_t)2);
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
    if (llvm::isa<AddressType>(expr.getType())) {
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
    double v = expr->getValue();
    // SPEC[ST] IEC 61131-3 ed.3 6.4.4.1: a real literal that is not exactly
    // representable in binary32 is an LREAL (binary64) literal (IEC mode).
    if (iecMode() && (double)(float)v != v) {
      auto type = getType(tok_lreal);
      return builder.create<ConstantOp>(location, type,
                                        builder.getF64FloatAttr(v));
    }
    auto type = getType(tok_real);
    auto value = builder.getF32FloatAttr((float)v);
    return builder.create<ConstantOp>(location, type, value);
  }

  mlir::Value mlirGen(const StringConstantAST *expr) {
    // A single-character string literal used as a value is a char constant
    // (e.g. comparing a CHAR with 'x').
    if (expr->getValue().size() == 1) {
      auto i8 = IntegerType::get(builder.getContext(), 8);
      return intConst(loc(expr->loc()), i8, 8, (uint8_t)expr->getValue()[0]);
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
    if (auto it = llvm::dyn_cast<IntegerType>(resTy))
      width = it.getWidth();
    else if (auto lt = llvm::dyn_cast<LogicalType>(resTy))
      width = lt.getWidth();
    if (auto *ic = llvm::dyn_cast<IntegerConstantAST>(ce))
      return intConst(location, resTy, width, ic->getValue());
    if (llvm::isa<RealConstantAST>(ce) || llvm::isa<TimeConstantAST>(ce))
      return mlirGen(ce);
    if (llvm::isa<StringConstantAST>(ce)) {
      emitWarning(location) << "string constant '" << name
                            << "' used as a value, assuming 0";
      return intConst(location, getType(tok_int), 16, 0);
    }
    // constant expression (e.g. A + B): fold to an integer
    int64_t v = 0;
    if (mlir::failed(foldConstIntAST(ce, v)))
      v = 0;
    return intConst(location, resTy, width, v);
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
    // implicit EN/ENO of the enclosing FB (e.g. `IF NOT ENO THEN` inside a
    // function block): the block's own pin cell, shared with the caller's
    // `inst.ENO` writes (same instance address).
    {
      std::string up = llvm::StringRef(name).upper();
      if (up == "ENO" || up == "EN") {
        auto self = symbolTable.lookup("$self");
        if (self.isAddress()) {
          auto cell = fbEnoCell(location, self.getValue());
          byteViewTypes[cell] = getType(tok_bool);
          return cell;
        }
        // Outside an FB (e.g. in a FUNCTION) there is no instance pin;
        // keep the historical simulator behavior: ENO reads as FALSE.
        return builder.create<ConstantOp>(
            location, getType(tok_bool),
            builder.getIntegerAttr(builder.getI1Type(), 0));
      }
    }
    // enumerated value (e.g. `c := GREEN`)?
    {
      std::string up = llvm::StringRef(name).upper();
      auto it = enumConsts.find(up);
      if (it != enumConsts.end()) {
        auto i16 = getType(tok_int);
        return intConst(location, i16, 16, it->second);
      }
    }
    // IEC VAR_EXTERNAL (ADR 004): a member of the configuration's global DB.
    {
      auto eit = externalVars.find(llvm::StringRef(name).upper());
      if (eit != externalVars.end()) {
        auto dbOp = mlir::cast<DataBlockOp>(
            mlir::SymbolTable::lookupNearestSymbolFrom<DataBlockOp>(
                theModule, builder.getStringAttr(eit->second.first)));
        auto dbAddrTy = AddressType::get(dbOp.getType());
        auto dbAddr =
            builder.create<GetGlobalOp>(location, dbAddrTy, eit->second.first);
        auto elem = mlirGenElement(location, dbAddr, name);
        if (elem)
          byteViewTypes[elem] = eit->second.second;
        return elem;
      }
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
      return intConst(location, getType(tok_int), 16, 1);
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
    auto baseTy = llvm::cast<AddressType>(base.getType()).getElementType();
    if (auto idbTy = llvm::dyn_cast<InstanceDbType>(baseTy)) {
      auto fbName = mlir::StringAttr::get(theModule->getContext(),
                                          idbTy.getFbSymbol());
      auto fb = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
          theModule, fbName);
      if (!fb)
        return nullptr;
      auto var = lookupVariableCI(fb, name);
      return var ? var.getType() : nullptr;
    }
    if (auto structTy = llvm::dyn_cast<StructType>(baseTy))
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
    if (!address) {
      emitError(location) << "internal: null base address for element access '"
                          << name << "'";
      return nullptr;
    }
    mlir::Type baseType = llvm::cast<AddressType>(address.getType()).getElementType();
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
      // A user-defined FB with the same name wins over the TIMER/COUNTER
      // builtin aliases (e.g. an IEC PROGRAM named "counter"): only treat the
      // name as a builtin when no FunctionBlockOp is defined for it.
      auto fb = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionBlockOp>(
          theModule, fbName);
      std::string fbUp =
          (fb && !stubFbNames.count(fb.getName().str()))
              ? ""
              : builtinStandardFb(fbName.getValue());
      if (fb && fbUp.empty() && stubFbNames.count(fb.getName().str()))
        fb = nullptr;
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
          // implicit EN/ENO pins of the instance
          std::string mUp0 = std::string(llvm::StringRef(name).upper());
          if (mUp0 == "ENO" || mUp0 == "EN") {
            auto cell = fbEnoCell(location, address);
            byteViewTypes[cell] = getType(tok_bool);
            return cell;
          }
          emitError(location) << "unknown variable '" << name
                              << "' in builtin FB '" << fbName << "'";
          return nullptr;
        }
        auto addrElem = llvm::cast<AddressType>(address.getType()).getElementType();
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
        // implicit EN/ENO pins of the instance
        std::string mUp0 = std::string(llvm::StringRef(name).upper());
        if (mUp0 == "ENO" || mUp0 == "EN") {
          auto cell = fbEnoCell(location, address);
          byteViewTypes[cell] = getType(tok_bool);
          return cell;
        }
        emitError(location) << "unknown variable '" << name << "' in FB '" << fbName << "'";
        return nullptr;
      }
      // Address the member as an i8 view into the instance byte buffer.
      auto addrElem = llvm::cast<AddressType>(address.getType()).getElementType();
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
      if (auto idbTy = llvm::dyn_cast<InstanceDbType>(it->second))
        return genIdbMember(idbTy);
      if (!llvm::isa<StructType>(it->second)) {
        emitError(location) << "unsupported base type " << baseType
                            << " for element '" << name << "'";
        return nullptr;
      }
      auto structType = llvm::cast<StructType>(it->second);
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
    auto baseElem = llvm::cast<AddressType>(base.getType()).getElementType();
    mlir::Type resultElem;
    mlir::TypeAttr arrayTypeAttr;
    ArrayType arrayTy;
    if (auto at = llvm::dyn_cast<ArrayType>(baseElem)) {
      // Plain array variable: element address has the element type.
      arrayTy = at;
      resultElem = at.getElementType();
    } else if (llvm::isa<IntegerType>(baseElem) &&
               llvm::cast<IntegerType>(baseElem).getWidth() == 8) {
      // Array member of a FB/struct: the base is a byte view; keep the
      // result a byte view and record the declared array type on the op.
      auto it = byteViewTypes.find(base);
      if (it == byteViewTypes.end() || !llvm::isa<ArrayType>(it->second)) {
        emitError(location) << "cannot index into '" << baseElem << "'";
        return nullptr;
      }
      arrayTy = llvm::cast<ArrayType>(it->second);
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

    // IEC 61131-3 ed.3 table 30 date/time arithmetic (DATE - DATE, DT +/- TIME,
    // DT - DT) is expanded into integer arithmetic on the decoded values.
    if (expr->getOp() == tok_plus || expr->getOp() == tok_minus) {
      if (mlir::Value r = mlirGenDateTimeArith(location, expr->getOp(), lhs, rhs))
        return r;
    }

    // Ordering comparisons on DATE / DATE_AND_TIME: their storage is unsigned
    // (day count / BCD calendar), so flip the sign bit before the signed
    // comparison to get the correct unsigned (chronological for BCD) order.
    switch (expr->getOp()) {
    case tok_cmp_eq:
    case tok_cmp_ne:
    case tok_cmp_lt:
    case tok_cmp_le:
    case tok_cmp_gt:
    case tok_cmp_ge:
      if (llvm::isa<DateType>(lhs.getType()) && llvm::isa<DateType>(rhs.getType())) {
        auto flip = [&](mlir::Value v) {
          return builder.create<XOrOp>(
              location, v,
                         intConst(location, v.getType(), 16, 0x8000)
                  .getResult());
        };
        lhs = flip(lhs);
        rhs = flip(rhs);
      } else if (llvm::isa<DateAndTimeType>(lhs.getType()) &&
                 llvm::isa<DateAndTimeType>(rhs.getType())) {
        auto i64 = IntegerType::get(builder.getContext(), 64);
        auto flip = [&](mlir::Value v) {
          mlir::Value c = builder.create<IntegerCastOp>(location, i64, v).getResult();
          return builder.create<XOrOp>(
              location, c,
                         intConst(location, i64, 64, (int64_t)0x8000000000000000LL)
                  .getResult());
        };
        lhs = flip(lhs);
        rhs = flip(rhs);
      }
      break;
    default:
      break;
    }

    // Implicit integer widening for mixed-width operands (e.g. char + int,
    // byte + word): normalize both operands to an IntegerType of the wider
    // width, like SCL. LogicalType (byte/word/dword/bool) values are
    // bitcast to the same-width IntegerType first (IntegerCastOp only
    // accepts IntegerType).
    auto widthOfTy = [](mlir::Type t) -> int {
      if (auto it = llvm::dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto lt = llvm::dyn_cast<LogicalType>(t))
        return lt.getWidth();
      // time/date types compare and mix arithmetically with integers as
      // plain integers of their storage width
      if (llvm::isa<S5TimeType>(t) || llvm::isa<DateType>(t))
        return 16;
      if (llvm::isa<TimeType>(t) || llvm::isa<TimeOfDayType>(t))
        return 32;
      if (llvm::isa<DateAndTimeType>(t))
        return 64;
      return -1;
    };
    auto toIntOfWidth = [&](mlir::Value v, int w) -> mlir::Value {
      int vw = widthOfTy(v.getType());
      if (vw < 0 || (vw == w && llvm::isa<IntegerType>(v.getType())))
        return v;
      // LogicalType (byte/word/dword/bool) is unsigned: zero-extend.
      bool zext = llvm::isa<LogicalType>(v.getType());
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
    // LREAL (binary64) counts as real; mixed REAL/LREAL promotes to LREAL.
    auto isRealTy = [](mlir::Type t) {
      return llvm::isa<RealType>(t) || llvm::isa<LrealType>(t);
    };
    if (isRealTy(lhs.getType()) != isRealTy(rhs.getType()) ||
        (isRealTy(lhs.getType()) && isRealTy(rhs.getType()) &&
         lhs.getType() != rhs.getType())) {
      bool lLreal = llvm::isa<LrealType>(lhs.getType());
      bool rLreal = llvm::isa<LrealType>(rhs.getType());
      auto realTy = (lLreal || rLreal) ? getType(tok_lreal)
                                       : (isRealTy(lhs.getType()) ? lhs.getType()
                                                                  : rhs.getType());
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
      // NOT on a bit string (BYTE/WORD/DWORD) is bitwise; only a BOOL operand
      // is coerced to a logical value.
      if (llvm::isa<LogicalType>(rhs.getType()) &&
          llvm::cast<LogicalType>(rhs.getType()).getWidth() > 1)
        return builder.create<UnaryNotOp>(location, rhs);
      if (llvm::isa<IntegerType>(rhs.getType()))
        return builder.create<UnaryNotOp>(location, rhs);
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
      // `instance.method(...)` (IEC 61131-3 ed.3 METHOD call)
      {
        mlir::Value instAddr;
        if (auto m = resolveMethodCallee(callee, instAddr)) {
          mlirGenFcCall(expr, *m, /*allowVoid=*/true, instAddr);
          return mlir::success();
        }
      }
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
      if (fbNames.contains(llvm::StringRef(fbVar->getName()).upper()) ||
          fbUp == "TON" ||
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
      if (it == byteViewTypes.end() || !llvm::isa<InstanceDbType>(it->second)) {
        emitError(location, "call target '")
            << fbVar->getName() << "." << idbVar->getName()
            << "' is not a function block instance";
        return mlir::failure();
      }
      if (failed(mlirGenFbCallOn(expr, instAddr,
                                 llvm::cast<InstanceDbType>(it->second).getFbSymbol())))
        return mlir::failure();
      return mlir::success();
    })
    .Default([&](auto callee) {
      emitError(location, "invalid callee in function or FB call");
      return mlir::failure();
    });
  }

  /// If `callee` is an FB-method call `instance.method`, resolve the
  /// qualified name "<FB>.<METHOD>" and the instance address.
  std::optional<std::string> resolveMethodCallee(const ExpressionAST *callee,
                                                 mlir::Value &instAddr) {
    auto bin = dyn_cast<BinaryExpressionAST>(callee);
    if (!bin || bin->getOp() != tok_dot)
      return std::nullopt;
    auto rhs = dyn_cast<SimpleVariableAST>(bin->getRhs());
    if (!rhs)
      return std::nullopt;
    std::string fb;
    mlir::Value addr;
    if (auto *sv = dyn_cast<SimpleVariableAST>(bin->getLhs())) {
      // NB: resolve without emitting errors — a plain name may be an FB
      // name (`main."DB"()`) or a global DB, neither of which is a local
      // variable.
      auto sym = symbolTable.lookup(sv->getName());
      if (auto f = fbInstanceOf(sv->getName()))
        fb = *f;
      if (sym.isAddress())
        addr = sym.getValue();
      else if (sym.isElement())
        addr = mlirGenVariable(loc(callee->loc()), sv->getName());
      else if (mlir::SymbolTable::lookupNearestSymbolFrom<DataBlockOp>(
                   theModule, builder.getStringAttr(sv->getName())))
        addr = mlirGen(bin->getLhs());
      else
        return std::nullopt;
    } else {
      addr = mlirGen(bin->getLhs());
    }
    if (!addr)
      return std::nullopt;
    if (fb.empty()) {
      // global instance DB: the address element type is the IDB type
      if (auto idb = llvm::dyn_cast<InstanceDbType>(
              llvm::cast<AddressType>(addr.getType()).getElementType()))
        fb = unquote(idb.getFbSymbol());
    }
    if (fb.empty()) {
      auto it = byteViewTypes.find(addr);
      if (it != byteViewTypes.end())
        if (auto idb = llvm::dyn_cast<InstanceDbType>(it->second))
          fb = unquote(idb.getFbSymbol());
    }
    if (fb.empty())
      return std::nullopt;
    std::string mname = fb + "." + rhs->getName().str();
    for (auto &kv : functionResultTypes) {
      if (kv.getKey().equals_insensitive(mname)) {
        instAddr = addr;
        return kv.getKey().str();
      }
    }
    return std::nullopt;
  }

  mlir::Value mlirGen(const FunctionCallAST *expr) {
    auto location = loc(expr->loc());

    // The function name is parsed as a variable
    auto callee = dyn_cast<SimpleVariableAST>(expr->getFunction());
    if (!callee) {
      // `instance.method(...)` (IEC 61131-3 ed.3 METHOD call)
      mlir::Value instAddr;
      if (auto m = resolveMethodCallee(expr->getFunction(), instAddr))
        return mlirGenFcCall(expr, *m, /*allowVoid=*/false, instAddr);
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
          fStr = nullptr,
      std::function<mlir::LogicalResult (StringRef, const BinaryExpressionAST *)>
          fOut = nullptr) {
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
        if (binary->getOp() == tok_assign_output && fOut) {
          if (failed(fOut(name->getName(), binary)))
            return mlir::failure();
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

  // MARK: date/time arithmetic (IEC 61131-3 ed.3 table 30)
  //
  // SPEC[SCL] IEC 61131-3 ed.3 table 30 date/time arithmetic (DATE - DATE -> TIME, DT +/- TIME -> DT, DT - DT -> TIME), DT_TO_DATE / DT_TO_TOD conversions and DT_YEAR..DT_WEEKDAY field extractors -> mlirGenDateTimeArith / bcdDtToMs / msToBcdDt / mlirGenDtField (DATE = days since 1990-01-01 u16; DT = 8 big-endian BCD bytes in an i64, ADR-001; civil-calendar math via Hinnant's algorithms; BCD byte extraction is floor-div based because the DT sign bit is set for years 1990-1999)
  //
  // DATE is stored as days since 1990-01-01 (u16) and DATE_AND_TIME as 8
  // big-endian BCD bytes in an i64 (ADR-001), so the table-30 operations
  // (DATE - DATE, DT +/- TIME, DT - DT) are expanded here into plain integer
  // arithmetic on the decoded values. All helpers work on i64 values.

  mlir::Value dtConst(mlir::Location loc, int64_t v) {
    // NB: the attr must be typed with an mlir::IntegerType (IntegerAttr
    // requires it); the op result keeps the scl !scl.dint type.
    auto i64 = IntegerType::get(builder.getContext(), 64);
    return  intConst(loc, i64, 64, v)
        .getResult();
  }
  mlir::Value dtAdd(mlir::Location loc, mlir::Value a, mlir::Value b) {
    return builder.create<AddOp>(loc, a, b).getResult();
  }
  mlir::Value dtSub(mlir::Location loc, mlir::Value a, mlir::Value b) {
    return builder.create<SubOp>(loc, a, b).getResult();
  }
  mlir::Value dtMul(mlir::Location loc, mlir::Value a, mlir::Value b) {
    return builder.create<MulOp>(loc, a, b).getResult();
  }
  mlir::Value dtDiv(mlir::Location loc, mlir::Value a, mlir::Value b) {
    return builder.create<DivOp>(loc, a, b).getResult();
  }
  mlir::Value dtMod(mlir::Location loc, mlir::Value a, mlir::Value b) {
    return builder.create<ModOp>(loc, a, b).getResult();
  }
  /// Zero-extend a BOOL comparison result to i64 (0/1).
  mlir::Value dtZextBool(mlir::Location loc, mlir::Value b) {
    auto i64 = IntegerType::get(builder.getContext(), 64);
    auto c = builder.create<IntegerCastOp>(loc, i64, b);
    c.setZextAttr(builder.getUnitAttr());
    return c.getResult();
  }
  /// Floor division (scl div truncates toward zero, which breaks modular
  /// identities for negative operands).
  mlir::Value dtFloorDiv(mlir::Location loc, mlir::Value a, int64_t p) {
    mlir::Value q = dtDiv(loc, a, dtConst(loc, p));
    mlir::Value r = dtMod(loc, a, dtConst(loc, p));
    mlir::Value neg = dtZextBool(
        loc, builder.create<LessThanOp>(loc, r, dtConst(loc, 0)).getResult());
    return dtSub(loc, q, neg);
  }
  /// Unsigned field extraction from a possibly negative i64 (the BCD DT
  /// value has its sign bit set for years 1990-1999): floor-div by p, then
  /// fix the remainder of the mod up into [0, m).
  mlir::Value dtUField(mlir::Location loc, mlir::Value arg, int64_t p,
                       int64_t m) {
    mlir::Value q = dtFloorDiv(loc, arg, p);
    mlir::Value r = dtMod(loc, q, dtConst(loc, m));
    return dtMod(loc, dtAdd(loc, r, dtConst(loc, m)), dtConst(loc, m));
  }
  /// Byte k (0 = most significant) of the big-endian i64 `arg` (unsigned).
  mlir::Value dtByte(mlir::Location loc, mlir::Value arg, int k) {
    int64_t p = 1;
    for (int i = 0; i < 7 - k; ++i)
      p *= 256;
    return dtUField(loc, arg, p, 256);
  }
  /// Decode a 2-digit BCD byte: (b % 16) + 10 * (b / 16).
  mlir::Value dtBcd2(mlir::Location loc, mlir::Value b) {
    return dtAdd(loc, dtMod(loc, b, dtConst(loc, 16)),
                 dtMul(loc, dtDiv(loc, b, dtConst(loc, 16)), dtConst(loc, 10)));
  }

  /// Days since 1970-01-01 from a civil date (Howard Hinnant's
  /// days_from_civil; valid for dates >= 1970, which covers the DT range
  /// starting at 1990). All operands i64.
  mlir::Value daysFromCivil(mlir::Location loc, mlir::Value y, mlir::Value m,
                            mlir::Value d) {
    // y -= (m <= 2)
    y = dtSub(loc, y,
              dtZextBool(loc, builder.create<LessEqualOp>(loc, m, dtConst(loc, 2))
                                 .getResult()));
    mlir::Value era = dtDiv(loc, y, dtConst(loc, 400));
    mlir::Value yoe = dtSub(loc, y, dtMul(loc, era, dtConst(loc, 400)));
    // mAdj = m > 2 ? m - 3 : m + 9
    mlir::Value mAdj = dtAdd(
        loc, m,
        dtSub(loc, dtConst(loc, 9),
              dtMul(loc, dtConst(loc, 12),
                    dtZextBool(loc, builder.create<GreaterThanOp>(loc, m, dtConst(loc, 2))
                                       .getResult()))));
    mlir::Value doy =
        dtAdd(loc,
              dtDiv(loc, dtAdd(loc, dtMul(loc, dtConst(loc, 153), mAdj),
                               dtConst(loc, 2)),
                    dtConst(loc, 5)),
              dtSub(loc, d, dtConst(loc, 1)));
    mlir::Value doe = dtAdd(
        loc,
        dtSub(loc, dtAdd(loc, dtMul(loc, yoe, dtConst(loc, 365)),
                         dtDiv(loc, yoe, dtConst(loc, 4))),
              dtDiv(loc, yoe, dtConst(loc, 100))),
        doy);
    return dtSub(loc, dtAdd(loc, dtMul(loc, era, dtConst(loc, 146097)), doe),
                 dtConst(loc, 719468));
  }

  /// Inverse of daysFromCivil; returns {year, month, day} for a day count
  /// since 1970-01-01 (valid for z >= 0).
  llvm::SmallVector<mlir::Value, 3> civilFromDays(mlir::Location loc,
                                                  mlir::Value z) {
    z = dtAdd(loc, z, dtConst(loc, 719468));
    mlir::Value era = dtDiv(loc, z, dtConst(loc, 146097));
    mlir::Value doe = dtMod(loc, z, dtConst(loc, 146097));
    mlir::Value yoe = dtDiv(
        loc,
        dtSub(loc,
              dtSub(loc, dtAdd(loc, doe, dtDiv(loc, doe, dtConst(loc, 36524))),
                    dtDiv(loc, doe, dtConst(loc, 1460))),
              dtDiv(loc, doe, dtConst(loc, 146096))),
        dtConst(loc, 365));
    mlir::Value y = dtAdd(loc, yoe, dtMul(loc, era, dtConst(loc, 400)));
    mlir::Value doy = dtSub(
        loc, doe,
        dtSub(loc, dtAdd(loc, dtMul(loc, yoe, dtConst(loc, 365)),
                         dtDiv(loc, yoe, dtConst(loc, 4))),
              dtDiv(loc, yoe, dtConst(loc, 100))));
    mlir::Value mp =
        dtDiv(loc, dtAdd(loc, dtMul(loc, dtConst(loc, 5), doy), dtConst(loc, 2)),
              dtConst(loc, 153));
    mlir::Value d = dtAdd(
        loc,
        dtSub(loc, doy,
              dtDiv(loc, dtAdd(loc, dtMul(loc, dtConst(loc, 153), mp),
                               dtConst(loc, 2)),
                    dtConst(loc, 5))),
        dtConst(loc, 1));
    // m = mp >= 10 ? mp - 9 : mp + 3
    mlir::Value m = dtAdd(
        loc, mp,
        dtSub(loc, dtConst(loc, 3),
              dtMul(loc, dtConst(loc, 12),
                    dtZextBool(loc,
                               builder.create<GreaterEqualOp>(loc, mp, dtConst(loc, 10))
                                   .getResult()))));
    // y += (m <= 2)
    y = dtAdd(loc, y,
              dtZextBool(loc, builder.create<LessEqualOp>(loc, m, dtConst(loc, 2))
                                 .getResult()));
    return {y, m, d};
  }

  /// Full year from the 2-digit BCD year byte: the S7 DATE_AND_TIME range is
  /// 1990-2089, so 90..99 mean 19xx and 00..89 mean 20xx.
  mlir::Value dtFullYear(mlir::Location loc, mlir::Value yy) {
    // y = yy >= 90 ? 1900 + yy : 2000 + yy
    mlir::Value ge90 = dtZextBool(loc, builder.create<GreaterEqualOp>(
                                            loc, yy, dtConst(loc, 90))
                                         .getResult());
    return dtAdd(loc, yy,
                 dtSub(loc, dtConst(loc, 2000),
                       dtMul(loc, ge90, dtConst(loc, 100))));
  }

  /// Decode a BCD DATE_AND_TIME value to ms since 1990-01-01 00:00:00 (i64).
  mlir::Value bcdDtToMs(mlir::Location loc, mlir::Value dt) {
    auto i64 = IntegerType::get(builder.getContext(), 64);
    if (dt.getType() != i64)
      dt = builder.create<IntegerCastOp>(loc, i64, dt).getResult();
    mlir::Value y = dtFullYear(loc, dtBcd2(loc, dtByte(loc, dt, 0)));
    mlir::Value mo = dtBcd2(loc, dtByte(loc, dt, 1));
    mlir::Value d = dtBcd2(loc, dtByte(loc, dt, 2));
    mlir::Value h = dtBcd2(loc, dtByte(loc, dt, 3));
    mlir::Value mi = dtBcd2(loc, dtByte(loc, dt, 4));
    mlir::Value s = dtBcd2(loc, dtByte(loc, dt, 5));
    // msec: 3 BCD nibbles at bits 4..15
    mlir::Value nib = dtUField(loc, dt, 16, 4096);
    mlir::Value msec = dtAdd(
        loc, dtMul(loc, dtDiv(loc, nib, dtConst(loc, 256)), dtConst(loc, 100)),
        dtAdd(loc,
              dtMul(loc, dtDiv(loc, dtMod(loc, nib, dtConst(loc, 256)),
                               dtConst(loc, 16)),
                    dtConst(loc, 10)),
              dtMod(loc, nib, dtConst(loc, 16))));
    // 1970-01-01 .. 1990-01-01 = 7305 days
    mlir::Value days =
        dtSub(loc, daysFromCivil(loc, y, mo, d), dtConst(loc, 7305));
    // ms = ((days*24 + h)*3600 + mi*60 + s)*1000 + msec
    return dtAdd(
        loc,
        dtMul(loc,
              dtAdd(loc,
                    dtMul(loc, dtAdd(loc, dtMul(loc, days, dtConst(loc, 24)), h),
                          dtConst(loc, 3600)),
                    dtAdd(loc, dtMul(loc, mi, dtConst(loc, 60)), s)),
              dtConst(loc, 1000)),
        msec);
  }

  /// Encode ms since 1990-01-01 00:00:00 (i64, may be negative) as a BCD
  /// DATE_AND_TIME value (i64), including the S7 weekday nibble.
  mlir::Value msToBcdDt(mlir::Location loc, mlir::Value ms) {
    auto i64 = IntegerType::get(builder.getContext(), 64);
    if (ms.getType() != i64)
      ms = builder.create<IntegerCastOp>(loc, i64, ms).getResult();
    // floor div/mod by 86400000 (ms may be negative)
    // ms is relative to 1990-01-01 (as produced by bcdDtToMs), so q is a
    // day count since 1990; civilFromDays wants days since 1970-01-01.
    mlir::Value q = dtDiv(loc, ms, dtConst(loc, 86400000));
    mlir::Value r = dtMod(loc, ms, dtConst(loc, 86400000));
    mlir::Value neg = dtZextBool(
        loc, builder.create<LessThanOp>(loc, r, dtConst(loc, 0)).getResult());
    q = dtSub(loc, q, neg);
    r = dtAdd(loc, r, dtMul(loc, neg, dtConst(loc, 86400000)));
    mlir::Value days1990 = q;
    auto ymd = civilFromDays(loc, dtAdd(loc, q, dtConst(loc, 7305)));
    mlir::Value h = dtDiv(loc, r, dtConst(loc, 3600000));
    r = dtMod(loc, r, dtConst(loc, 3600000));
    mlir::Value mi = dtDiv(loc, r, dtConst(loc, 60000));
    r = dtMod(loc, r, dtConst(loc, 60000));
    mlir::Value s = dtDiv(loc, r, dtConst(loc, 1000));
    mlir::Value msec = dtMod(loc, r, dtConst(loc, 1000));
    // S7 weekday: 1=Sunday..7=Saturday; 1990-01-01 was a Monday (2)
    mlir::Value wd = dtAdd(
        loc, dtMod(loc, dtAdd(loc, dtMod(loc, days1990, dtConst(loc, 7)),
                              dtConst(loc, 1)),
                      dtConst(loc, 7)),
        dtConst(loc, 1));
    auto bcd2e = [&](mlir::Value v) {
      return dtAdd(loc, dtMul(loc, dtDiv(loc, v, dtConst(loc, 10)),
                              dtConst(loc, 16)),
                   dtMod(loc, v, dtConst(loc, 10)));
    };
    auto bcd3e = [&](mlir::Value v) {
      return dtAdd(loc, dtMul(loc, dtDiv(loc, v, dtConst(loc, 100)),
                              dtConst(loc, 256)),
                   dtAdd(loc,
                         dtMul(loc, dtDiv(loc, dtMod(loc, v, dtConst(loc, 100)),
                                          dtConst(loc, 10)),
                               dtConst(loc, 16)),
                         dtMod(loc, v, dtConst(loc, 10))));
    };
    auto scale = [&](mlir::Value v, int shift) {
      int64_t p = 1;
      for (int i = 0; i < shift; ++i)
        p *= 256;
      return dtMul(loc, v, dtConst(loc, p));
    };
    mlir::Value val = dtAdd(
        loc, scale(bcd2e(dtMod(loc, ymd[0], dtConst(loc, 100))), 7),
        dtAdd(loc, scale(bcd2e(ymd[1]), 6),
              dtAdd(loc, scale(bcd2e(ymd[2]), 5),
                    dtAdd(loc, scale(bcd2e(h), 4),
                          dtAdd(loc, scale(bcd2e(mi), 3),
                                dtAdd(loc, scale(bcd2e(s), 2),
                                      dtAdd(loc, dtMul(loc, bcd3e(msec),
                                                       dtConst(loc, 16)),
                                            wd)))))));
    return val;
  }

  /// IEC 61131-3 ed.3 table 30 arithmetic on DATE / TIME_OF_DAY /
  /// DATE_AND_TIME operands. Returns nullptr if the combination is not a
  /// date/time operation (the caller falls back to generic arithmetic).
  mlir::Value mlirGenDateTimeArith(mlir::Location location, Token op,
                                   mlir::Value lhs, mlir::Value rhs) {
    auto *ctx = builder.getContext();
    auto i64 = IntegerType::get(ctx, 64);
    auto i32 = IntegerType::get(ctx, 32);
    auto timeTy = TimeType::get(ctx);
    bool minus = op == tok_minus;
    auto toI64 = [&](mlir::Value v, bool zext) {
      if (v.getType() == i64)
        return v;
      auto c = builder.create<IntegerCastOp>(location, i64, v);
      if (zext)
        c.setZextAttr(builder.getUnitAttr());
      return c.getResult();
    };

    // DATE - DATE -> TIME (day difference in ms)
    if (minus && llvm::isa<DateType>(lhs.getType()) && llvm::isa<DateType>(rhs.getType())) {
      mlir::Value d = dtSub(location, toI64(lhs, true), toI64(rhs, true));
      mlir::Value ms = dtMul(location, d, dtConst(location, 86400000));
      mlir::Value t = builder.create<IntegerCastOp>(location, i32, ms).getResult();
      return builder.create<DialectCastOp>(location, timeTy, t).getResult();
    }
    // DT - DT -> TIME (ms difference of the decoded calendar values)
    if (minus && llvm::isa<DateAndTimeType>(lhs.getType()) &&
        llvm::isa<DateAndTimeType>(rhs.getType())) {
      mlir::Value ms = dtSub(location, bcdDtToMs(location, lhs),
                             bcdDtToMs(location, rhs));
      mlir::Value t = builder.create<IntegerCastOp>(location, i32, ms).getResult();
      return builder.create<DialectCastOp>(location, timeTy, t).getResult();
    }
    // DT +/- TIME -> DT (TIME + DT allowed for ADD)
    if ((llvm::isa<DateAndTimeType>(lhs.getType()) && llvm::isa<TimeType>(rhs.getType())) ||
        (!minus && llvm::isa<TimeType>(lhs.getType()) && llvm::isa<DateAndTimeType>(rhs.getType()))) {
      mlir::Value dt = llvm::isa<DateAndTimeType>(lhs.getType()) ? lhs : rhs;
      mlir::Value tm = llvm::isa<DateAndTimeType>(lhs.getType()) ? rhs : lhs;
      mlir::Value ms = bcdDtToMs(location, dt);
      mlir::Value t = toI64(tm, false); // TIME is a signed i32
      mlir::Value r = minus ? dtSub(location, ms, t) : dtAdd(location, ms, t);
      mlir::Value bcd = msToBcdDt(location, r);
      return builder.create<DialectCastOp>(location, DateAndTimeType::get(ctx), bcd)
          .getResult();
    }
    return nullptr;
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
    else if (field == "TOD" || field == "TO_TOD")
      byteIdx = -2; // special: time-of-day in ms (DT_TO_TOD)
    else if (field == "TO_DATE")
      byteIdx = -3; // special: DATE as days since 1990 (DT_TO_DATE)
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
      if (llvm::isa<LogicalType>(arg.getType()))
        cast.setZextAttr(builder.getUnitAttr());
      arg = cast.getResult();
    }
    auto i64Const = [&](int64_t v) {
      // NB: the attr must be typed with an mlir::IntegerType (IntegerAttr
      // requires it); the op result keeps the scl !scl.dint type.
      return    intConst(location, i64, 64, v)
          .getResult();
    };
    if (byteIdx == -2) {
      // DT_TO_TOD: ms since midnight from the BCD hour/min/sec/msec fields
      mlir::Value h = dtBcd2(location, dtByte(location, arg, 3));
      mlir::Value mi = dtBcd2(location, dtByte(location, arg, 4));
      mlir::Value s = dtBcd2(location, dtByte(location, arg, 5));
      // msec: 3 BCD nibbles at bits 4..15
      mlir::Value nib = dtUField(location, arg, 16, 4096);
      mlir::Value msec = dtAdd(
          location, dtMul(location, dtDiv(location, nib, i64Const(256)),
                          i64Const(100)),
          dtAdd(location,
                dtMul(location, dtDiv(location, dtMod(location, nib, i64Const(256)),
                                      i64Const(16)),
                      i64Const(10)),
                dtMod(location, nib, i64Const(16))));
      mlir::Value ms = dtAdd(
          location, dtMul(location, h, i64Const(3600000)),
          dtAdd(location, dtMul(location, mi, i64Const(60000)),
                dtAdd(location, dtMul(location, s, i64Const(1000)), msec)));
      auto todTy = getType(tok_time_of_day);
      auto i32 = IntegerType::get(ctx, 32);
      if (ms.getType() != i32)
        ms = builder.create<IntegerCastOp>(location, i32, ms).getResult();
      if (ms.getType() != todTy)
        ms = builder.create<DialectCastOp>(location, todTy, ms).getResult();
      return ms;
    }
    if (byteIdx == -3) {
      // DT_TO_DATE: days since 1990-01-01 from the BCD year/month/day bytes
      mlir::Value y = dtFullYear(location,
                                 dtBcd2(location, dtByte(location, arg, 0)));
      mlir::Value mo = dtBcd2(location, dtByte(location, arg, 1));
      mlir::Value d = dtBcd2(location, dtByte(location, arg, 2));
      mlir::Value days = dtSub(location, daysFromCivil(location, y, mo, d),
                               i64Const(7305));
      auto i16 = IntegerType::get(ctx, 16);
      auto dv = builder.create<IntegerCastOp>(location, i16, days);
      dv.setZextAttr(builder.getUnitAttr()); // day count is non-negative
      return builder
          .create<DialectCastOp>(location, DateType::get(ctx), dv.getResult())
          .getResult();
    }
    // byte = (v / 256^(7-byteIdx)) % 256, unsigned-safe (see dtUField)
    int64_t pow = 1;
    for (int i = 0; i < 7 - byteIdx; ++i)
      pow *= 256;
    auto byteVal = dtUField(location, arg, pow, 256);
    mlir::Value fieldVal = byteVal;
    if (bcd) {
      // BCD decode: (b % 16) + 10 * (b / 16)
      auto lo = builder.create<ModOp>(location, byteVal, i64Const(16)).getResult();
      auto hi = builder.create<DivOp>(location, byteVal, i64Const(16)).getResult();
      fieldVal = builder.create<AddOp>(
          location, lo, builder.create<MulOp>(location, hi, i64Const(10)).getResult()).getResult();
      // DT_YEAR returns the full year (1990-2089 century rule)
      if (field == "YEAR")
        fieldVal = dtFullYear(location, fieldVal);
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
    else if (dst == "REAL")
      dstTok = tok_real;
    else if (dst == "LREAL")
      dstTok = tok_lreal;
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
    if (llvm::isa<ArrayType>(arg.getType())) {
      emitWarning(location) << "conversion of a string value, assuming 0";
      return intConst(location, resTy, 16, 0);
    }
    // REAL/LREAL source: dialect-cast straight to the target (fptosi/… at
    // lowering); REAL <-> LREAL is an fpext/fptrunc.
    auto isRealTy = [](mlir::Type t) {
      return llvm::isa<RealType>(t) || llvm::isa<LrealType>(t);
    };
    if (isRealTy(arg.getType()) && !isRealTy(resTy)) {
      auto cast = builder.create<DialectCastOp>(location, resTy, arg);
      // IEC 61131-3 ed.3 table 22: real -> integer conversions round to the
      // nearest integer (ties are implementation-defined)
      if (llvm::isa<IntegerType>(resTy) || llvm::isa<LogicalType>(resTy))
        cast.setRoundAttr(builder.getUnitAttr());
      return cast.getResult();
    }
    if (isRealTy(arg.getType()) && isRealTy(resTy) && arg.getType() != resTy)
      return builder.create<DialectCastOp>(location, resTy, arg).getResult();
    // cast to the destination width as an IntegerType first (int.cast only
    // has integer results), then dialect-cast to the scl target type
    int width = 16;
    if (auto it = llvm::dyn_cast<IntegerType>(resTy))
      width = it.getWidth();
    else if (auto lt = llvm::dyn_cast<LogicalType>(resTy))
      width = lt.getWidth();
    else if (llvm::isa<RealType>(resTy))
      width = 32;
    else if (llvm::isa<LrealType>(resTy))
      width = 64;
    else if (llvm::isa<S5TimeType>(resTy) || llvm::isa<DateType>(resTy))
      width = 16;
    else if (llvm::isa<TimeType>(resTy) || llvm::isa<TimeOfDayType>(resTy))
      width = 32;
    else if (llvm::isa<DateAndTimeType>(resTy))
      width = 64;
    if (isRealTy(resTy)) {
      // int -> real/lreal: sitofp at lowering, via mlir.cast i32 -> !scl.real
      auto intTy = IntegerType::get(context, 32);
      if (arg.getType() != intTy) {
        auto cast = builder.create<IntegerCastOp>(location, intTy, arg);
        if (llvm::isa<LogicalType>(arg.getType()))
          cast.setZextAttr(builder.getUnitAttr()); // byte/word/dword: unsigned
        arg = cast.getResult();
      }
      return builder.create<DialectCastOp>(location, resTy, arg).getResult();
    }
    auto intTy = IntegerType::get(context, width);
    if (arg.getType() != intTy) {
      auto cast = builder.create<IntegerCastOp>(location, intTy, arg);
      // BYTE/WORD/DWORD (LogicalType) are unsigned bit strings: zero-extend
      // on widening (UINT_TO_DINT(16#FFFF) = 65535, not -1). CHAR/SINT are
      // IntegerType(8) and keep sign extension.
      if (llvm::isa<LogicalType>(resTy) || llvm::isa<LogicalType>(arg.getType()))
        cast.setZextAttr(builder.getUnitAttr());
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
    if (auto it = llvm::dyn_cast<IntegerType>(ty))
      width = it.getWidth();
    else if (auto lt = llvm::dyn_cast<LogicalType>(ty))
      width = lt.getWidth();
    return intConst(location, ty, width, v);
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
    if (auto it = llvm::dyn_cast<IntegerType>(v.getType())) {
      auto c = builder.create<IntegerCastOp>(location, d, v);
      if (it.getWidth() == 8)
        c.setZextAttr(builder.getUnitAttr()); // char: unsigned
      return c.getResult();
    }
    if (llvm::isa<LogicalType>(v.getType())) {
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
    if (llvm::isa<IntegerType>(v.getType()) || llvm::isa<LogicalType>(v.getType()))
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
    if (llvm::isa<IntegerType>(target))
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
    if (llvm::isa<AddressType>(v.getType())) {
      if (ArrayType at = arrayTypeOfAddress(v)) {
        s.addr = v;
        s.arrTy = at;
        return s;
      }
      // scalar address (char var / member): load with the declared type
      mlir::Type vt = llvm::cast<AddressType>(v.getType()).getElementType();
      auto it = byteViewTypes.find(v);
      if (it != byteViewTypes.end())
        vt = it->second;
      s.scalar = toChar(location, builder.create<LoadOp>(location, vt, v));
      return s;
    }
    if (auto at = llvm::dyn_cast<ArrayType>(v.getType())) {
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
        // integer parameters L / P (1-based position), as scl dint; their
        // positional index follows the ed.3 table 34 signature of each
        // function: LEFT/RIGHT(IN,L), MID/DELETE(IN,L,P), INSERT(IN,IN2,P),
        // REPLACE(IN,IN2,L,P), FIND(IN1,IN2) has no numeric parameter.
        auto numArg = [&](const char *name, int pos) -> mlir::Value {
          auto *e = arg(name, pos);
          if (!e || llvm::isa<StringConstantAST>(e))
            return mlir::Value();
          return numArgD(location, e);
        };
        int lPos = -1, pPos = -1;
        if (up == "LEFT" || up == "RIGHT" || up == "MID" || up == "DELETE")
          lPos = 1;
        if (up == "MID" || up == "DELETE" || up == "INSERT")
          pPos = 2;
        if (up == "REPLACE") {
          lPos = 2;
          pPos = 3;
        }
        mlir::Value L = lPos >= 0 ? numArg("L", lPos) : mlir::Value();
        mlir::Value P = pPos >= 0 ? numArg("P", pPos) : mlir::Value();
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
            // INSERT(IN,IN2,P): IEC ed.3 inserts IN2 AFTER 1-based position P
            // (0-based index P); Siemens SCL inserts AT position P (0-based
            // P-1) — a real S7-vs-IEC difference, so it is mode-dependent.
            // REPLACE(IN,IN2,L,P): IN2 replaces the L chars of IN starting at
            // 1-based position P (0-based P-1) in both.
            mlir::Value skip = up == "REPLACE" ? L : sclConst(location, d, 0);
            mlir::Value insPos =
                iecMode() ? P : pm1; // INSERT insertion index (0-based)
            mlir::Value pos =
                up == "INSERT"
                    ? mux(location,
                          builder.create<GreaterThanOp>(location, insPos, lenA),
                          lenA, insPos)
                    : pm1;
            int64_t nBInt = b.maxLen();
            mlir::Value srcB = clampIdx(
                location, builder.create<SubOp>(location, ic, pos).getResult(), nBInt);
            mlir::Value byteB = byteVD(location, b, srcB);
            byteB = mux(location,
                        builder.create<LessThanOp>(location, srcB, lenB), byteB,
                        sclConst(location, d, 0));
            // tail of a after the replaced/inserted region: a[i - lenB + skip];
            // keep the raw index for the range check (clampIdx is only for
            // the load itself)
            mlir::Value rawSrcA =
                builder.create<AddOp>(
                    location,
                    builder.create<SubOp>(location, ic, lenB).getResult(),
                    skip).getResult();
            mlir::Value srcA = clampIdx(location, rawSrcA, nA);
            mlir::Value inMid = builder.create<AndOp>(
                location,
                builder.create<GreaterEqualOp>(location, ic, pos),
                builder.create<LessThanOp>(
                    location, ic,
                    builder.create<AddOp>(location, pos, lenB).getResult()));
            mlir::Value inTail = builder.create<AndOp>(
                location,
                builder.create<GreaterEqualOp>(
                    location, ic,
                    builder.create<AddOp>(location, pos, lenB).getResult()),
                builder.create<AndOp>(
                    location,
                    builder.create<GreaterEqualOp>(location, rawSrcA,
                                                   sclConst(location, d, 0)),
                    builder.create<LessThanOp>(location, rawSrcA, lenA)));
            mlir::Value byteHead = mux(
                location, builder.create<LessThanOp>(location, ic, pos),
                byteD(location, a, i), sclConst(location, d, 0));
            mlir::Value byteTail = mux(location, inTail,
                                       byteVD(location, a, srcA), sclConst(location, d, 0));
            byte = mux(location, builder.create<LessThanOp>(location, ic, pos),
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
        {"ACOS", "acos"}, {"ATAN", "atan"}, {"SQRT", "sqrt"}, {"EXP", "exp"},
        {"LN", "ln"}, {"LOG", "ln"}, {"ABS", "abs"}};
    static const std::map<std::string, bool> selFns = {
        {"MAX", true}, {"MIN", false}}; // max=true, min=false

    SmallVector<mlir::Value, 4> args;
    SmallVector<std::pair<std::string, mlir::Value>, 4> namedArgs;
    bool sawNamed = false;
    if (failed(mlirGenCallParameters(
            expr,
            [&](StringRef n, mlir::Value v) {
              namedArgs.push_back({std::string(n.upper()), v});
              sawNamed = true;
            },
            [&](mlir::Value v) { args.push_back(v); })))
      return nullptr;
    if (sawNamed) {
      // Formal call: reorder the named arguments to the function's declared
      // parameter order (IEC ed.3 tables 22/23/32), e.g. LIMIT(MX, IN, MN).
      args.clear();
      auto pushByName = [&](StringRef n) -> bool {
        for (auto &p : namedArgs)
          if (p.first == n) {
            args.push_back(p.second);
            return true;
          }
        return false;
      };
      if (up == "MUX") { // MUX(K, IN0..INn): K first, then IN0.. by index
        pushByName("K");
        for (int i = 0;; ++i) {
          std::string n = "IN" + std::to_string(i);
          if (!pushByName(n))
            break;
        }
      } else {
        static const std::map<std::string, std::vector<std::string>>
            paramOrder = {{"LIMIT", {"MN", "IN", "MX"}},
                          {"SEL", {"G", "IN0", "IN1"}}};
        auto it = paramOrder.find(up);
        if (it != paramOrder.end()) {
          for (auto &n : it->second)
            pushByName(n);
        } else {
          for (auto &p : namedArgs)
            args.push_back(p.second);
        }
      }
    }
    if (args.empty() || !args[0])
      return nullptr;
    auto ctx = builder.getContext();
    mlir::Type t0 = args[0].getType();

    // Unary real functions on LREAL: scl_rt_realmath_d (libm on binary64).
    if (auto it = realFns.find(up); it != realFns.end() &&
                                      args.size() == 1 &&
                                      llvm::isa<LrealType>(t0)) {
      static const std::map<std::string, int> kindOf = {
          {"sin", 1}, {"cos", 2},  {"tan", 3},  {"asin", 4}, {"acos", 5},
          {"atan", 6}, {"sqrt", 7}, {"exp", 8},  {"ln", 9},   {"abs", 10}};
      auto kIt = kindOf.find(it->second);
      if (kIt == kindOf.end())
        return nullptr;
      auto lrealTy = getType(tok_lreal);
      auto i32 = builder.getI32Type();
      auto fnTy = builder.getFunctionType({i32, lrealTy}, {lrealTy});
      auto fn = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionOp>(
          theModule, builder.getStringAttr("scl_rt_realmath_d"));
      if (!fn) {
        mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
        fn = b.create<FunctionOp>(location, "scl_rt_realmath_d", fnTy);
        fn.setPrivate();
      }
      auto kindC = builder.create<ConstantOp>(
          location, i32, builder.getIntegerAttr(i32, kIt->second));
      return builder.create<CallFcOp>(location, lrealTy, "scl_rt_realmath_d",
                                      mlir::ValueRange{kindC, args[0]},
                                      mlir::ArrayAttr::get(ctx, {}),
                                      nullptr, nullptr);
    }
    // Unary real functions (operand REAL/LREAL -> awl.real_math, f32).
    if (auto it = realFns.find(up); it != realFns.end() &&
                                      args.size() == 1 &&
                                      llvm::isa<RealType>(t0)) {
      auto f32 = mlir::Float32Type::get(ctx);
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

    // Common std type of numeric-ish scl args (int/logical/time/date): f32 if
    // any REAL, else the max integer width (bit strings and date/time types
    // map to their storage width).
    auto commonStdTy = [&](ArrayRef<mlir::Value> vs) -> mlir::Type {
      if (llvm::any_of(vs,
                       [](mlir::Value v) { return llvm::isa<RealType>(v.getType()); }))
        return (mlir::Type)mlir::Float32Type::get(ctx);
      unsigned w = 1;
      for (auto v : vs) {
        mlir::Type t = v.getType();
        if (auto it = llvm::dyn_cast<IntegerType>(t))
          w = std::max(w, (unsigned)it.getWidth());
        else if (auto lt = llvm::dyn_cast<LogicalType>(t))
          w = std::max(w, (unsigned)lt.getWidth());
        else if (llvm::isa<TimeType>(t) || llvm::isa<TimeOfDayType>(t))
          w = std::max(w, 32u);
        else if (llvm::isa<S5TimeType>(t) || llvm::isa<DateType>(t))
          w = std::max(w, 16u);
        else if (llvm::isa<DateAndTimeType>(t))
          w = std::max(w, 64u);
      }
      return (mlir::Type)mlir::IntegerType::get(ctx, w);
    };
    // Widest scl type among args (result type of n-ary numeric functions).
    auto commonSclTy = [&](ArrayRef<mlir::Value> vs) -> mlir::Type {
      mlir::Type best = vs[0].getType();
      auto widthOf = [&](mlir::Type t) -> unsigned {
        if (auto it = llvm::dyn_cast<IntegerType>(t))
          return it.getWidth();
        if (auto lt = llvm::dyn_cast<LogicalType>(t))
          return lt.getWidth();
        if (llvm::isa<RealType>(t))
          return 1000; // real dominates
        if (llvm::isa<TimeType>(t) || llvm::isa<TimeOfDayType>(t))
          return 32;
        if (llvm::isa<DateAndTimeType>(t))
          return 64;
        return 16;
      };
      for (auto v : vs)
        if (widthOf(v.getType()) > widthOf(best))
          best = v.getType();
      return best;
    };

    // Comparison functions LT/LE/GT/GE/EQ/NE -> bool (SPEC[SCL] table 23).
    static const std::map<std::string, mlir::arith::CmpIPredicate> cmpFns = {
        {"LT", mlir::arith::CmpIPredicate::slt},
        {"LE", mlir::arith::CmpIPredicate::sle},
        {"GT", mlir::arith::CmpIPredicate::sgt},
        {"GE", mlir::arith::CmpIPredicate::sge},
        {"EQ", mlir::arith::CmpIPredicate::eq},
        {"NE", mlir::arith::CmpIPredicate::ne}};
    if (auto it = cmpFns.find(up); it != cmpFns.end() && args.size() == 2) {
      mlir::Type stdTy = commonStdTy(args);
      mlir::Value a = toStd(args[0], stdTy), b = toStd(args[1], stdTy);
      mlir::Value res;
      if (llvm::isa<mlir::FloatType>(stdTy)) {
        static const std::map<std::string, mlir::arith::CmpFPredicate> cmpFFns = {
            {"LT", mlir::arith::CmpFPredicate::OLT},
            {"LE", mlir::arith::CmpFPredicate::OLE},
            {"GT", mlir::arith::CmpFPredicate::OGT},
            {"GE", mlir::arith::CmpFPredicate::OGE},
            {"EQ", mlir::arith::CmpFPredicate::OEQ},
            {"NE", mlir::arith::CmpFPredicate::ONE}};
        res = mlir::sclang::withFPFastMath(builder.create<mlir::arith::CmpFOp>(location, cmpFFns.at(up), a, b))
                  .getResult();
      } else {
        res = builder.create<mlir::arith::CmpIOp>(location, it->second, a, b)
                  .getResult();
      }
      return fromStd(res, getType(tok_bool));
    }

    // N-ary arithmetic functions ADD/SUB/MUL/DIV (fold left; DIV is signed
    // integer division for ints, float division for reals).
    static const std::map<std::string, int> arithFns = {
        {"ADD", 0}, {"SUB", 1}, {"MUL", 2}, {"DIV", 3}};
    if (auto it = arithFns.find(up); it != arithFns.end() && args.size() >= 2) {
      mlir::Type stdTy = commonStdTy(args);
      mlir::Value acc = toStd(args[0], stdTy);
      for (size_t i = 1; i < args.size(); ++i) {
        mlir::Value v = toStd(args[i], stdTy);
        if (llvm::isa<mlir::FloatType>(stdTy)) {
          switch (it->second) {
          case 0: acc = mlir::sclang::withFPFastMath(builder.create<mlir::arith::AddFOp>(location, acc, v)).getResult(); break;
          case 1: acc = mlir::sclang::withFPFastMath(builder.create<mlir::arith::SubFOp>(location, acc, v)).getResult(); break;
          case 2: acc = mlir::sclang::withFPFastMath(builder.create<mlir::arith::MulFOp>(location, acc, v)).getResult(); break;
          default: acc = mlir::sclang::withFPFastMath(builder.create<mlir::arith::DivFOp>(location, acc, v)).getResult(); break;
          }
        } else {
          switch (it->second) {
          case 0: acc = builder.create<mlir::arith::AddIOp>(location, acc, v).getResult(); break;
          case 1: acc = builder.create<mlir::arith::SubIOp>(location, acc, v).getResult(); break;
          case 2: acc = builder.create<mlir::arith::MulIOp>(location, acc, v).getResult(); break;
          default: acc = builder.create<mlir::arith::DivSIOp>(location, acc, v).getResult(); break;
          }
        }
      }
      return fromStd(acc, commonSclTy(args));
    }

    // Bit-string shift/rotate SHL/SHR/ROL/ROR(x, n): x is a BYTE/WORD/DWORD
    // (LogicalType) or int; n an INT count. SHL/SHR clamp the count at the
    // width (result 0); ROL/ROR rotate modulo the width.
    if ((up == "SHL" || up == "SHR" || up == "ROL" || up == "ROR") &&
        args.size() == 2) {
      unsigned w = 8;
      if (auto it = llvm::dyn_cast<IntegerType>(t0))
        w = it.getWidth();
      else if (auto lt = llvm::dyn_cast<LogicalType>(t0))
        w = lt.getWidth();
      auto stdTy = mlir::IntegerType::get(ctx, w);
      mlir::Value x = toStd(args[0], stdTy);
      auto i32 = mlir::IntegerType::get(ctx, 32);
      mlir::Value n32 = toStd(args[1], i32);
      auto c32 = [&](int64_t v) {
        return builder.create<mlir::arith::ConstantOp>(
            location, i32, builder.getIntegerAttr(i32, v));
      };
      auto truncW = [&](mlir::Value v) -> mlir::Value {
        if (v.getType() == stdTy)
          return v;
        return builder.create<mlir::arith::TruncIOp>(location, stdTy, v)
            .getResult();
      };
      mlir::Value res;
      if (up == "SHL" || up == "SHR") {
        // shift >= w yields 0 (avoid poisonous over-shift)
        auto ge = builder.create<mlir::arith::CmpIOp>(
            location, mlir::arith::CmpIPredicate::uge, n32, c32(w));
        mlir::Value sh = up == "SHL"
            ? (mlir::Value)builder.create<mlir::arith::ShLIOp>(location, x, truncW(n32)).getResult()
            : (mlir::Value)builder.create<mlir::arith::ShRUIOp>(location, x, truncW(n32)).getResult();
        auto zero = builder.create<mlir::arith::ConstantOp>(
            location, stdTy, builder.getIntegerAttr(stdTy, 0));
        res = builder.create<mlir::arith::SelectOp>(location, ge.getResult(), zero, sh).getResult();
      } else {
        // rotate: n mod w; (w - n) & (w-1) keeps the complementary shift in
        // range (and yields 0 when n mod w == 0, making rol(x,0) = x)
        auto mask = builder.create<mlir::arith::ConstantOp>(
            location, i32, builder.getIntegerAttr(i32, (int64_t)(w - 1)));
        mlir::Value nM = builder.create<mlir::arith::AndIOp>(location, n32, mask).getResult();
        mlir::Value comp = builder.create<mlir::arith::AndIOp>(
            location,
            builder.create<mlir::arith::SubIOp>(location, c32(w), nM).getResult(),
            mask).getResult();
        mlir::Value a = up == "ROL"
            ? (mlir::Value)builder.create<mlir::arith::ShLIOp>(location, x, truncW(nM)).getResult()
            : (mlir::Value)builder.create<mlir::arith::ShRUIOp>(location, x, truncW(nM)).getResult();
        mlir::Value b = up == "ROL"
            ? (mlir::Value)builder.create<mlir::arith::ShRUIOp>(location, x, truncW(comp)).getResult()
            : (mlir::Value)builder.create<mlir::arith::ShLIOp>(location, x, truncW(comp)).getResult();
        res = builder.create<mlir::arith::OrIOp>(location, a, b).getResult();
      }
      return fromStd(res, t0);
    }

    // EXPT(a, b) = a ** b (float pow if any REAL, else integer pow).
    if (up == "EXPT" && args.size() == 2) {
      mlir::Type stdTy = commonStdTy(args);
      mlir::Value a = toStd(args[0], stdTy), b = toStd(args[1], stdTy);
      mlir::Value res = llvm::isa<mlir::FloatType>(stdTy)
          ? (mlir::Value)builder.create<mlir::math::PowFOp>(location, a, b).getResult()
          : (mlir::Value)builder.create<mlir::math::IPowIOp>(location, a, b).getResult();
      return fromStd(res, commonSclTy(args));
    }

    // TRUNC(r): REAL -> integer, rounding toward zero (arith.truncf).
    if (up == "TRUNC" && args.size() == 1 && llvm::isa<RealType>(t0)) {
      auto f32 = mlir::Float32Type::get(ctx);
      mlir::Value v = toStd(args[0], f32);
      auto i32 = mlir::IntegerType::get(ctx, 32);
      mlir::Value r = builder.create<mlir::arith::FPToSIOp>(location, i32, v).getResult();
      return fromStd(r, IntegerType::get(ctx, 32));
    }

    if (up == "ABS" && args.size() == 1 && llvm::isa<IntegerType>(t0)) {
      auto it = llvm::cast<IntegerType>(t0);
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
        return llvm::isa<RealType>(v.getType());
      });
      mlir::Type resTy = t0;
      mlir::Type stdTy;
      if (isFloat)
        stdTy = mlir::Float32Type::get(ctx);
      else {
        unsigned w = 32;
        for (auto a : args)
          if (auto it2 = llvm::dyn_cast<IntegerType>(a.getType()))
            w = std::max<unsigned>(w, it2.getWidth());
        stdTy = mlir::IntegerType::get(ctx, w);
      }
      mlir::Value acc = toStd(args[0], stdTy);
      for (size_t i = 1; i < args.size(); ++i) {
        mlir::Value v = toStd(args[i], stdTy);
        // max/min are order-insensitive for values, so arg order is fine.
        if (isFloat)
          acc = it->second
                    ? mlir::sclang::withFPFastMath(builder.create<mlir::arith::MaximumFOp>(location, acc, v)).getResult()
                    : mlir::sclang::withFPFastMath(builder.create<mlir::arith::MinimumFOp>(location, acc, v)).getResult();
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
      if (llvm::isa<RealType>(t0))
        stdTy = mlir::Float32Type::get(ctx);
      else if (auto it2 = llvm::dyn_cast<IntegerType>(args[1].getType()))
        stdTy = mlir::IntegerType::get(ctx, it2.getWidth());
      mlir::Value in0 = toStd(args[1], stdTy), in1 = toStd(args[2], stdTy);
      auto res = builder.create<mlir::arith::SelectOp>(location, g, in1, in0).getResult();
      return fromStd(res, llvm::isa<RealType>(t0) ? t0 : args[1].getType());
    }
    if (up == "LIMIT" && args.size() == 3) {
      // LIMIT(MN, IN, MX) = MIN(MAX(IN, MN), MX)
      bool isFloat = llvm::any_of(args, [](mlir::Value v) {
        return llvm::isa<RealType>(v.getType());
      });
      mlir::Type stdTy = isFloat ? (mlir::Type)mlir::Float32Type::get(ctx)
                                 : (mlir::Type)mlir::IntegerType::get(ctx, 32);
      if (!isFloat)
        for (auto a : args)
          if (auto it2 = llvm::dyn_cast<IntegerType>(a.getType()))
            stdTy = mlir::IntegerType::get(
                ctx, std::max<unsigned>(
                         it2.getWidth(),
                         llvm::cast<mlir::IntegerType>(stdTy).getWidth()));
      mlir::Value mn = toStd(args[0], stdTy), in = toStd(args[1], stdTy),
                  mx = toStd(args[2], stdTy);
      mlir::Value r;
      if (isFloat)
        r = mlir::sclang::withFPFastMath(builder.create<mlir::arith::MaximumFOp>(location, in, mn)).getResult();
      else
        r = builder.create<mlir::arith::MaxSIOp>(location, in, mn).getResult();
      if (isFloat)
        r = mlir::sclang::withFPFastMath(builder.create<mlir::arith::MinimumFOp>(location, r, mx)).getResult();
      else
        r = builder.create<mlir::arith::MinSIOp>(location, r, mx).getResult();
      return fromStd(r, args[1].getType());
    }
    if (up == "MUX" && args.size() >= 2) {
      // MUX(K, IN0..INn): nested selects, INn is the ELSE
      mlir::Value k = toStd(args[0], mlir::IntegerType::get(ctx, 32));
      mlir::Type stdTy = mlir::Float32Type::get(ctx);
      bool isFloat = false;
      for (size_t i = 1; i < args.size(); ++i)
        if (llvm::isa<RealType>(args[i].getType()))
          isFloat = true;
      if (!isFloat) {
        unsigned w = 32;
        for (size_t i = 1; i < args.size(); ++i)
          if (auto it2 = llvm::dyn_cast<IntegerType>(args[i].getType()))
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
                            bool allowVoid = false,
                            mlir::Value selfAddr = nullptr) {
    auto location = loc(expr->loc());
    auto context = builder.getContext();

    // SPEC[SCL] Test-support builtins (awlsim-style oracles in SCL form,
    // used by test/conformance/run_st_traces.py and SCL oracle lit tests):
    //   __ASSERT(<bool expr>) -> awl.assert (id=line, op ==, actual=i64(expr),
    //                            expected=1) -> scl_rt_assert_cmp
    //   __BITS(<value>)       -> bit.cast to the same-width integer (lets the
    //                            trace driver observe REAL bit patterns)
    //   __STRTO_M(<byteBase>, <string lvalue>)
    //                         -> scl_rt_str_to_m: copy the S7-layout string
    //                            buffer into the M area at byteBase (lets
    //                            --scan-io observe STRING outputs)
    std::string upTest = std::string(callee.upper());
    if (upTest == "__STRTO_M") {
      auto params = expr->getParameters();
      if (params.size() != 3) {
        emitError(location) << "__STRTO_M needs exactly three arguments "
                               "(byte base, max length, string lvalue)";
        return nullptr;
      }
      mlir::Value base = mlirGen(params[0].get());
      mlir::Value maxLen = mlirGen(params[1].get());
      if (!base || !maxLen)
        return nullptr;
      auto i32 = builder.getI32Type();
      if (base.getType() != i32)
        base = builder.create<DialectCastOp>(location, i32, base);
      if (maxLen.getType() != i32)
        maxLen = builder.create<DialectCastOp>(location, i32, maxLen);
      // The string argument is taken as an lvalue: its !scl.address is the
      // buffer pointer passed to the runtime (lowers to i64).
      mlir::Value strAddr = mlirGenLValue(params[2].get());
      if (!strAddr || !llvm::isa<AddressType>(strAddr.getType())) {
        emitError(location) << "__STRTO_M: third argument must be a string "
                               "lvalue";
        return nullptr;
      }
      // The string buffer's raw data pointer as i64 (DialectCastOp on a
      // byte view lowers to the memref base+offset pointer).
      auto i64 = mlir::IntegerType::get(context, 64);
      mlir::Value strPtr =
          builder.create<DialectCastOp>(location, i64, strAddr);
      auto fnTy =
          builder.getFunctionType({i32, i32, i64}, {getType(tok_dint)});
      auto stub = mlir::SymbolTable::lookupNearestSymbolFrom<FunctionOp>(
          theModule, builder.getStringAttr("scl_rt_str_to_m"));
      if (!stub) {
        mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
        stub = b.create<FunctionOp>(location, "scl_rt_str_to_m", fnTy);
        stub.setPrivate();
      }
      builder.create<CallFcOp>(location, getType(tok_dint), "scl_rt_str_to_m",
                               mlir::ValueRange{base, maxLen, strPtr},
                               mlir::ArrayAttr::get(context, {}),
                               nullptr, nullptr);
      return nullptr; // statement, no value
    }
    //   __LOADM64(<byteBase>)         -> scl_rt_read('M', 0, base, 8) as i64
    //   __STOREM64(<byteBase>, <v>)   -> scl_rt_write('M', 0, base, 8, v)
    // 64-bit M-area access, used by the trace harness for LREAL bit patterns
    // (a %MD direct is only 32 bit wide).
    if (upTest == "__LOADM64" || upTest == "__STOREM64") {
      auto params = expr->getParameters();
      if (params.size() != (upTest == "__LOADM64" ? 1 : 2)) {
        emitError(location) << upTest << " needs "
                            << (upTest == "__LOADM64" ? "one" : "two")
                            << " argument(s) (byte base[, value])";
        return nullptr;
      }
      mlir::Value base = mlirGen(params[0].get());
      if (!base)
        return nullptr;
      auto i8 = builder.getI8Type();
      auto i32 = builder.getI32Type();
      auto i64 = mlir::IntegerType::get(context, 64);
      if (base.getType() != i32)
        base = builder.create<DialectCastOp>(location, i32, base);
      auto areaC = builder.create<ConstantOp>(
          location, i8, builder.getIntegerAttr(i8, (uint8_t)'M'));
      auto dbC = builder.create<ConstantOp>(location, i32,
                                            builder.getIntegerAttr(i32, 0));
      auto widthC = builder.create<ConstantOp>(location, i8,
                                               builder.getIntegerAttr(i8, 8));
      // func.call directly (not scl.call.fc): CallFcOp's result must be an
      // Scl_Type, and the 64-bit pattern needs a builtin i64.
      if (upTest == "__LOADM64") {
        auto fnTy = builder.getFunctionType({i8, i32, i32, i8}, {i64});
        auto stub = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
            theModule, builder.getStringAttr("scl_rt_read"));
        if (!stub) {
          mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
          stub = b.create<mlir::func::FuncOp>(location, "scl_rt_read", fnTy);
          stub.setPrivate();
        }
        return builder
            .create<mlir::func::CallOp>(location, i64, "scl_rt_read",
                                        mlir::ValueRange{areaC, dbC, base, widthC})
            .getResult(0);
      }
      mlir::Value val = mlirGen(params[1].get());
      if (!val)
        return nullptr;
      if (val.getType() != i64)
        val = builder.create<DialectCastOp>(location, i64, val);
      auto fnTy = builder.getFunctionType({i8, i32, i32, i8, i64}, {});
      auto stub = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          theModule, builder.getStringAttr("scl_rt_write"));
      if (!stub) {
        mlir::OpBuilder b(theModule.getBody(), theModule.getBody()->begin());
        stub = b.create<mlir::func::FuncOp>(location, "scl_rt_write", fnTy);
        stub.setPrivate();
      }
      builder.create<mlir::func::CallOp>(location, mlir::TypeRange{},
                                         "scl_rt_write",
                                         mlir::ValueRange{areaC, dbC, base,
                                                          widthC, val});
      return nullptr; // statement, no value
    }
    if (upTest == "__ASSERT" || upTest == "__BITS" || upTest == "__UNBITS") {
      mlir::Value arg;
      if (failed(mlirGenCallParameters(
              expr, [&](StringRef, mlir::Value v) { arg = v; },
              [&](mlir::Value v) { arg = v; })))
        return nullptr;
      if (!arg) {
        emitError(location) << upTest << " needs exactly one argument";
        return nullptr;
      }
      if (upTest == "__UNBITS") {
        // __UNBITS(intBits): bit-cast a 32-bit pattern back to REAL (64-bit
        // to LREAL) — the inverse of __BITS; used by the trace harness to
        // feed REAL/LREAL inputs.
        unsigned argWidth = 0;
        if (auto it = llvm::dyn_cast<IntegerType>(arg.getType()))
          argWidth = it.getWidth();
        else if (auto it = llvm::dyn_cast<mlir::IntegerType>(arg.getType()))
          argWidth = it.getWidth();
        auto resTy = argWidth == 64 ? getType(tok_lreal) : getType(tok_real);
        return builder.create<BitCastOp>(location, resTy, arg).getResult();
      }
      if (upTest == "__BITS") {
        unsigned width;
        if (auto it = llvm::dyn_cast<IntegerType>(arg.getType()))
          width = it.getWidth();
        else if (auto lt = llvm::dyn_cast<LogicalType>(arg.getType()))
          width = lt.getWidth();
        else if (llvm::isa<LrealType>(arg.getType()))
          width = 64;
        else if (llvm::isa<RealType>(arg.getType()) || llvm::isa<TimeType>(arg.getType()))
          width = 32;
        else {
          emitError(location) << "__BITS: unsupported argument type "
                              << arg.getType();
          return nullptr;
        }
        mlir::Type resTy =
            width > 32
                ? mlir::Type(mlir::IntegerType::get(context, width))
                : mlir::Type(IntegerType::get(context, width));
        return builder.create<BitCastOp>(location, resTy, arg).getResult();
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
    // by-ref params of this call: (cell, target address, type) copied back
    // after the call (VAR_IN_OUT / VAR_OUTPUT passed by copy-in/copy-out)
    SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Type>, 4>
        fcWriteBacks;
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

    // Rebuild the argument list in the callee's declared parameter order
    // (IEC ed.3 6.6.1.7.1): omitted inputs take their declared default;
    // VAR_IN_OUT / VAR_OUTPUT parameters are passed by address (the `=>`
    // binding for outputs, `:= var` for in/outs; unbound by-ref params get a
    // fresh temp cell) so the callee reads/writes the caller's variable.
    auto allIt = callableAllOrder.find(callee);
    if (allIt == callableAllOrder.end())
      for (auto &kv : callableAllOrder)
        if (kv.getKey().equals_insensitive(callee)) {
          allIt = callableAllOrder.find(kv.getKey());
          break;
        }
    if (allIt != callableAllOrder.end()) {
      auto inoutSet = callableInOut.find(callee);
      if (inoutSet == callableInOut.end())
        for (auto &kv : callableInOut)
          if (kv.getKey().equals_insensitive(callee)) {
            inoutSet = callableInOut.find(kv.getKey());
            break;
          }
      auto outSet = callableOutputs.find(callee);
      if (outSet == callableOutputs.end())
        for (auto &kv : callableOutputs)
          if (kv.getKey().equals_insensitive(callee)) {
            outSet = callableOutputs.find(kv.getKey());
            break;
          }
      auto defIt = callableVarDefaults.find(callee);
      if (defIt == callableVarDefaults.end())
        for (auto &kv : callableVarDefaults)
          if (kv.getKey().equals_insensitive(callee)) {
            defIt = callableVarDefaults.find(kv.getKey());
            break;
          }
      auto containsCI = [](const std::set<std::string> *st,
                           StringRef n) -> bool {
        if (!st)
          return false;
        for (auto &e : *st)
          if (llvm::StringRef(e).equals_insensitive(n))
            return true;
        return false;
      };
      // collect the call's bindings: named (name -> rhs, isOutput, origIdx)
      // + positional (rhs, origIdx); origIdx indexes the already-evaluated
      // `arguments` from the first mlirGenCallParameters pass, so argument
      // expressions are not evaluated twice
      auto origArgs = arguments;
      auto origNames = argNames;
      llvm::StringMap<std::tuple<const ExpressionAST *, bool, size_t>> named;
      SmallVector<std::pair<const ExpressionAST *, size_t>, 4> positional;
      size_t origIdx = 0;
      for (auto &arg : expr->getParameters()) {
        auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
        bool isNamed = binary && (binary->getOp() == tok_assignment ||
                                  binary->getOp() == tok_assign_output);
        if (isNamed) {
          auto lhs = dyn_cast<SimpleVariableAST>(binary->getLhs());
          if (lhs)
            named[llvm::StringRef(lhs->getName()).upper()] = {
                binary->getRhs(), binary->getOp() == tok_assign_output,
                origIdx};
        } else {
          positional.push_back({arg.get(), origIdx});
        }
        ++origIdx;
      }
      SmallVector<mlir::Value, 8> fullArgs;
      SmallVector<mlir::Attribute, 8> fullNames;
      size_t posIdx = 0;
      for (auto &pname : allIt->second) {
        std::string pUp = llvm::StringRef(pname).upper();
        const ExpressionAST *rhs = nullptr;
        mlir::Value v0; // already-evaluated argument value from the first pass
        size_t v0Idx = origArgs.size();
        auto nit = named.find(pUp);
        if (nit != named.end()) {
          rhs = std::get<0>(nit->second);
          v0Idx = std::get<2>(nit->second);
        } else if (posIdx < positional.size()) {
          rhs = positional[posIdx].first;
          v0Idx = positional[posIdx].second;
          ++posIdx;
        }
        if (v0Idx < origArgs.size())
          v0 = origArgs[v0Idx];
        bool isInOut = containsCI(inoutSet != callableInOut.end() ? &inoutSet->second : nullptr, pname);
        bool isOutput = containsCI(outSet != callableOutputs.end() ? &outSet->second : nullptr, pname);
        mlir::Value v;
        if (!isInOut && !isOutput) {
          if (!rhs) {
            // omitted input: declared default (IEC ed.3 6.6.1.7.1)
            mlir::Type ty = getType(tok_int);
            int64_t dv = 0;
            bool hasDef = false;
            if (defIt != callableVarDefaults.end())
              for (auto &kv : defIt->second)
                if (kv.getKey().equals_insensitive(pname) && kv.second.first) {
                  dv = kv.second.second;
                  hasDef = true;
                }
            if (paramTypes)
              for (auto &kv : *paramTypes)
                if (kv.getKey().equals_insensitive(pname))
                  ty = kv.second;
            if (!hasDef)
              dv = 0; // omitted input without a declared default: zero
            unsigned w = 16;
            if (auto it = llvm::dyn_cast<IntegerType>(ty))
              w = it.getWidth();
            else if (auto lt = llvm::dyn_cast<LogicalType>(ty))
              w = lt.getWidth();
            v = builder.create<ConstantOp>(
                location, ty,
                builder.getIntegerAttr(mlir::IntegerType::get(context, w), dv));
          } else if (auto *str = dyn_cast<StringConstantAST>(rhs)) {
            v = mlirGenStringLiteralValue(loc(rhs->loc()), paramTypes, pname, str);
            if (!v)
              return nullptr;
          } else {
            v = v0 ? v0 : mlirGenRValue(rhs);
            if (!v)
              return nullptr;
          }
        } else {
          // By-reference parameter: stage through a temp cell of the declared
          // type — copy the bound variable's value in before the call, copy it
          // back after (the callee sees a 0-d typed view of the cell).
          mlir::Type pty = getType(tok_int);
          if (paramTypes)
            for (auto &kv : *paramTypes)
              if (kv.getKey().equals_insensitive(pname))
                pty = kv.second;
          auto addrTy = AddressType::get(pty);
          v = builder.create<TempVariableOp>(location, addrTy,
                                             builder.getStringAttr(pname));
          if (rhs) {
            mlir::Value val = v0 ? v0 : mlirGenRValue(rhs);
            if (!val)
              return nullptr;
            builder.create<StoreOp>(location, v, val);
            mlir::Value target = mlirGen(rhs);
            if (!target)
              return nullptr;
            fcWriteBacks.push_back({v, target, pty});
          }
        }
        fullArgs.push_back(v);
        fullNames.push_back(mlir::StringAttr::get(context, pname));
      }
      arguments = std::move(fullArgs);
      argNames = std::move(fullNames);
    }

    // A METHOD call passes the instance-DB address as the first ($self)
    // argument.
    if (selfAddr) {
      arguments.insert(arguments.begin(), selfAddr);
      argNames.insert(argNames.begin(),
                      mlir::StringAttr::get(context, "$self"));
    }

    auto rtIt = functionResultTypes.find(callee);
    if (rtIt == functionResultTypes.end()) {
      // TIA DATE_AND_TIME field extractor / DT_TO_DATE, DT_TO_TOD (these
      // contain "_TO_" and must be claimed before the generic conversions)?
      if (mlir::Value v = mlirGenDtField(location, expr, callee))
        return v;
      // IEC conversion builtin (INT_TO_WORD, TIME_TO_DINT, ...)?
      if (mlir::Value v = mlirGenBuiltinConversion(location, expr, callee))
        return v;
      // IEC 61131-3 standard functions (ABS/MAX/MIN/SEL/MUX/LIMIT and the
      // real math functions)?
      if (mlir::Value v = mlirGenStdFunction(location, expr, callee))
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
                                      arguments, argNamesAttr, nullptr, nullptr);
    }
    mlir::Type resultType = rtIt->second;
    if (llvm::isa<mlir::NoneType>(resultType)) {
      if (!allowVoid)
        emitError(location) << "void function '" << callee << "' used as a value";
      return nullptr;
    }

    auto argNamesAttr = mlir::ArrayAttr::get(context, argNames);
    mlir::Value callRes =
        builder.create<CallFcOp>(location, resultType, callee, arguments,
                                 argNamesAttr, nullptr, nullptr);
    for (auto &wb : fcWriteBacks) {
      auto val = builder.create<LoadOp>(location, std::get<2>(wb), std::get<0>(wb));
      builder.create<StoreOp>(location, std::get<1>(wb), val);
    }
    return callRes;
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
    if (llvm::isa<AddressType>(t))
      t = llvm::cast<AddressType>(t).getElementType();
    if (auto idbTy = llvm::dyn_cast<InstanceDbType>(t))
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
    // `out => var` bindings: written back from the instance member after the
    // call (IEC ed.3 6.6.2.3 formal call output assignment). VAR_IN_OUT
    // parameters bound with `:= var` are written back the same way (an
    // approximation of by-reference semantics: the value is stored into the
    // instance member before the call and copied back after it).
    // (instance member address, caller target address, member type) triples
    // written back after the call: `out => var` (formal output assignment,
    // IEC ed.3 6.6.2.3) and VAR_IN_OUT `inout := var` (approximated as
    // copy-in before / copy-out after the call).
    SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Type>, 4>
        outBindings;
    {
      auto ioIt = callableInOut.find(fb);
      if (ioIt == callableInOut.end())
        for (auto &kv : callableInOut)
          if (kv.getKey().equals_insensitive(fb)) {
            ioIt = callableInOut.find(kv.getKey());
            break;
          }
      if (ioIt != callableInOut.end())
        for (auto &arg : expr->getParameters()) {
          auto binary = dyn_cast<BinaryExpressionAST>(arg.get());
          if (!binary || binary->getOp() != tok_assignment)
            continue;
          auto lhs = dyn_cast<SimpleVariableAST>(binary->getLhs());
          if (!lhs)
            continue;
          bool isInOut = false;
          for (auto &e : ioIt->second)
            if (llvm::StringRef(e).equals_insensitive(lhs->getName()))
              isInOut = true;
          if (!isInOut)
            continue;
          mlir::Value elem = mlirGenElement(location, idbRef, lhs->getName());
          mlir::Value target = mlirGen(binary->getRhs());
          if (!elem || !target)
            continue;
          auto it = byteViewTypes.find(elem);
          mlir::Type ty = it != byteViewTypes.end()
                               ? it->second
                               : llvm::cast<AddressType>(target.getType()).getElementType();
          outBindings.push_back({elem, target, ty});
        }
    }
    // The implicit EN input (IEC ed.3 6.6.2.2): captured here and used to
    // gate the call below; ENO is stored after the call.
    mlir::Value enVal;
    if (failed(mlirGenCallParameters(expr, [&] (StringRef name, mlir::Value value) {
      if (name.equals_insensitive("EN")) {
        enVal = value;
        return mlir::success();
      }
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
        if (it == byteViewTypes.end() || !llvm::isa<ArrayType>(it->second))
          return mlir::failure(); // not a string member: fall back to value
        return mlirGenStoreStringBytes(location, elem,
                                       llvm::cast<ArrayType>(it->second),
                                       str->getValue());
      },
      [&] (StringRef name, const BinaryExpressionAST *binary)
          -> mlir::LogicalResult {
        mlir::Value elem = mlirGenElement(location, idbRef, name);
        mlir::Value target = mlirGen(binary->getRhs());
        if (!elem || !target)
          return mlir::failure();
        auto it = byteViewTypes.find(elem);
        mlir::Type ty = it != byteViewTypes.end()
                             ? it->second
                             : llvm::cast<AddressType>(target.getType()).getElementType();
        outBindings.push_back({elem, target, ty});
        return mlir::success();
      })))
        return mlir::failure();

    // Unknown FB (TIA library FB): declare an empty stub so the symbolic
    // call verifies and lowers to a no-op (like the awl.call stubs).
    // Builtin standard FBs are exempt: they lower to scl_rt_*_fb runtime
    // calls (no stub op, so the forced layout is not shadowed).
    // A user-defined FB with the same name wins over the TIMER/COUNTER
    // builtin aliases (e.g. an IEC PROGRAM named "counter").
    bool builtin = !builtinStandardFb(fb).empty() &&
                   !fbNames.count(llvm::StringRef(unquote(fb)).upper());
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
    // The call plus the output write-backs, gated by EN when bound
    // (IEC ed.3: with EN=FALSE the FB body is not executed).
    auto emitCall = [&]() {
      builder.create<CallFbOp>(location, unquote(fb), idbRef, nullptr, nullptr);
      for (auto &ob : outBindings) {
        auto val = builder.create<LoadOp>(location, std::get<2>(ob), std::get<0>(ob));
        builder.create<StoreOp>(location, std::get<1>(ob), val);
      }
    };
    if (enVal) {
      auto cond = coerceToBool(location, enVal);
      auto ifOp = builder.create<IfThenElseOp>(location, cond);
      builder.createBlock(&ifOp.getThenBody());
      emitCall();
      createEndOp(location);
      builder.createBlock(&ifOp.getElseBody());
      createEndOp(location);
      builder.setInsertionPointAfter(ifOp);
    } else {
      emitCall();
    }
    // ENO := EN (the body cannot fail); stored in a per-instance cell so a
    // later `inst.ENO` read observes it.
    {
      auto eno = fbEnoCell(location, idbRef);
      mlir::Value v = enVal ? coerceToBool(location, enVal)
                            : builder.create<ConstantOp>(
                                  location, getType(tok_bool),
                                  builder.getIntegerAttr(builder.getI1Type(), 1));
      builder.create<StoreOp>(location, eno, v);
    }
    return mlir::success();
  }

  /// Storage cell for the implicit ENO pin of an FB instance (keyed by the
  /// instance address's defining op), initialized FALSE.
  mlir::Value fbEnoCell(mlir::Location location, mlir::Value idbRef) {
    // Stable per-instance key, resolved recursively through the address
    // chain: a global DB by symbol name; a member access by the member name
    // within its base; else the defining op.
    std::function<std::string(mlir::Value)> keyOf = [&](mlir::Value v) {
      std::string k;
      if (auto *def = v.getDefiningOp()) {
        if (auto gg = llvm::dyn_cast<GetGlobalOp>(def))
          k = "db:" + gg.getName().str();
        else if (auto gv = llvm::dyn_cast<GetVariableOp>(def))
          k = keyOf(gv.getAddr()) + "." + gv.getSymName().str();
        else if (auto ge = llvm::dyn_cast<GetElementOp>(def))
          k = keyOf(ge.getAddr()) + "." + ge.getName().str();
        else
          k = "op:" + std::to_string((uintptr_t)def);
      } else {
        k = "arg:" + std::to_string((uintptr_t)v.getAsOpaquePointer());
      }
      return k;
    };
    std::string key = keyOf(idbRef);
    auto it = enoCells.find(key);
    if (it != enoCells.end())
      return it->second;
    auto boolTy = getType(tok_bool);
    // Create the cell at the entry of the enclosing function so it dominates
    // every use (call sites may be inside IF branches).
    mlir::OpBuilder eb(builder);
    if (auto *parent = builder.getInsertionBlock()->getParentOp()) {
      if (auto fn = parent->getParentOfType<FunctionOp>())
        eb.setInsertionPointToStart(&fn.getBody().front());
      else if (auto fb = parent->getParentOfType<FunctionBlockOp>())
        eb.setInsertionPointToStart(&fb.getBody().front());
    }
    auto cell = eb.create<TempVariableOp>(location, AddressType::get(boolTy),
                                          eb.getStringAttr("ENO"));
    eb.create<StoreOp>(
        location, cell,
        eb.create<ConstantOp>(location, boolTy,
                              eb.getIntegerAttr(eb.getI1Type(), 0)));
    enoCells[key] = cell;
    return cell;
  }

  /// Shorthand: scl integer constant with result type `sclTy` and an
  /// integer attribute of the given bit `width`.
  ConstantOp intConst(mlir::Location loc, mlir::Type sclTy, unsigned width,
                     int64_t v) {
    return builder.create<ConstantOp>(
        loc, sclTy, builder.getIntegerAttr(builder.getIntegerType(width), v));
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
    case tok_lreal:
      return LrealType::get(builder.getContext());
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
    case ElementaryDataTypeAST::Type_Lreal:
      return getType(tok_lreal);
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
    // S5 TIMER / COUNTER variables are instances of the builtin TON / CTU —
    // unless the project declares its own FB under that name (the soft
    // keywords may also name user FBs, e.g. FUNCTION_BLOCK COUNTER).
    case ElementaryDataTypeAST::Type_Timer:
      return InstanceDbType::get(
          builder.getContext(),
          fbNameOrig.count("TIMER") ? fbNameOrig["TIMER"] : "TON");
    case ElementaryDataTypeAST::Type_Counter:
      return InstanceDbType::get(
          builder.getContext(),
          fbNameOrig.count("COUNTER") ? fbNameOrig["COUNTER"] : "CTU");
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
        // enumerated value (CASE labels like `GREEN:`)?
        auto it = enumConsts.find(llvm::StringRef(v->getName()).upper());
        if (it != enumConsts.end()) {
          value = it->second;
          return mlir::success();
        }
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
        // enumerated value (CASE labels like `GREEN:`)?
        auto it = enumConsts.find(llvm::StringRef(constName).upper());
        if (it != enumConsts.end()) {
          value = (int)it->second;
          return mlir::success();
        }
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
    if (fbNames.contains(llvm::StringRef(type->getName()).upper()))
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

  /// Enumerated types are stored as INT (i16); the enumerated values are
  /// registered as named constants (enumConsts, case-insensitive).
  mlir::Type getType(const EnumDataTypeSpecAST *type) {
    int64_t next = 0;
    for (auto &v : type->getValues()) {
      int64_t val = v.value.value_or(next);
      next = val + 1;
      enumConsts[llvm::StringRef(v.name).upper()] = val;
    }
    return getType(tok_int);
  }

  /// Subrange types use the storage of their base type; the range itself is
  /// not enforced (out-of-range assignment is implementation-defined).
  mlir::Type getType(const SubrangeDataTypeSpecAST *type) {
    return getType(type->getBase());
  }

  mlir::Type getType(const DataTypeSpecAST *type) {
    return TypeSwitch<const DataTypeSpecAST *, mlir::Type>(type)
        .Case<ElementaryDataTypeAST>([&](auto type) { return getType(type); })
        .Case<StringDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<ArrayDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<StructDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<UserDefinedTypeIdentifierAST>([&](auto type) { return getType(type); })
        .Case<EnumDataTypeSpecAST>([&](auto type) { return getType(type); })
        .Case<SubrangeDataTypeSpecAST>([&](auto type) { return getType(type); })
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
    mlir::Value incr = nullptr;
    if (forDo->getIncrement().has_value())
      incr = mlirGenRValue(forDo->getIncrement().value());

    // The TO bound lives in a temp cell so it can be re-evaluated at the end
    // of every iteration (ed.3 7.3.3.4.2: modifying the final value inside the
    // loop is undefined; the reference traces record the re-read behavior).
    mlir::Value last = mlirGenRValue(forDo->getLast());
    if (!last)
      return mlir::failure();
    if (last.getType() != start.getType())
      last = builder.create<DialectCastOp>(location, start.getType(), last)
                 .getResult();
    auto boundCell = builder.create<TempVariableOp>(
        location, AddressType::get(start.getType()),
        builder.getStringAttr("$for_bound"));
    builder.create<StoreOp>(location, boundCell, last);

    auto forDoOp = builder.create<ForDoOp>(location, var, start, boundCell,
                                           incr);

    auto old = builder.saveInsertionPoint();

    builder.createBlock(&forDoOp.getDoBody());
    if (failed(mlirGen(forDo->getCode())))
      return mlir::failure();

    // Re-read the TO bound for the next iteration's condition check.
    mlir::Value lastAgain = mlirGenRValue(forDo->getLast());
    if (lastAgain) {
      if (lastAgain.getType() != start.getType())
        lastAgain = builder.create<DialectCastOp>(location, start.getType(),
                                                   lastAgain)
                        .getResult();
      builder.create<StoreOp>(location, boundCell, lastAgain);
    }

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
    else if (llvm::isa<ArrayType>(t)) {
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
    if (llvm::isa<RealType>(selector.getType()))
      return mlirGenRealCase(location, caseOf, selector);
    if (!llvm::isa<IntegerType>(selector.getType()) &&
        !llvm::isa<LogicalType>(selector.getType())) {
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
  return MLIRGenImpl(context, moduleAST.getMode()).mlirGen(&moduleAST);
}

} // namespace sclang
