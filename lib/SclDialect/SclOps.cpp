//===- Dialect.cpp - SCL IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the Scl IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::scl;

//===----------------------------------------------------------------------===//
// MARK: ConstantOp
//===----------------------------------------------------------------------===//

/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.
///
OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return value(); }

//===----------------------------------------------------------------------===//
// MARK: FunctionOp
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionOp(OpAsmParser &parser, OperationState &state) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return mlir::impl::parseFunctionLikeOp(parser, state, /*allowVariadic=*/false,
                                         buildFuncType);
}

static void print(FunctionOp fnOp, OpAsmPrinter &printer) {
  FunctionType fnType = fnOp.getType();
  mlir::impl::printFunctionLikeOp(printer, fnOp, fnType.getInputs(),
                                  /*isVariadic=*/false, fnType.getResults());
}

LogicalResult FunctionOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  if (getType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult FunctionOp::verifyBody() {
  FunctionType fnType = getType();
  auto walkResult = walk([fnType](Operation *op) -> WalkResult {
    if (auto retOp = dyn_cast<ReturnOp>(op)) {
      if (fnType.getNumResults() != 0)
        return retOp.emitOpError("cannot be used in functions returning value");
    } else if (auto retOp = dyn_cast<ReturnValueOp>(op)) {
      if (fnType.getNumResults() != 1)
        return retOp.emitOpError(
                   "returns 1 value but enclosing function requires ")
               << fnType.getNumResults() << " results";

      auto returnType = retOp.getReturnType();
      auto fnResultType = fnType.getResult(0);
      if (returnType != fnResultType)
        return retOp.emitOpError(" return value's type (")
               << returnType << ") mismatch with function's result type ("
               << fnResultType << ")";
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

// CallableOpInterface
Region *FunctionOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface
ArrayRef<Type> FunctionOp::getCallableResults() {
  return getType().getResults();
}

// MARK: CallFcOp

CallInterfaceCallable CallFcOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>(callee());
}

Operation::operand_range CallFcOp::getArgOperands() { return arguments(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sclang/SclDialect/SclOps.cpp.inc"
