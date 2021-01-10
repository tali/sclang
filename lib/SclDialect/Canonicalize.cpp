//===- Canonicalizer.cpp - Transformations for SCL Operations -------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This file implements the canonicalization transformations for custom
// SCL operations.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using llvm::SmallBitVector;

//===----------------------------------------------------------------------===//
// MARK: CallFcOp
//===----------------------------------------------------------------------===//

/// Reorders named function call arguments to match the order of the callee.
/// Also replaces omitted arguments with its default value.
struct CanonicalizeCallArguments : public OpRewritePattern<scl::CallFcOp> {

  CanonicalizeCallArguments(MLIRContext *context)
  : OpRewritePattern<scl::CallFcOp>(context, /*benefit=*/1) {}

  /// Fill in the argument name attribute from the referenced function.
  mlir::LogicalResult
  canonicalizeSingleArgument(scl::CallFcOp op, scl::FunctionOp func, PatternRewriter &rewriter) const {
    auto context = op->getContext();

    auto argCallNames = op.argNames().getValue();
    if (argCallNames.size() == 1)
      return failure();

    assert(argCallNames.size() == 0);
    auto nameId = Identifier::get("scl.name", context);
    auto nameAttr = func.getArgAttrOfType<StringAttr>(0, nameId);
    SmallVector<Attribute, 1> argNames;
    argNames.push_back(nameAttr);
    auto argNamesAttr = ArrayAttr::get(argNames, context);

    rewriter.replaceOpWithNewOp<scl::CallFcOp>(op, op.getType(), op.callee(),
                                               op.arguments(), argNamesAttr);

    return success();
  }

  /// Reorder arguments to match the referenced function.
  mlir::LogicalResult
  canonicalizeArguments(scl::CallFcOp op, scl::FunctionOp func, PatternRewriter &rewriter) const {
    auto context = op->getContext();
    auto funcType = func.getType();

    auto argCallNames = op.argNames().getValue();
    SmallBitVector argUsed(argCallNames.size());

    // map named arguments
    bool reordered = false;
    SmallVector<Value, 4> arguments;
    SmallVector<Attribute, 4> argNames;
    arguments.reserve(func.getNumArguments());
    argNames.reserve(func.getNumArguments());
    auto nameId = Identifier::get("scl.name", context);
    auto defaultId = Identifier::get("scl.default", context);
    auto defaultLocId = Identifier::get("scl.defaultLoc", context);
    for (unsigned i = 0; i < func.getNumArguments(); i++) {
      auto nameAttr = func.getArgAttrOfType<StringAttr>(i, nameId);
      argNames.push_back(nameAttr);
      StringRef name = nameAttr.getValue();
      bool found = false;
      for (unsigned j = 0; j < argCallNames.size(); j++) {
        auto argCallName = argCallNames[j].dyn_cast<StringAttr>().getValue();

        if (argCallName == name) {
          if (i != j)
            reordered = true;
          arguments.push_back(op.arguments()[j]);
          argUsed[j] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        reordered = true;
        auto defaultValue = func.getArgAttr(i, defaultId);
        if (defaultValue) {
          Dialect * scl = op->getDialect();
          auto loc = Location(func.getArgAttrOfType<LocationAttr>(i, defaultId));
          auto c = scl->materializeConstant(rewriter, defaultValue,
                                             funcType.getInput(i), loc);
          arguments.push_back(c->getResult(0));
        }
        assert(false && "have to provide value for argument");
        return failure();
      }
    }
    if (!argUsed.all() || !reordered)
      return failure();

    auto argNamesAttr = ArrayAttr::get(argNames, context);
    rewriter.replaceOpWithNewOp<scl::CallFcOp>(op, op.getType(), op.callee(),
                                               arguments, argNamesAttr);

    return success();
  }

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(scl::CallFcOp op, PatternRewriter &rewriter) const override {
    scl::FunctionOp func =
        SymbolTable::lookupNearestSymbolFrom<scl::FunctionOp>(op->getParentOp(),
                                                              op.callee());
    assert(func && "cannot lookup FunctionOp");

    switch (func.getNumArguments()) {
    case 0:
      return failure();
    case 1:
      return canonicalizeSingleArgument(op, func, rewriter);
    default:
      return canonicalizeArguments(op, func, rewriter);
    }
  }
};

void scl::CallFcOp::getCanonicalizationPatterns(
                    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CanonicalizeCallArguments>(context);
}

