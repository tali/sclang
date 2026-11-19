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

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace {
enum class FpMode { None, Contract, Fast };
llvm::cl::opt<FpMode> fpModeOpt(
    "fp-mode",
    llvm::cl::desc("Floating-point fast-math mode for generated arith float ops"),
    llvm::cl::values(
        clEnumValN(FpMode::None, "none",
                   "no fast-math flags (strict IEEE 754)"),
        clEnumValN(FpMode::Contract, "contract",
                   "allow FMA contraction only (default; like clang "
                   "-ffp-contract=on)"),
        clEnumValN(FpMode::Fast, "fast",
                   "all fast-math flags incl. reassociation (like "
                   "-ffast-math)")),
    llvm::cl::init(FpMode::Contract));
} // namespace

namespace mlir {
namespace sclang {
arith::FastMathFlags fpFastMathFlags() {
  switch (fpModeOpt) {
  case FpMode::None:
    return arith::FastMathFlags::none;
  case FpMode::Contract:
    return arith::FastMathFlags::contract;
  case FpMode::Fast:
    return arith::FastMathFlags::fast;
  }
  return arith::FastMathFlags::contract; // defensive: default fp-mode
}
} // namespace sclang
} // namespace mlir
using namespace mlir::scl;

#include "sclang/SclDialect/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MARK: SclDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void SclDialect::initialize() {
  registerOps();
  registerTypes();
}

Operation *SclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return llvm::TypeSwitch<Type, Operation*>(type)
  .Case<S5TimeType>([&](auto type) {
    IntegerAttr intAttr = llvm::cast<IntegerAttr>(value);
    return builder.create<ConstantS5TimeOp>(loc, type, intAttr);
  })
  .Case<TimeType>([&](auto type) {
    IntegerAttr intAttr = llvm::cast<IntegerAttr>(value);
    return builder.create<ConstantTimeOp>(loc, type, intAttr);
  })
  .Case<TimeOfDayType>([&](auto type) {
    IntegerAttr intAttr = llvm::cast<IntegerAttr>(value);
    return builder.create<ConstantTimeOfDayOp>(loc, type, intAttr);
  })
  .Case<DateType>([&](auto type) {
    IntegerAttr intAttr = llvm::cast<IntegerAttr>(value);
    return builder.create<ConstantDateOp>(loc, type, intAttr);
  })
  .Case<DateAndTimeType>([&](auto type) {
    IntegerAttr intAttr = llvm::cast<IntegerAttr>(value);
    return builder.create<ConstantDateAndTimeOp>(loc, type, intAttr);
  })
  .Default([&](auto type) {
    return builder.create<ConstantOp>(loc, type, value);
  });
}
