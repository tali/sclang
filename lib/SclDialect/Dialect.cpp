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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
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
    IntegerAttr intAttr = value.cast<IntegerAttr>();
    return builder.create<ConstantS5TimeOp>(loc, type, intAttr);
  })
  .Case<TimeType>([&](auto type) {
    IntegerAttr intAttr = value.cast<IntegerAttr>();
    return builder.create<ConstantTimeOp>(loc, type, intAttr);
  })
  .Case<TimeOfDayType>([&](auto type) {
    IntegerAttr intAttr = value.cast<IntegerAttr>();
    return builder.create<ConstantTimeOfDayOp>(loc, type, intAttr);
  })
  .Case<DateType>([&](auto type) {
    IntegerAttr intAttr = value.cast<IntegerAttr>();
    return builder.create<ConstantDateOp>(loc, type, intAttr);
  })
  .Case<DateAndTimeType>([&](auto type) {
    IntegerAttr intAttr = value.cast<IntegerAttr>();
    return builder.create<ConstantDateAndTimeOp>(loc, type, intAttr);
  })
  .Default([&](auto type) {
    return builder.create<ConstantOp>(loc, type, value);
  });
}
