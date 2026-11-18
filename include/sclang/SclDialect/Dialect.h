//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
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
// This file implements the IR Dialect for the SCL language.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_DIALECT_H_
#define SCLANG_DIALECT_H_

#include "sclang/SclDialect/SclTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

/// Include the auto-generated header file containing the declaration of the SCL dialect.
#include "sclang/SclDialect/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// SCL operations.
#define GET_OP_CLASSES
#include "sclang/SclDialect/SclOps.h.inc"

namespace mlir {
namespace sclang {

/// Fast-math mode selected by the `--fp-mode` command line option
/// (default: contract, i.e. FMA contraction only; `fast` additionally
/// allows reassociation and NaN/Inf assumptions, `none` is strict IEEE).
/// Mirrors clang/gcc `-ffp-contract` + `-ffast-math` configuration.
arith::FastMathFlags fpFastMathFlags();

/// Detects whether an op class carries a fastmath attribute.
template <typename OpTy, typename = void>
struct HasFastmathAttrName : std::false_type {};
template <typename OpTy>
struct HasFastmathAttrName<
    OpTy, std::void_t<decltype(std::declval<OpTy>().getFastmathAttrName())>>
    : std::true_type {};

/// Set the fastmath attribute on an op that carries one (arith float ops),
/// according to the selected --fp-mode. No-op for ops without the attribute.
template <typename OpTy>
OpTy withFPFastMath(OpTy op) {
  auto flags = fpFastMathFlags();
  if constexpr (HasFastmathAttrName<OpTy>::value) {
    if (flags != arith::FastMathFlags::none)
      op->setAttr(op.getFastmathAttrName(),
                  arith::FastMathFlagsAttr::get(op->getContext(), flags));
  }
  return op;
}

} // namespace sclang
} // namespace mlir

#endif // SCLANG_DIALECT_H_
