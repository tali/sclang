//===- Dialect.h - AWL dialect definition --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_AWLDIALECT_DIALECT_H_
#define SCLANG_AWLDIALECT_DIALECT_H_

#include "sclang/SclDialect/SclTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sclang/AwlDialect/Dialect.h.inc"

#endif // SCLANG_AWLDIALECT_DIALECT_H_
