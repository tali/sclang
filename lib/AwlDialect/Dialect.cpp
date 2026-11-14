//===- Dialect.cpp - AWL dialect registration ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sclang/AwlDialect/Dialect.h"

#include "sclang/AwlDialect/AwlOps.h"

using namespace mlir;
using namespace mlir::awl;

#include "sclang/AwlDialect/Dialect.cpp.inc"

void AwlDialect::initialize() { registerOps(); }
