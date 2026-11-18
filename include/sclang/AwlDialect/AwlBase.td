//===- AwlBase.td - AWL dialect base definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base definitions for the AWL (STL) dialect: parallel source dialect to
// `scl` (ADR 003), converging at std IR. Operands use !scl.direct from the
// scl base dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AWL_BASE
#define AWL_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Awl_Dialect : Dialect {
  let name = "awl";
  let cppNamespace = "::mlir::awl";
  let extraClassDeclaration = [{
  private:
    void registerOps();
  }];
}

// A !scl.direct<...> value (memory area / DB operand), shared with SCL.
def Awl_Direct : Type<CPred<"::llvm::isa<mlir::scl::DirectType>($_self)">,
                      "!scl.direct operand">;

// i1 (RLO: result of logic operation).
def Awl_Bool : Type<CPred<"$_self.isInteger(1)">, "i1 (RLO)">;

class AwlOp<string mnemonic, list<Trait> traits = []> :
    Op<Awl_Dialect, mnemonic, traits>;

class AwlTopLevelOp<string mnemonic, list<Trait> traits = []> :
    AwlOp<mnemonic, traits>;

#endif // AWL_BASE
