//===- Ops.td - SCL dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the SCL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SCL_OPS
#define SCL_OPS

include "mlir/IR/OpBase.td"

// Provide a definition of the 'scl' dialect in the ODS framework so that we
// can define our operations.
def SCL_Dialect : Dialect {
  let name = "scl";
  let cppNamespace = "scl";
}

def scl_Type : Type<CPred<"scl::isa_scl_or_std_type($_self)">,
    "SCL dialect type">;

def scl_CharacterType : Type<CPred<"$_self.isa<scl::CharacterType>()">,
    "SCL character type">;
def scl_IntegerType : Type<CPred<"$_self.isa<scl::IntType>()">,
    "SCL integer type">;
def scl_RealType : Type<CPred<"$_self.isa<scl::RealType>()">,
    "SCL real type">;


// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class SCL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SCL_Dialect, mnemonic, traits>;

/*
def LoadOp : SCL_Op<"load"> {
  let summary = "load a value from memory";
  let description = [{
    The "load" operation reads a value from a data block.
  }];

  let arguments = (ins AnyMemRef);
  let results = (outs AnyType);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}

def StoreOp : SCL_Op<"store"> {
  let summary = "store a value to memory";
  let description = [{
    The "store" operation writes a value to a data block.
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyType:$rhs);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}
*/


def ContinueOp : SCL_Op<"continue", [Terminator, HasParent<"FuncOp">]> {
  let summary = "continue operation";
  let description = [{
    A CONTINUE statement is used to terminate the execution of the current iteration of a loop (initiated by a FOR, WHILE or REPEAT statement) and to restart processing within the loop.
  }];
}


def ReturnOp : SCL_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    A RETURN statement causes the program to exit the block (OB, FB or FC) currently being processed and to return to the calling block or the operating system if the block being exited is an OB.
  }];
}


def ExitOp : SCL_Op<"exit", [Terminator, HasParent<"FuncOp">]> {
  let summary = "exit operation";
  let description = [{
    An EXIT statement is used to exit a loop (FOR, WHILE or REPEAT loop) at any point regardless of whether the break condition is satisfied.

  }];
}

def MulOp : SCL_Op<"mul", [Commutative]> {
  let summary = "multiplication operation";
  let description = [{
    The "mul" operation performs multiplication between two values.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def AndOp : SCL_Op<"and"> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def OrOp : SCL_Op<"or"> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def XOrOp : SCL_Op<"xor"> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def UnaryNotOp : SCL_Op<"not"> {
  let summary = "unary not";
  let description = [{
    TBD
  }];

  let arguments = (ins AnyInteger:$rhs);
  let results = (outs AnyInteger);

  // Allow building a UnaryNotOp with from the input operand.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value rhs">
  ];
}

def UnaryMinusOp : SCL_Op<"negate"> {
  let summary = "negate";
  let description = [{
    TBD
  }];

  let arguments = (ins AnyType:$rhs);
  let results = (outs AnyType);

  // Allow building a UnaryMinusOp with from the input operand.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value rhs">
  ];
}
#endif // SCL_OPS
