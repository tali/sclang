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

def Scl_StructType : Type<CPred<"scl::isa<::StructType>()">, "SCL STRUCT">;

def Scl_Type : AnyTypeOf<[Scl_StructType]>;

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

class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    SCL_Op<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs AnyType)>;
class CompareOp<string mnemonic, list<OpTrait> traits = []> :
    SCL_Op<mnemonic, !listconcat(traits, [SameTypeOperands])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs I1)>;
class UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    SCL_Op<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$rhs)>,
    Results<(outs AnyType)>;

class FuncTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    SCL_Op<mnemonic, !listconcat(traits, [Terminator, HasParent<"FuncOp">])>;

//
// MARK: variable handling
//

def TempVariableOp : SCL_Op<"tmpvar"> {
  let summary = "allocate stack memory for a temporary variable";
  let description = [{ TBD }];
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyMemRef);
}

def ConstantOp : SCL_Op<"constant", [NoSideEffect]> {
  let summary = "literal constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);
}

def LoadOp : SCL_Op<"load"> {
  let summary = "load a value from memory";
  let description = [{
    The "load" operation reads a value from a data block.
  }];

  let arguments = (ins AnyMemRef);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value memref", [{
      auto elementType = memref.getType().cast<MemRefType>().getElementType();
      build(b, state, elementType, memref);
    }]>
  ];
}

def StoreOp : SCL_Op<"store"> {
  let summary = "store a value to memory";
  let description = [{
    The "store" operation writes a value to a data block.
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyType:$rhs);
}

//
// MAKR: control flow
//

def TerminatorOp : SCL_Op<"terminator", [Terminator]> {}

def IfThenElseOp : SCL_Op<"if">, SingleBlockImplicitTerminator<"terminator"> {
  let summary = "if-then-else operation";
  let description = [{
    Example:
      scl.if (%cond) {
        ...
      } else {
        ...
      }
  }];

  let arguments = (ins I1:$cond);
  let regions = (region AnyRegion:$thenBody, AnyRegion:$elseBody);
}

//
// MARK: terminator operations
//

def ContinueOp : SCL_Op<"continue", [Terminator]> {
  let summary = "continue operation";
  let description = [{
    A CONTINUE statement is used to terminate the execution of the current iteration of a loop (initiated by a FOR, WHILE or REPEAT statement) and to restart processing within the loop.
  }];
}


def ReturnOp : FuncTerminatorOp<"return"> {
  let summary = "return operation";
  let description = [{
    A RETURN statement causes the program to exit the block (OB, FB or FC) currently being processed and to return to the calling block or the operating system if the block being exited is an OB.
  }];
}


def ExitOp : FuncTerminatorOp<"exit"> {
  let summary = "exit operation";
  let description = [{
    An EXIT statement is used to exit a loop (FOR, WHILE or REPEAT loop) at any point regardless of whether the break condition is satisfied.

  }];
}

//
// MARK: arithmetic operations
//

def AddOp : ArithmeticOp<"add", [Commutative]> {
  let summary = "addition operation";
  let description = [{
    The "add" operation performs addition between two values.
  }];
}

def SubOp : ArithmeticOp<"sub"> {
  let summary = "subtract operation";
  let description = [{
    The "sub" operation performs substraction of two values.
  }];
}

def MulOp : ArithmeticOp<"mul", [Commutative]> {
  let summary = "multiplication operation";
  let description = [{
    The "mul" operation performs multiplication between two values.
  }];
}

def DivOp : ArithmeticOp<"div"> {
  let summary = "division operation";
  let description = [{
    The "div" operation performs division of two values.
  }];
}

def ModOp : ArithmeticOp<"mod"> {
  let summary = "modulo of division operation";
  let description = [{
    The "mod" operation performs division of two values and returns the rest.
  }];
}

def ExpOp : ArithmeticOp<"exp", [Commutative]> {
  let summary = "exponentiation operation";
  let description = [{
    The "exp" operation performs exponentiation of two values.
  }];
}

def AndOp : ArithmeticOp<"and"> {
}

def OrOp : ArithmeticOp<"or"> {
}

def XOrOp : ArithmeticOp<"xor", [Commutative]> {
}

//
// MARK: comparition operations
//

def EqualOp : CompareOp<"cmpeq"> {
}

def NotEqualOp : CompareOp<"cmpne"> {
}

def LessThanOp : CompareOp<"cmplt"> {
}

def LessEqualOp : CompareOp<"cmple"> {
}

def GreaterThanOp : CompareOp<"cmpgt"> {
}

def GreaterEqualOp : CompareOp<"cmpge"> {
}

//
// MARK: unary operations
//

def UnaryNotOp : UnaryArithmeticOp<"not"> {
  let summary = "negate a logical value";
  let description = [{
    TBD
  }];
}

def UnaryMinusOp : UnaryArithmeticOp<"negate"> {
  let summary = "toggle the sign of a number";
  let description = [{
    TBD
  }];
}
#endif // SCL_OPS
