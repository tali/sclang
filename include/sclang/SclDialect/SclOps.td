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

include "sclang/SclDialect/SclBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"


// MARK: data block

def DataBlockOp : SCL_Op<"db", [Symbol]> {
  let summary = "Define a data block";

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name
  );

  let results = (outs);

  let assemblyFormat = "attr-dict $sym_name `:` $type";
}

//
// MARK: functions / function blocks
//

def CallFbOp : SCL_Op<"call.fb", [
    //TBD, InFunctionScope,
    DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call a function block";

  let arguments = (ins
    FlatSymbolRefAttr:$fb,
    Scl_AddressType:$idb
  );

  let results = (outs);

  let assemblyFormat = [{
    $fb `(` $idb `:` type($idb) `)` attr-dict
  }];

  let hasCanonicalizer = 1;
}

def CallFcOp : SCL_Op<"call.fc", [
    //TBD, InFunctionScope,
    DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call a function";

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Scl_Type>:$arguments,
    StrArrayAttr:$argNames
  );

  let results = (outs
    Scl_Type:$result
  );

  let assemblyFormat = [{
    $callee `(` $arguments `)` attr-dict `:`
    functional-type($arguments, results)
  }];

  let hasCanonicalizer = 1;
}

def FunctionOp : TopLevelOp<"function", [
    AutomaticAllocationScope, DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionLike, IsolatedFromAbove, Symbol
  ]> {
  let summary = "Declare or define a function";

  let description = [{
    This op declares or defines an SCL function using one region, which
    contains one or more blocks.

    This op itself takes no operands and generates no results. Its region
    can take zero or more arguments and return zero or one values.
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let parser = [{ return ::parseFunctionOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return success(); }];

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
  ];

  let extraClassDeclaration = [{
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FunctionOp>;

    /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }

    /// Returns the number of results. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called safely
    LogicalResult verifyType();

    /// Hook for OpTrait::FunctionLike, called after verifying the function
    /// type and the presence of the (potentially empty) function body.
    LogicalResult verifyBody();
  }];
}

def FunctionBlockOp : TopLevelOp<"function_block", [
    AutomaticAllocationScope, DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionLike, IsolatedFromAbove, Symbol
  ]> {
  let summary = "Declare or define a function block";

  let description = [{
    This op declares or defines an SCL function block using one region, which
    contains one or more blocks.

    This op itself takes no operands and generates no results. Its region
    takes one arguments with the address to its IDB and returns zero values.
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let parser = [{ return ::parseFunctionBlockOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return success(); }];

  let builders = [
    OpBuilder<(ins "StringRef":$name)>
  ];

  let extraClassDeclaration = [{
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FunctionBlockOp>;

    /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }

    /// Returns the number of results. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called safely
    LogicalResult verifyType();

    /// Hook for OpTrait::FunctionLike, called after verifying the function
    /// type and the presence of the (potentially empty) function body.
    LogicalResult verifyBody();
  }];
}


//
// MARK: variable handling
//

def VariableOp : SCL_Op<"var", [Symbol]> {
  let summary = "declare a static/input/output variable in a FB";

  let arguments = (ins
    TypeAttr:$type,
    OptionalAttr<UnitAttr>:$input,
    OptionalAttr<UnitAttr>:$output,
    StrAttr:$sym_name
  );
  let results = (outs);

  let assemblyFormat = "(`input` $input^)? (`output` $output^)? $sym_name attr-dict `:` $type";

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$isInput, "bool":$isOutput, "StringRef":$name)>
  ];
}

def TempVariableOp : SCL_Op<"tmpvar",
    [DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "allocate stack memory for a temporary variable";
  let description = [{ TBD }];
  let arguments = (ins StrAttr:$name);
  let results = (outs Scl_AddressType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def BitCastOp : SCL_Op<"bit.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between different representations of the same bits";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def DialectCastOp : SCL_Op<"mlir.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between SCL and Standard dialect";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def IntegerCastOp : SCL_Op<"int.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between integers with different bitwidths";
  let arguments = (ins Scl_IntegerType:$value);
  let results = (outs Scl_IntegerType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def ConstantOp : SCL_Op<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "literal constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `->` type($result)";
  let hasFolder = 1;
}

def GetGlobalOp : SCL_Op<"getglobal",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $name `:` type($element)";
}

def GetElementOp : SCL_Op<"getelement",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "get the address of a struct element";
  let arguments = (ins Scl_AddressType:$addr, StrAttr:$name);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $name `from` $addr `:` type($addr) `->` type($element)";

  let builders = [
    OpBuilder<(ins "Value":$address, "StringRef":$name), [{
      auto structType = address.getType().cast<AddressType>().getElementType();
      auto elementType = structType.cast<StructType>().getElementType(name);
      auto resultType = AddressType::get(elementType);
      build($_builder, $_state, resultType, address, name);
    }]>
  ];
}

def LoadOp : SCL_Op<"load", [/*SameOperandsAndResultElementType*/]> {
  let summary = "load a value from memory";
  let description = [{
    The "load" operation reads a value from a data block.
  }];

  let arguments = (ins Scl_AddressType:$address);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$address `:` type($address) attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "Value":$address), [{
      auto elementType = address.getType().cast<AddressType>().getElementType();
      build($_builder, $_state, elementType, address);
    }]>
  ];
}

def StoreOp : SCL_Op<"store", [/*SameOperandsElementType]*/]> {
  let summary = "store a value to memory";
  let description = [{
    The "store" operation writes a value to a data block.
  }];

  let arguments = (ins Scl_AddressType:$lhs, AnyType:$rhs);
  let assemblyFormat = "attr-dict $lhs `:` type($lhs) `=` $rhs `:` type($rhs)";
}


//
// MARK: control flow
//

def EndOp : SCL_Op<"end", [NoSideEffect, Terminator]> {
  let assemblyFormat = "attr-dict";
}

def IfThenElseOp : SCL_Op<"if">, SingleBlockImplicitTerminator<"terminator"> {
  let summary = "if-then-else operation";
  let description = [{
    Example:
      scl.if %cond {
        ...
      } else {
        ...
      }
  }];

  let arguments = (ins Scl_Bool:$cond);
  let regions = (region SizedRegion<1>:$thenBody, AnyRegion:$elseBody);
  let assemblyFormat = "attr-dict $cond `:` type($cond) $thenBody `else` $elseBody";
}

//
// MARK: terminator operations
//

def ContinueOp : SCL_Op<"continue", [Terminator]> {
  let summary = "continue operation";
  let description = [{
    A CONTINUE statement is used to terminate the execution of the current
    iteration of a loop (initiated by a FOR, WHILE or REPEAT statement)
    and to restart processing within the loop.
  }];
  let assemblyFormat = "attr-dict";
}

def ReturnOp : FuncTerminatorOp<"return"> {
  let summary = "return from a function/function_block";
  let description = [{
    A RETURN statement causes the program to exit the block (OB, FB or FC)
    currently being processed and to return to the calling block or the
    operating system if the block being exited is an OB.
  }];
  let assemblyFormat = "attr-dict";
}

def ReturnValueOp : FuncTerminatorOp<"return.value"> {
  let summary = "return a value from a function";
  let description = [{
    The function is exited and the value from the given Address is
    loaded to obtain the function return value.
  }];
  let arguments = (ins Scl_AddressType:$value);
  let assemblyFormat = "attr-dict operands `:` type(operands)";

  let extraClassDeclaration = [{
    Type getReturnType() {
      return value().getType().dyn_cast<AddressType>().getElementType();
    }
  }];
}

def ExitOp : SCL_Op<"exit", [Terminator]> {
  let summary = "exit operation";
  let description = [{
    An EXIT statement is used to exit a loop (FOR, WHILE or REPEAT loop) at
    any point regardless of whether the break condition is satisfied.
  }];
  let assemblyFormat = "attr-dict";
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

def ExpOp : ArithmeticOp<"exp"> {
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
