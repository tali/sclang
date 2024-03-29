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

def DataBlockOp : TopLevelOp<"db", [Symbol]> {
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

def CallFbOp : SclOp<"call.fb", [
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

def CallFcOp : SclOp<"call.fc", [
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
  let printer = [{ return ::print(p, *this); }];
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
    FunctionLike, IsolatedFromAbove, Symbol, SymbolTable
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
  let printer = [{ return ::print(p, *this); }];
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

def VariableOp : SclOp<"var", [Symbol, HasParent<"FunctionBlockOp">]> {
  let summary = "declare a static/input/output variable in a FB";

  let arguments = (ins
    TypeAttr:$type,
    OptionalAttr<UnitAttr>:$input,
    OptionalAttr<UnitAttr>:$output,
    StrAttr:$sym_name,
    Optional<AnyType>:$init
  );
  let results = (outs);

  let assemblyFormat = "(`input` $input^)? (`output` $output^)? $sym_name attr-dict `:` $type (`=` $init^ `:` type($init))?";

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$isInput, "bool":$isOutput, "StringRef":$name)>,
    OpBuilder<(ins "Type":$type, "bool":$isInput, "bool":$isOutput, "StringRef":$name, "Value":$init)>
  ];
}

def TempVariableOp : SclOp<"tmpvar",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "allocate stack memory for a temporary variable";
  let description = [{ TBD }];
  let arguments = (ins StrAttr:$name);
  let results = (outs Scl_AddressType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def BitCastOp : SclOp<"bit.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between different representations of the same bits";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def DialectCastOp : SclOp<"mlir.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between SCL and Standard dialect";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def IntegerCastOp : SclOp<"int.cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let summary = "Type cast between integers with different bitwidths";
  let arguments = (ins Scl_IntegerType:$value);
  let results = (outs Scl_IntegerType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def ConstantOp : SclOp<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `->` type($result)";
  let hasFolder = 1;
}

def ConstantS5TimeOp : SclOp<"constant.s5time",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal time or date constant";
  let arguments = (ins UI16Attr:$value);
  let results = (outs Scl_S5TimeType:$result);
  let parser = [{ return ::parseConstantS5TimeOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let builders = [ OpBuilder<(ins "unsigned int":$timeMS)> ];
  let extraClassDeclaration = [{
    unsigned int getTimeMS();
  }];
  let hasFolder = 1;
}

def ConstantTimeOp : SclOp<"constant.time",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal time constant";
  let arguments = (ins SI32Attr:$value);
  let results = (outs Scl_TimeType:$result);
  let parser = [{ return ::parseConstantTimeOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let builders = [ OpBuilder<(ins "int":$timeMS)> ];
  let hasFolder = 1;
}

def ConstantDateOp : SclOp<"constant.date",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal time constant";
  let arguments = (ins UI16Attr:$value);
  let results = (outs Scl_DateType:$result);
  let parser = [{ return ::parseConstantDateOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let builders = [ OpBuilder<(ins "int":$year, "int":$month, "int":$day)> ];
  let hasFolder = 1;
}

def ConstantTimeOfDayOp : SclOp<"constant.tod",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal time constant";
  let arguments = (ins UI32Attr:$value);
  let results = (outs Scl_TimeOfDayType:$result);
  let parser = [{ return ::parseConstantTimeOfDayOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let builders = [ OpBuilder<(ins "uint32_t":$timeMS)> ];
  let hasFolder = 1;
}

def ConstantDateAndTimeOp : SclOp<"constant.dt",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "literal time constant";
  let arguments = (ins UI64Attr:$value);
  let results = (outs Scl_DateAndTimeType:$result);
  let parser = [{ return ::parseConstantDateAndTimeOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let builders = [
    OpBuilder<(ins
      "int":$year, "int":$month, "int":$day,
      "int":$hour, "int":$min, "int":$sec, "int":$msec
    )>
  ];
  let hasFolder = 1;
}

def GetElementOp : SclOp<"getelement",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "get the address of a struct element";
  let arguments = (ins Scl_AddressType:$addr, StrAttr:$name);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $name `from` $addr `:` type($addr) `->` type($element)";
}

def GetGlobalOp : SclOp<"getglobal",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $name `:` type($element)";
}

def GetIndexOp : SclOp<"getindex", [NoSideEffect]> {
  let summary = "get the address of an indexed element";
  let arguments = (ins Scl_AddressType:$addr, Variadic<Scl_IntegerType>:$indices);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $indices `:` type($indices) `from` $addr `:` type($addr) `->` type($element)";
}

def GetVariableOp : SclOp<"getvariable",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "get the address of a FB variable";
  let arguments = (ins Scl_AddressType:$addr, FlatSymbolRefAttr:$sym_name);
  let results = (outs Scl_AddressType:$element);
  let assemblyFormat = "attr-dict $sym_name `from` $addr `:` type($addr) `->` type($element)";
}

def LoadOp : SclOp<"load", [/*SameOperandsAndResultElementType*/]> {
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

def StoreOp : SclOp<"store", [/*SameOperandsElementType]*/]> {
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

def ConditionOp : SclOp<"condition",
    [NoSideEffect, HasParent<"WhileDoOp">, Terminator]> {
  let summary = "condition for WhileDoOp";
  let arguments = (ins Scl_Bool:$cond);
  let assemblyFormat = "attr-dict $cond `:` type($cond)";
}

def EndOp : SclOp<"end", [NoSideEffect, Terminator]> {
  let summary = "implicit terminator for blocks";
  let assemblyFormat = "attr-dict";
}

def ForDoOp : SclOp<"for"> {
  let summary = "for-do loop";
  let arguments = (
    ins Scl_AddressType:$var,
    Scl_NumericType:$init,
    Scl_NumericType:$to,
    Optional<Scl_NumericType>:$by
  );
  let regions = (region AnyRegion:$doBody);
  let assemblyFormat = "attr-dict $var `:` type($var) `=` $init `:` type($init) `to` $to `:` type($to) (`by` $by^ `:` type($by))? `do` $doBody";
}

def IfThenElseOp : SclOp<"if", [SingleBlockImplicitTerminator<"scl::EndOp">]> {
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

def WhileDoOp : SclOp<"while"> {
  let summary = "while-do loop";
  let regions = (region AnyRegion:$whileBody, AnyRegion:$doBody);
  let assemblyFormat = "$whileBody attr-dict `do` $doBody";
}

def RepeatOp : SclOp<"repeat"> {
  let summary = "repeat-until loop";
  let regions = (region AnyRegion:$repeatBody);
  let assemblyFormat = "$repeatBody attr-dict";
}

def UntilOp : SclOp<"until", [NoSideEffect, HasParent<"RepeatOp">, Terminator]> {
  let summary = "condition for RepeatOp";
  let arguments = (ins Scl_Bool:$cond);
  let assemblyFormat = "attr-dict $cond `:` type($cond)";
}


//
// MARK: terminator operations
//

def ContinueOp : SclOp<"continue", [Terminator]> {
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

def ExitOp : SclOp<"exit", [Terminator]> {
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

//
// Debug helpers
//

def DebugPrintOp : SclOp<"debug.print"> {
  let summary = "debug print";
  let arguments = (ins StrAttr:$msg);
  let assemblyFormat = "attr-dict $msg";
}

#endif // SCL_OPS
