//===- AwlOps.td - AWL dialect operation definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the AWL (STL) dialect. AWL is an accumulator
// machine: `awl.load` pushes ACC1, arithmetic pops two and pushes one, RLO
// is an explicit i1 SSA value. Jumps use string labels resolved to cf blocks
// by the lowering (awl.label markers + awl.jump* ops in a flat block).
//
//===----------------------------------------------------------------------===//

#ifndef AWL_OPS
#define AWL_OPS

include "sclang/AwlDialect/AwlBase.td"
include "mlir/Interfaces/CallInterfaces.td"

// MARK: blocks

// An AWL block (OB n / FB n / FC n) with a flat instruction list; the single
// block ends with awl.be.
def BlockOp : AwlTopLevelOp<"block", [Symbol]> {
  let summary = "An AWL block (OB/FB/FC) with a flat instruction list";
  let description = [{
    The region holds one flat block of awl ops ending in `awl.be`;
    `awl.label` markers and `awl.jump*` ops reference labels by name and
    are turned into cf blocks by the lowering.

    Example:
      awl.block @OB1 {
        awl.label "loop"
        ...
        awl.be
      }
  }];
  let arguments = (ins StrAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

// MARK: operands and constants

// A memory-area / DB operand, e.g. M 10.3 or DB1.DBW 4 -> !scl.direct<...>
def DirectOp : AwlOp<"direct", [Pure, ConstantLike]> {
  let summary = "a directly-represented operand (memory area / DB)";
  let description = [{
    Same addressing as scl.getdirect but yields the !scl.direct value itself
    (not an address); shared by both frontends (ADR 003). Prints like
    `awl.direct M, 0, 10, 3, 1 : !scl.direct<M, 0, 10, 3, 1>`.
  }];
  let arguments = (ins
    StrAttr:$area,
    I32Attr:$db,
    I32Attr:$byte,
    I32Attr:$bit,
    I32Attr:$width
  );
  let results = (outs Awl_Direct:$result);
  let extraClassDeclaration = [{
    static void build(OpBuilder &builder, OperationState &state,
                      mlir::scl::DirectArea area, int32_t db, int32_t byte,
                      int32_t bit, int32_t width);
  }];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ConstOp : AwlOp<"const", [Pure, ConstantLike]> {
  let summary = "a literal constant (L 5, L 16#FF, L 5.0)";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `->` type($result)";
}

// MARK: load / store (L / T)

def LoadOp : AwlOp<"load"> {
  let summary = "L: load ACC1 from an operand";
  let arguments = (ins Awl_Direct:$addr);
  let results = (outs AnyType:$result);
  let extraClassDeclaration = [{
    // Result type from the operand's width: i1 for bit access, else an
    // integer of the access width.
    static void build(OpBuilder &builder, OperationState &state, Value addr);
  }];
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
}

def StoreOp : AwlOp<"store"> {
  let summary = "T: store ACC1 to an operand";
  let arguments = (ins AnyType:$value, Awl_Direct:$addr);
  let assemblyFormat = "attr-dict $value `,` $addr `:` type($value) `,` type($addr)";
}

// MARK: RLO logic (U/UN/O/ON/NOT, A(/O(), =, S, R)

// U/UN: rlo = rlo AND bit (negate set for UN)
def RloAndOp : AwlOp<"rlo_and"> {
  let arguments = (ins Awl_Bool:$rlo, Awl_Direct:$addr,
                       OptionalAttr<UnitAttr>:$negate);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $rlo `,` $addr (`negate` $negate^)? `:` type($rlo) `,` type($addr) `->` type($result)";
}

// O/ON: rlo = rlo OR bit (negate set for ON)
def RloOrOp : AwlOp<"rlo_or"> {
  let arguments = (ins Awl_Bool:$rlo, Awl_Direct:$addr,
                       OptionalAttr<UnitAttr>:$negate);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $rlo `,` $addr (`negate` $negate^)? `:` type($rlo) `,` type($addr) `->` type($result)";
}

def NotOp : AwlOp<"not"> {
  let summary = "NOT: invert RLO";
  let arguments = (ins Awl_Bool:$rlo);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $rlo `:` type($rlo) `->` type($result)";
}

// = bit: assign RLO to a bit
def AssignOp : AwlOp<"assign"> {
  let arguments = (ins Awl_Bool:$rlo, Awl_Direct:$addr);
  let assemblyFormat = "attr-dict $rlo `,` $addr `:` type($rlo) `,` type($addr)";
}

// S bit: set the bit, RLO becomes 1
def SetOp : AwlOp<"set"> {
  let arguments = (ins Awl_Direct:$addr);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
}

// R bit: reset the bit, RLO becomes 0
def ResetOp : AwlOp<"reset"> {
  let arguments = (ins Awl_Direct:$addr);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
}

// MARK: arithmetic (+I -I *I /I +D ... +R ...), word logic (AW OW XOW), cast

class AwlArithOp<string mnemonic> : AwlOp<mnemonic> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result)";
}
def AddOp : AwlArithOp<"add">;
def SubOp : AwlArithOp<"sub">;
def MulOp : AwlArithOp<"mul">;
def DivOp : AwlArithOp<"div">;
def AndOp : AwlArithOp<"and">;
def OrOp : AwlArithOp<"or">;
def XOrOp : AwlArithOp<"xor">;

def NegOp : AwlOp<"neg"> {
  let arguments = (ins AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $rhs `:` type($rhs) `->` type($result)";
}

def CastOp : AwlOp<"cast"> {
  let summary = "cast between accumulator widths/types";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

// MARK: comparisons (==I <>I >I <I >=I <=I, ==D ..., ==R ...): set RLO
class AwlCmpOp<string mnemonic> : AwlOp<mnemonic> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs Awl_Bool:$result);
  let assemblyFormat = "attr-dict $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result)";
}
def CmpEqOp : AwlCmpOp<"cmpeq">;
def CmpNeOp : AwlCmpOp<"cmpne">;
def CmpLtOp : AwlCmpOp<"cmplt">;
def CmpLeOp : AwlCmpOp<"cmple">;
def CmpGtOp : AwlCmpOp<"cmpgt">;
def CmpGeOp : AwlCmpOp<"cmpge">;

// MARK: jumps, labels, calls, misc

def LabelOp : AwlOp<"label", [Pure]> {
  let summary = "a jump label marker";
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict $name";
}

def JumpOp : AwlOp<"jump"> {
  let summary = "SPA: unconditional jump to a label";
  let arguments = (ins StrAttr:$label);
  let assemblyFormat = "attr-dict $label";
}

def JumpIfOp : AwlOp<"jump_if"> {
  let summary = "SPB: jump if RLO = 1";
  let arguments = (ins Awl_Bool:$rlo, StrAttr:$label);
  let assemblyFormat = "attr-dict $rlo `,` $label `:` type($rlo)";
}

def JumpIfNotOp : AwlOp<"jump_if_not"> {
  let summary = "SPN: jump if RLO = 0";
  let arguments = (ins Awl_Bool:$rlo, StrAttr:$label);
  let assemblyFormat = "attr-dict $rlo `,` $label `:` type($rlo)";
}

def CallOp : AwlOp<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "CALL/UC: call a block (FC/OB)";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$arguments);
  let results = (outs);
  let assemblyFormat = "attr-dict $callee `(` $arguments `)` `:` functional-type($arguments, results)";
}

// MARK: timers (SP/SE/SS/SV/SI T n) and counters (ZV/ZR/S/R C n)

// Timer start: pops the S5TIME-encoded duration from ACC1, starts the timer.
// `mode` is one of SclRtTimerMode (SP=0, SE=1, SS=2, SV=3, SI=4).
def TimerStartOp : AwlOp<"timer_start"> {
  let summary = "SP/SE/SS/SV/SI: start timer with ACC1 = S5TIME duration";
  let arguments = (ins I32Attr:$mode, AnyType:$dur, Awl_Direct:$timer);
  let assemblyFormat = "attr-dict $mode `,` $dur `:` type($dur) `,` $timer `:` type($timer)";
}

// Counter operation. `op` is one of SclRtCounterOp (ZV=0, ZR=1, S=2, R=3);
// `value` is the new count for S (from ACC1), 0 otherwise.
def CounterOp : AwlOp<"counter"> {
  let summary = "ZV/ZR/S/R: counter operation on C n";
  let arguments = (ins I32Attr:$op, AnyType:$value, Awl_Direct:$counter);
  let assemblyFormat = "attr-dict $op `,` $value `:` type($value) `,` $counter `:` type($counter)";
}

def NopOp : AwlOp<"nop"> {
  let summary = "NOP 0/1";
  let assemblyFormat = "attr-dict";
}

def BeOp : AwlOp<"be", [Terminator]> {
  let summary = "BE: block end";
  let assemblyFormat = "attr-dict";
}

#endif // AWL_OPS
