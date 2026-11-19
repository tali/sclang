// RUN: sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std)' %s | FileCheck %s --check-prefix=STD
// RUN: sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std,scl-lower-to-llvm)' %s | FileCheck %s --check-prefix=LLVM

// The sclang lowering passes are TableGen-registered (Passes.td, see
// include/sclang/SclTransforms/Passes.h) and addressable by name in
// --pass-pipeline strings through sclang-opt — this exercises exactly that.

scl.function @add_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.add %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// STD-LABEL: func.func @add_i8
// STD:         arith.addi %arg0, %arg1 : i8
// STD-NOT:     scl.add

// LLVM-LABEL: llvm.func @add_i8
// LLVM:         llvm.add %arg0, %arg1 : i8
// LLVM-NOT:     arith.addi
