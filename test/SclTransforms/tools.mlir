// RUN: sclang-translate --serialize %s -o %t.mlirbc && sclang-translate %t.mlirbc > %t.rt.mlir && sclang-translate %s > %t.orig.mlir && diff %t.rt.mlir %t.orig.mlir
// RUN: sclang-opt --pass-pipeline='builtin.module(scl-lower-to-std,scl-lower-to-llvm)' %s | sclang-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LL
// RUN: echo 'match hasOpName("scl.function")' | sclang-query %s | FileCheck %s --check-prefix=Q

// Exercises the MLIR-level tool variants on scl-dialect IR:
// sclang-translate (bytecode round-trip + mlir-to-llvmir), sclang-query.

scl.function @add_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.add %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// LL: define {{.*}}@add_i8
// Q: Match #1
