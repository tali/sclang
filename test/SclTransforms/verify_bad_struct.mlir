// RUN: not sclang-gen -x=mlir --emit=mlir %s 2>&1 | FileCheck %s
// Negative verifier test: duplicate struct member names.
// CHECK: duplicate struct member 'x'
module {
  scl.function @f() {
    %a = scl.tmpvar "a" : !scl.address<!scl.struct<x: !scl.int, x: !scl.int>>
    scl.end
  }
}
