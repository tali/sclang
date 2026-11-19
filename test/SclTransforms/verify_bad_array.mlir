// RUN: not sclang-gen -x=mlir --emit=mlir %s 2>&1 | FileCheck %s
// Negative verifier test: array dimension with lower > upper.
// CHECK: array dimension lower > upper
module {
  scl.function @f() {
    %a = scl.tmpvar "a" : !scl.address<!scl.array<!scl.int, 5:0>>
    scl.end
  }
}
