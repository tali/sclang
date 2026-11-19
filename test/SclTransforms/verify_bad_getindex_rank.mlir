// RUN: not sclang-gen -x=mlir --emit=mlir %s 2>&1 | FileCheck %s
// Negative verifier test: GetIndexOp index count must match the array rank.
// CHECK: 'scl.getindex' op index count must match the array rank (2 vs 1)
module {
  scl.function @f(%a: !scl.address<!scl.array<!scl.int, 0:9>>) {
    %i = scl.constant 0 : i16 -> !scl.int
    %e = scl.getindex %i, %i : !scl.int, !scl.int from %a : !scl.address<!scl.array<!scl.int, 0:9>> -> !scl.address<!scl.int>
    scl.end
  }
}
