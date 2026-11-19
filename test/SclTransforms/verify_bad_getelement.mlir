// RUN: not sclang-gen -x=mlir --emit=mlir %s 2>&1 | FileCheck %s
// Negative verifier test: GetElementOp member must exist in the struct.
// CHECK: 'scl.getelement' op 'nope' is not a member of
module {
  scl.function @f(%a: !scl.address<!scl.struct<a1: !scl.int>>) {
    %e = scl.getelement "nope" from %a : !scl.address<!scl.struct<a1: !scl.int>> -> !scl.address<!scl.char>
    scl.end
  }
}
