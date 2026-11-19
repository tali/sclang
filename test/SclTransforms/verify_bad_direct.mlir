// RUN: not sclang-gen -x=mlir --emit=mlir %s 2>&1 | FileCheck %s
// Negative verifier test: a direct address with a negative byte offset must
// be rejected at parse time (type parser validation).
// CHECK: direct db/byte must be >= 0 and bit in 0..7
module {
  scl.function @f() {
    %d = scl.getdirect M, 0, -5, 0, 16 : !scl.address<!scl.direct<M, 0, -5, 0, 16>>
    scl.end
  }
}
