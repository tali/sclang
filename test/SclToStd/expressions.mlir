// RUN: sclang-gen --emit=mlir-std -x=mlir %s | FileCheck %s

// MARK: scl.add

// CHECK-LABEL: @add_i8
// CHECK:       addi %arg0, %arg1 : i8
func @add_i8(%arg0: i8, %arg1: i8) -> i8 {
  %res = scl.add %arg0, %arg1 : i8
  return %res : i8
}

// CHECK-LABEL: @add_i16
// CHECK:       addi %arg0, %arg1 : i16
func @add_i16(%arg0: i16, %arg1: i16) -> i16 {
  %res = scl.add %arg0, %arg1 : i16
  return %res : i16
}

// CHECK-LABEL: @add_i32
// CHECK:       addi %arg0, %arg1 : i32
func @add_i32(%arg0: i32, %arg1: i32) -> i32 {
  %res = scl.add %arg0, %arg1 : i32
  return %res : i32
}

// CHECK-LABEL: @add_f32
// CHECK:       addf %arg0, %arg1 : f32
func @add_f32(%arg0: f32, %arg1: f32) -> f32 {
  %res = scl.add %arg0, %arg1 : f32
  return %res : f32
}


// MARK: scl.sub

// CHECK-LABEL: @sub_i8
// CHECK:       subi %arg0, %arg1 : i8
func @sub_i8(%arg0: i8, %arg1: i8) -> i8 {
  %res = scl.sub %arg0, %arg1 : i8
  return %res : i8
}

// CHECK-LABEL: @sub_i16
// CHECK:       subi %arg0, %arg1 : i16
func @sub_i16(%arg0: i16, %arg1: i16) -> i16 {
  %res = scl.sub %arg0, %arg1 : i16
  return %res : i16
}

// CHECK-LABEL: @sub_i32
// CHECK:       subi %arg0, %arg1 : i32
func @sub_i32(%arg0: i32, %arg1: i32) -> i32 {
  %res = scl.sub %arg0, %arg1 : i32
  return %res : i32
}

// CHECK-LABEL: @sub_f32
// CHECK:       subf %arg0, %arg1 : f32
func @sub_f32(%arg0: f32, %arg1: f32) -> f32 {
  %res = scl.sub %arg0, %arg1 : f32
  return %res : f32
}


// MARK: scl.mul

// CHECK-LABEL: @mul_i8
// CHECK:       muli %arg0, %arg1 : i8
func @mul_i8(%arg0: i8, %arg1: i8) -> i8 {
  %res = scl.mul %arg0, %arg1 : i8
  return %res : i8
}

// CHECK-LABEL: @mul_i16
// CHECK:       muli %arg0, %arg1 : i16
func @mul_i16(%arg0: i16, %arg1: i16) -> i16 {
  %res = scl.mul %arg0, %arg1 : i16
  return %res : i16
}

// CHECK-LABEL: @mul_i32
// CHECK:       muli %arg0, %arg1 : i32
func @mul_i32(%arg0: i32, %arg1: i32) -> i32 {
  %res = scl.mul %arg0, %arg1 : i32
  return %res : i32
}

// CHECK-LABEL: @mul_f32
// CHECK:       mulf %arg0, %arg1 : f32
func @mul_f32(%arg0: f32, %arg1: f32) -> f32 {
  %res = scl.mul %arg0, %arg1 : f32
  return %res : f32
}


// MARK: scl.div

// CHECK-LABEL: @div_i8
// CHECK:       divi_signed %arg0, %arg1 : i8
func @div_i8(%arg0: i8, %arg1: i8) -> i8 {
  %res = scl.div %arg0, %arg1 : i8
  return %res : i8
}

// CHECK-LABEL: @div_i16
// CHECK:       divi_signed %arg0, %arg1 : i16
func @div_i16(%arg0: i16, %arg1: i16) -> i16 {
  %res = scl.div %arg0, %arg1 : i16
  return %res : i16
}

// CHECK-LABEL: @div_i32
// CHECK:       divi_signed %arg0, %arg1 : i32
func @div_i32(%arg0: i32, %arg1: i32) -> i32 {
  %res = scl.div %arg0, %arg1 : i32
  return %res : i32
}

// CHECK-LABEL: @div_f32
// CHECK:       divf %arg0, %arg1 : f32
func @div_f32(%arg0: f32, %arg1: f32) -> f32 {
  %res = scl.div %arg0, %arg1 : f32
  return %res : f32
}


// MARK: scl.mod

// CHECK-LABEL: @mod_i8
// CHECK:       remi_signed %arg0, %arg1 : i8
func @mod_i8(%arg0: i8, %arg1: i8) -> i8 {
  %res = scl.mod %arg0, %arg1 : i8
  return %res : i8
}

// CHECK-LABEL: @mod_i16
// CHECK:       remi_signed %arg0, %arg1 : i16
func @mod_i16(%arg0: i16, %arg1: i16) -> i16 {
  %res = scl.mod %arg0, %arg1 : i16
  return %res : i16
}

// CHECK-LABEL: @mod_i32
// CHECK:       remi_signed %arg0, %arg1 : i32
func @mod_i32(%arg0: i32, %arg1: i32) -> i32 {
  %res = scl.mod %arg0, %arg1 : i32
  return %res : i32
}

// CHECK-LABEL: @mod_f32
// CHECK:       remf %arg0, %arg1 : f32
func @mod_f32(%arg0: f32, %arg1: f32) -> f32 {
  %res = scl.mod %arg0, %arg1 : f32
  return %res : f32
}


// MARK: scl.cmpeq

// CHECK-LABEL: @cmpeq_i8
// CHECK:       cmpi "eq", %arg0, %arg1 : i8
func @cmpeq_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmpeq %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmpeq_i16
// CHECK:       cmpi "eq", %arg0, %arg1 : i16
func @cmpeq_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmpeq %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmpeq_i32
// CHECK:       cmpi "eq", %arg0, %arg1 : i32
func @cmpeq_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmpeq %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmpeq_f32
// CHECK:       cmpf "ueq", %arg0, %arg1 : f32
func @cmpeq_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmpeq %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.cmpne

// CHECK-LABEL: @cmpne_i8
// CHECK:       cmpi "ne", %arg0, %arg1 : i8
func @cmpne_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmpne %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmpne_i16
// CHECK:       cmpi "ne", %arg0, %arg1 : i16
func @cmpne_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmpne %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmpne_i32
// CHECK:       cmpi "ne", %arg0, %arg1 : i32
func @cmpne_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmpne %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmpne_f32
// CHECK:       cmpf "une", %arg0, %arg1 : f32
func @cmpne_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmpne %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.cmpge

// CHECK-LABEL: @cmpge_i8
// CHECK:       cmpi "sge", %arg0, %arg1 : i8
func @cmpge_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmpge %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmpge_i16
// CHECK:       cmpi "sge", %arg0, %arg1 : i16
func @cmpge_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmpge %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmpge_i32
// CHECK:       cmpi "sge", %arg0, %arg1 : i32
func @cmpge_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmpge %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmpge_f32
// CHECK:       cmpf "uge", %arg0, %arg1 : f32
func @cmpge_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmpge %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.cmpgt

// CHECK-LABEL: @cmpgt_i8
// CHECK:       cmpi "sgt", %arg0, %arg1 : i8
func @cmpgt_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmpgt %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmpgt_i16
// CHECK:       cmpi "sgt", %arg0, %arg1 : i16
func @cmpgt_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmpgt %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmpgt_i32
// CHECK:       cmpi "sgt", %arg0, %arg1 : i32
func @cmpgt_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmpgt %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmpgt_f32
// CHECK:       cmpf "ugt", %arg0, %arg1 : f32
func @cmpgt_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmpgt %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.cmple

// CHECK-LABEL: @cmple_i8
// CHECK:       cmpi "sle", %arg0, %arg1 : i8
func @cmple_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmple %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmple_i16
// CHECK:       cmpi "sle", %arg0, %arg1 : i16
func @cmple_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmple %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmple_i32
// CHECK:       cmpi "sle", %arg0, %arg1 : i32
func @cmple_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmple %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmple_f32
// CHECK:       cmpf "ule", %arg0, %arg1 : f32
func @cmple_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmple %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.cmplt

// CHECK-LABEL: @cmplt_i8
// CHECK:       cmpi "slt", %arg0, %arg1 : i8
func @cmplt_i8(%arg0: i8, %arg1: i8) -> i1 {
  %res = scl.cmplt %arg0, %arg1 : i8
  return %res : i1
}

// CHECK-LABEL: @cmplt_i16
// CHECK:       cmpi "slt", %arg0, %arg1 : i16
func @cmplt_i16(%arg0: i16, %arg1: i16) -> i1 {
  %res = scl.cmplt %arg0, %arg1 : i16
  return %res : i1
}

// CHECK-LABEL: @cmplt_i32
// CHECK:       cmpi "slt", %arg0, %arg1 : i32
func @cmplt_i32(%arg0: i32, %arg1: i32) -> i1 {
  %res = scl.cmplt %arg0, %arg1 : i32
  return %res : i1
}

// CHECK-LABEL: @cmplt_f32
// CHECK:       cmpf "ult", %arg0, %arg1 : f32
func @cmplt_f32(%arg0: f32, %arg1: f32) -> i1 {
  %res = scl.cmplt %arg0, %arg1 : f32
  return %res : i1
}


// MARK: scl.and

// CHECK-LABEL: @and
// CHECK:       and %arg0, %arg1 : i1
func @and(%arg0: i1, %arg1: i1) -> i1 {
  %res = scl.and %arg0, %arg1 : i1
  return %res : i1
}

// MARK: scl.or

// CHECK-LABEL: @or
// CHECK:       or %arg0, %arg1 : i1
func @or(%arg0: i1, %arg1: i1) -> i1 {
  %res = scl.or %arg0, %arg1 : i1
  return %res : i1
}

// MARK: scl.xor

// CHECK-LABEL: @xor
// CHECK:       xor %arg0, %arg1 : i1
func @xor(%arg0: i1, %arg1: i1) -> i1 {
  %res = scl.xor %arg0, %arg1 : i1
  return %res : i1
}

// MARK: scl.not

// CHECK-LABEL: @not
// CHECK:       [[FALSE:%[a-z_0-9]+]] = constant 0 : i1
// CHECK:       [[TRUE:%[a-z_0-9]+]] = constant 1 : i1
// CHECK:       select %arg0, [[FALSE]], [[TRUE]] : i1
func @not(%arg0: i1) -> i1 {
  %res = scl.not %arg0 : i1
  return %res : i1
}

