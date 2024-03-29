// RUN: sclang-gen --emit=mlir-std -x=mlir %s | FileCheck %s

// MARK: scl.add

// CHECK-LABEL: func @add_i8
// CHECK:       arith.addi %arg0, %arg1 : i8
scl.function @add_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.add %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// CHECK-LABEL: func @add_i16
// CHECK:       arith.addi %arg0, %arg1 : i16
scl.function @add_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.int {
  %res = scl.add %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.int>
  scl.store %return : !scl.address<!scl.int> = %res : !scl.int
  scl.return.value %return : !scl.address<!scl.int>
}

// CHECK-LABEL: func @add_i32
// CHECK:       arith.addi %arg0, %arg1 : i32
scl.function @add_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.dint {
  %res = scl.add %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.dint>
  scl.store %return : !scl.address<!scl.dint> = %res : !scl.dint
  scl.return.value %return : !scl.address<!scl.dint>
}

// CHECK-LABEL: func @add_f32
// CHECK:       arith.addf %arg0, %arg1 : f32
scl.function @add_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.real {
  %res = scl.add %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.real>
  scl.store %return : !scl.address<!scl.real> = %res : !scl.real
  scl.return.value %return : !scl.address<!scl.real>
}


// MARK: scl.sub

// CHECK-LABEL: func @sub_i8
// CHECK:       arith.subi %arg0, %arg1 : i8
scl.function @sub_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.sub %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// CHECK-LABEL: func @sub_i16
// CHECK:       arith.subi %arg0, %arg1 : i16
scl.function @sub_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.int {
  %res = scl.sub %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.int>
  scl.store %return : !scl.address<!scl.int> = %res : !scl.int
  scl.return.value %return : !scl.address<!scl.int>
}

// CHECK-LABEL: func @sub_i32
// CHECK:       arith.subi %arg0, %arg1 : i32
scl.function @sub_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.dint {
  %res = scl.sub %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.dint>
  scl.store %return : !scl.address<!scl.dint> = %res : !scl.dint
  scl.return.value %return : !scl.address<!scl.dint>
}

// CHECK-LABEL: func @sub_f32
// CHECK:       arith.subf %arg0, %arg1 : f32
scl.function @sub_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.real {
  %res = scl.sub %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.real>
  scl.store %return : !scl.address<!scl.real> = %res : !scl.real
  scl.return.value %return : !scl.address<!scl.real>
}


// MARK: scl.mul

// CHECK-LABEL: func @mul_i8
// CHECK:       arith.muli %arg0, %arg1 : i8
scl.function @mul_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.mul %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// CHECK-LABEL: func @mul_i16
// CHECK:       arith.muli %arg0, %arg1 : i16
scl.function @mul_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.int {
  %res = scl.mul %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.int>
  scl.store %return : !scl.address<!scl.int> = %res : !scl.int
  scl.return.value %return : !scl.address<!scl.int>
}

// CHECK-LABEL: func @mul_i32
// CHECK:       arith.muli %arg0, %arg1 : i32
scl.function @mul_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.dint {
  %res = scl.mul %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.dint>
  scl.store %return : !scl.address<!scl.dint> = %res : !scl.dint
  scl.return.value %return : !scl.address<!scl.dint>
}

// CHECK-LABEL: func @mul_f32
// CHECK:       arith.mulf %arg0, %arg1 : f32
scl.function @mul_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.real {
  %res = scl.mul %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.real>
  scl.store %return : !scl.address<!scl.real> = %res : !scl.real
  scl.return.value %return : !scl.address<!scl.real>
}


// MARK: scl.div

// CHECK-LABEL: func @div_i8
// CHECK:       arith.divsi %arg0, %arg1 : i8
scl.function @div_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.div %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// CHECK-LABEL: func @div_i16
// CHECK:       arith.divsi %arg0, %arg1 : i16
scl.function @div_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.int {
  %res = scl.div %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.int>
  scl.store %return : !scl.address<!scl.int> = %res : !scl.int
  scl.return.value %return : !scl.address<!scl.int>
}

// CHECK-LABEL: func @div_i32
// CHECK:       arith.divsi %arg0, %arg1 : i32
scl.function @div_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.dint {
  %res = scl.div %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.dint>
  scl.store %return : !scl.address<!scl.dint> = %res : !scl.dint
  scl.return.value %return : !scl.address<!scl.dint>
}

// CHECK-LABEL: func @div_f32
// CHECK:       arith.divf %arg0, %arg1 : f32
scl.function @div_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.real {
  %res = scl.div %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.real>
  scl.store %return : !scl.address<!scl.real> = %res : !scl.real
  scl.return.value %return : !scl.address<!scl.real>
}


// MARK: scl.mod

// CHECK-LABEL: func @mod_i8
// CHECK:       arith.remsi %arg0, %arg1 : i8
scl.function @mod_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.char {
  %res = scl.mod %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.char>
  scl.store %return : !scl.address<!scl.char> = %res : !scl.char
  scl.return.value %return : !scl.address<!scl.char>
}

// CHECK-LABEL: func @mod_i16
// CHECK:       arith.remsi %arg0, %arg1 : i16
scl.function @mod_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.int {
  %res = scl.mod %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.int>
  scl.store %return : !scl.address<!scl.int> = %res : !scl.int
  scl.return.value %return : !scl.address<!scl.int>
}

// CHECK-LABEL: func @mod_i32
// CHECK:       arith.remsi %arg0, %arg1 : i32
scl.function @mod_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.dint {
  %res = scl.mod %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.dint>
  scl.store %return : !scl.address<!scl.dint> = %res : !scl.dint
  scl.return.value %return : !scl.address<!scl.dint>
}

// CHECK-LABEL: func @mod_f32
// CHECK:       arith.remf %arg0, %arg1 : f32
scl.function @mod_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.real {
  %res = scl.mod %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.real>
  scl.store %return : !scl.address<!scl.real> = %res : !scl.real
  scl.return.value %return : !scl.address<!scl.real>
}


// MARK: scl.cmpeq

// CHECK-LABEL: func @cmpeq_i8
// CHECK:       arith.cmpi eq, %arg0, %arg1 : i8
scl.function @cmpeq_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmpeq %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpeq_i16
// CHECK:       arith.cmpi eq, %arg0, %arg1 : i16
scl.function @cmpeq_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmpeq %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpeq_i32
// CHECK:       arith.cmpi eq, %arg0, %arg1 : i32
scl.function @cmpeq_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmpeq %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpeq_f32
// CHECK:       arith.cmpf ueq, %arg0, %arg1 : f32
scl.function @cmpeq_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmpeq %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.cmpne

// CHECK-LABEL: func @cmpne_i8
// CHECK:       arith.cmpi ne, %arg0, %arg1 : i8
scl.function @cmpne_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmpne %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpne_i16
// CHECK:       arith.cmpi ne, %arg0, %arg1 : i16
scl.function @cmpne_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmpne %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpne_i32
// CHECK:       arith.cmpi ne, %arg0, %arg1 : i32
scl.function @cmpne_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmpne %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpne_f32
// CHECK:       arith.cmpf une, %arg0, %arg1 : f32
scl.function @cmpne_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmpne %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.cmpge

// CHECK-LABEL: func @cmpge_i8
// CHECK:       arith.cmpi sge, %arg0, %arg1 : i8
scl.function @cmpge_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmpge %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpge_i16
// CHECK:       arith.cmpi sge, %arg0, %arg1 : i16
scl.function @cmpge_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmpge %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpge_i32
// CHECK:       arith.cmpi sge, %arg0, %arg1 : i32
scl.function @cmpge_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmpge %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpge_f32
// CHECK:       arith.cmpf uge, %arg0, %arg1 : f32
scl.function @cmpge_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmpge %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.cmpgt

// CHECK-LABEL: func @cmpgt_i8
// CHECK:       arith.cmpi sgt, %arg0, %arg1 : i8
scl.function @cmpgt_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmpgt %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpgt_i16
// CHECK:       arith.cmpi sgt, %arg0, %arg1 : i16
scl.function @cmpgt_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmpgt %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpgt_i32
// CHECK:       arith.cmpi sgt, %arg0, %arg1 : i32
scl.function @cmpgt_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmpgt %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmpgt_f32
// CHECK:       arith.cmpf ugt, %arg0, %arg1 : f32
scl.function @cmpgt_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmpgt %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.cmple

// CHECK-LABEL: func @cmple_i8
// CHECK:       arith.cmpi sle, %arg0, %arg1 : i8
scl.function @cmple_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmple %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmple_i16
// CHECK:       arith.cmpi sle, %arg0, %arg1 : i16
scl.function @cmple_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmple %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmple_i32
// CHECK:       arith.cmpi sle, %arg0, %arg1 : i32
scl.function @cmple_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmple %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmple_f32
// CHECK:       arith.cmpf ule, %arg0, %arg1 : f32
scl.function @cmple_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmple %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.cmplt

// CHECK-LABEL: func @cmplt_i8
// CHECK:       arith.cmpi slt, %arg0, %arg1 : i8
scl.function @cmplt_i8(%arg0: !scl.char, %arg1: !scl.char) -> !scl.bool {
  %res = scl.cmplt %arg0, %arg1 : !scl.char
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmplt_i16
// CHECK:       arith.cmpi slt, %arg0, %arg1 : i16
scl.function @cmplt_i16(%arg0: !scl.int, %arg1: !scl.int) -> !scl.bool {
  %res = scl.cmplt %arg0, %arg1 : !scl.int
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmplt_i32
// CHECK:       arith.cmpi slt, %arg0, %arg1 : i32
scl.function @cmplt_i32(%arg0: !scl.dint, %arg1: !scl.dint) -> !scl.bool {
  %res = scl.cmplt %arg0, %arg1 : !scl.dint
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// CHECK-LABEL: func @cmplt_f32
// CHECK:       arith.cmpf ult, %arg0, %arg1 : f32
scl.function @cmplt_f32(%arg0: !scl.real, %arg1: !scl.real) -> !scl.bool {
  %res = scl.cmplt %arg0, %arg1 : !scl.real
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}


// MARK: scl.and

// CHECK-LABEL: func @and
// CHECK:       arith.andi %arg0, %arg1 : i1
scl.function @and(%arg0: !scl.bool, %arg1: !scl.bool) -> !scl.bool {
  %res = scl.and %arg0, %arg1 : !scl.bool
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// MARK: scl.or

// CHECK-LABEL: func @or
// CHECK:       arith.ori %arg0, %arg1 : i1
scl.function @or(%arg0: !scl.bool, %arg1: !scl.bool) -> !scl.bool {
  %res = scl.or %arg0, %arg1 : !scl.bool
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// MARK: scl.xor

// CHECK-LABEL: func @xor
// CHECK:       arith.xori %arg0, %arg1 : i1
scl.function @xor(%arg0: !scl.bool, %arg1: !scl.bool) -> !scl.bool {
  %res = scl.xor %arg0, %arg1 : !scl.bool
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

// MARK: scl.not

// CHECK-LABEL: func @not
// CHECK-DAG:       [[TRUE:%[a-z_0-9]+]] = arith.constant true
// CHECK:       arith.xori %arg0, [[TRUE]] : i1
scl.function @not(%arg0: !scl.bool) -> !scl.bool {
  %res = scl.not %arg0 : !scl.bool
  %return = scl.tmpvar "return" : !scl.address<!scl.bool>
  scl.store %return : !scl.address<!scl.bool> = %res : !scl.bool
  scl.return.value %return : !scl.address<!scl.bool>
}

