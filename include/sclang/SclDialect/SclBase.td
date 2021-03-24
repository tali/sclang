//===- Ops.td - SCL dialect operation definitions ----------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines the operations of the SCL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SCL_BASE
#define SCL_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AssemblyFormat<string format> {
  string assemblyFormat = format;
}


// Provide a definition of the 'scl' dialect in the ODS framework so that we
// can define our operations.
def Scl_Dialect : Dialect {
  let name = "scl";
  let cppNamespace = "::mlir::scl";
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;

    void registerTypes();
  }];
}

def Scl_AddressType : Type<CPred<"$_self.isa<scl::AddressType>()">, "AddressType">;
def Scl_ArrayType : Type<CPred<"$_self.isa<scl::ArrayType>()">, "ArrayType">;
def Scl_InstanceDbType : Type<CPred<"$_self.isa<scl::InstanceDbType>()">, "IDB">;
def Scl_IntegerType : Type<CPred<"$_self.isa<scl::IntegerType>()">, "IntegerType">;
def Scl_LogicalType : Type<CPred<"$_self.isa<scl::LogicalType>()">, "LogicalType">;
def Scl_RealType : Type<CPred<"$_self.isa<scl::RealType>()">, "RealType">;
def Scl_StructType : Type<CPred<"$_self.isa<scl::StructType>()">, "StructType">;

def Scl_Bool :
  Type<CPred<"$_self.isa<::LogicalType>() && $_self.cast<::LogicalType>().getWidth() == 1">, "bool">,
  BuildableType<"LogicalType::get($_builder.getContext(), 1)">;

// Any SCL type which can be used for function arguments
def Scl_Type : AnyTypeOf<[
  Scl_ArrayType,
  Scl_InstanceDbType,
  Scl_IntegerType,
  Scl_LogicalType,
  Scl_RealType,
  Scl_StructType
]>;
def Scl_NumericType : AnyTypeOf<[Scl_IntegerType, Scl_RealType]>;


// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class SclOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Scl_Dialect, mnemonic, traits>;

class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs AnyType:$result)>,
    AssemblyFormat<"attr-dict $lhs `,` $rhs `:` type($rhs)">;
class CompareOp<string mnemonic, list<OpTrait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameTypeOperands])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs Scl_Bool:$result)>,
    AssemblyFormat<"attr-dict $lhs `,` $rhs `:` type($rhs)"> {
      let builders = [OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
        auto resType = LogicalType::get($_builder.getContext(), 1);
        build($_builder, $_state, resType, lhs, rhs);
      }]>];
    }
class UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$rhs)>,
    Results<(outs AnyType:$result)>,
    AssemblyFormat<"attr-dict $rhs `:` type($rhs)">;


// Check that an op can only be used within the scope of a module-like op.
def InModuleScope : PredOpTrait<
  "op must appear in a module-like op's block",
//TBD  CPred<"isDirectInModuleLikeOp($_op.getParentOp())">>;
CPred<"true">>;

class TopLevelOp<string mnemonic, list<OpTrait> traits = []> :
  SclOp<mnemonic, !listconcat(traits, [InModuleScope])>;

class FuncTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [NoSideEffect, Terminator])>;


#endif // SCL_BASE
