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
include "mlir/Interfaces/InferTypeOpInterface.td"
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
  private:
    void registerOps();
    void registerTypes();
  }];
}

def Scl_AddressType : Type<CPred<"::llvm::isa<scl::AddressType>($_self)">, "AddressType">;
def Scl_ArrayType : Type<CPred<"::llvm::isa<scl::ArrayType>($_self)">, "ArrayType">;
def Scl_DateType : Type<CPred<"::llvm::isa<scl::DateType>($_self)">, "DateType">;
def Scl_DateAndTimeType : Type<CPred<"::llvm::isa<scl::DateAndTimeType>($_self)">, "DateAndTimeType">;
def Scl_InstanceDbType : Type<CPred<"::llvm::isa<scl::InstanceDbType>($_self)">, "IDB">;
def Scl_IntegerType : Type<CPred<"::llvm::isa<scl::IntegerType>($_self)">, "IntegerType">;
def Scl_LogicalType : Type<CPred<"::llvm::isa<scl::LogicalType>($_self)">, "LogicalType">;
def Scl_RealType : Type<CPred<"::llvm::isa<scl::RealType>($_self)">, "RealType">;
def Scl_LrealType : Type<CPred<"::llvm::isa<scl::LrealType>($_self)">, "LrealType">;
def Scl_BuiltinIntegerType : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, "builtin-integer">;
def Scl_StructType : Type<CPred<"::llvm::isa<scl::StructType>($_self)">, "StructType">;
def Scl_S5TimeType : Type<CPred<"::llvm::isa<scl::S5TimeType>($_self)">, "S5TimeType">;
def Scl_TimeType : Type<CPred<"::llvm::isa<scl::TimeType>($_self)">, "TimeType">;
def Scl_TimeOfDayType : Type<CPred<"::llvm::isa<scl::TimeOfDayType>($_self)">, "TimeOfDayType">;

def Scl_Bool :
  Type<CPred<"::llvm::isa<::LogicalType>($_self) && ::llvm::cast<::LogicalType>($_self).getWidth() == 1">, "bool">,
  BuildableType<"LogicalType::get($_builder.getContext(), 1)">;

// Any SCL type which can be used for function arguments
def Scl_Type : AnyTypeOf<[
  Scl_ArrayType,
  Scl_DateType,
  Scl_DateAndTimeType,
  Scl_InstanceDbType,
  Scl_IntegerType,
  Scl_LogicalType,
  Scl_RealType,
  Scl_LrealType,
  Scl_StructType,
  Scl_S5TimeType,
  Scl_TimeType,
  Scl_TimeOfDayType
]>;
def Scl_NumericType : AnyTypeOf<[Scl_IntegerType, Scl_RealType, Scl_LrealType]>;


// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class SclOp<string mnemonic, list<Trait> traits = []> :
    Op<Scl_Dialect, mnemonic, traits>;

class ArithmeticOp<string mnemonic, list<Trait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs AnyType:$result)>,
    AssemblyFormat<"attr-dict $lhs `,` $rhs `:` type($rhs)">;
class CompareOp<string mnemonic, list<Trait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameTypeOperands])>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs Scl_Bool:$result)>,
    AssemblyFormat<"attr-dict $lhs `,` $rhs `:` type($rhs)"> {
//      let builders = [OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
//        auto resType = LogicalType::get($_builder.getContext(), 1);
//        build($_builder, $_state, resType, lhs, rhs);
//      }]>];
    }
class UnaryArithmeticOp<string mnemonic, list<Trait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins AnyType:$rhs)>,
    Results<(outs AnyType:$result)>,
    AssemblyFormat<"attr-dict $rhs `:` type($rhs)">;


// Check that an op can only be used within the scope of a module-like op.
def InModuleScope : PredOpTrait<
  "op must appear in a module-like op's block",
//TBD  CPred<"isDirectInModuleLikeOp($_op.getParentOp())">>;
CPred<"true">>;


class TopLevelOp<string mnemonic, list<Trait> traits = []> :
  SclOp<mnemonic, !listconcat(traits, [InModuleScope])>;

class FuncTerminatorOp<string mnemonic, list<Trait> traits = []> :
    SclOp<mnemonic, !listconcat(traits, [Pure, Terminator])>;


#endif // SCL_BASE
