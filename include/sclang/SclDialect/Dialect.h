//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
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
// This file implements the IR Dialect for the SCL language.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_DIALECT_H_
#define SCLANG_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace scl {
namespace detail {
struct AddressTypeStorage;
struct ArrayTypeStorage;
struct StructTypeStorage;
struct BitWidthStorage;
} // end namespace detail

/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class SclDialect : public mlir::Dialect {
public:
  explicit SclDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "scl"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const;
};

/// Create a local enumeration with all of the types that are defined by Scl.
enum TypeKind {
  // TODO: register own space in mlir/include/mlir/IR/DialectSymbolRegistry.def
  SCL_TYPE = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  SCL_ADDRESS,
  SCL_ARRAY,
  SCL_STRUCT,
  SCL_INTEGER,
  SCL_LOGICAL,
  SCL_REAL,
};


/// Boilerplate mixin template
template <typename A, unsigned Id>
struct IntrinsicTypeMixin {
  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() { return Id; }
};


/// This class defines the SCL addres type.
class AddressType
  : public mlir::Type::TypeBase<AddressType, mlir::Type,
                                detail::AddressTypeStorage>,
    public IntrinsicTypeMixin<AddressType, SCL_ADDRESS> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `AddressType` with the given element types. There
  /// *must* be atleast one element type.
  static AddressType get(mlir::Type elementType);

  /// Returns the element types of this struct type.
  mlir::Type getElementType();
};

/// This class defines the SCL array type.
class ArrayType
  : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                detail::ArrayTypeStorage>,
    public IntrinsicTypeMixin<ArrayType, SCL_ARRAY> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;
  using DimTy = std::pair<int32_t, int32_t>;

  /// Create an instance of a `ArrayType` with the given dimensions and element type.
  static ArrayType get(llvm::ArrayRef<DimTy> dimensions,
                       mlir::Type elementType);

  /// Returns the dimensions of this array type.
  llvm::ArrayRef<DimTy> getDimensions();
  /// Returns the element types of this array type.
  mlir::Type getElementType();
};

/// This class defines the SCL struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType
  : public mlir::Type::TypeBase<StructType, mlir::Type,
                                detail::StructTypeStorage>,
    public IntrinsicTypeMixin<StructType, SCL_STRUCT> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};


class IntegerType
  : public mlir::Type::TypeBase<IntegerType, mlir::Type,
                                detail::BitWidthStorage>,
    public IntrinsicTypeMixin<IntegerType, SCL_INTEGER> {
public:
  using Base::Base;

  static IntegerType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();
};

class LogicalType
  : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                detail::BitWidthStorage>,
    public IntrinsicTypeMixin<LogicalType, SCL_LOGICAL> {
public:
  using Base::Base;

  static LogicalType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();
};

class RealType
  : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage>,
    public IntrinsicTypeMixin<RealType, SCL_REAL> {
public:
  using Base::Base;

  static RealType get(mlir::MLIRContext *ctx);
};


/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "sclang/SclDialect/Ops.h.inc"

} // end namespace scl
} // end namespace mlir

#endif // SCLANG_DIALECT_H_
