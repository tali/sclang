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
// This file implements the IR Dialect for the Toy language.
// See g3doc/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_DIALECT_H_
#define SCLANG_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace scl {
namespace detail {
struct ArrayTypeStorage;
struct StructTypeStorage;
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
namespace SclTypes {
enum Types {
  // TODO: register own space in mlir/include/mlir/IR/DialectSymbolRegistry.def
  Array = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  Struct,
};
} // end namespace SclTypes

/// This class defines the SCL array type.
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                               detail::ArrayTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;
  using DimTy = std::pair<int32_t, int32_t>;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == SclTypes::Array; }

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static ArrayType get(llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<DimTy> getDimensions();
  mlir::Type getElementType();
};

/// This class defines the SCL struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == SclTypes::Struct; }

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};


/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "sclang/Ops.h.inc"

} // end namespace scl
} // end namespace mlir

#endif // SCLANG_DIALECT_H_
