//===- Dialect.cpp - SCL IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the Scl IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "sclang/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::scl;

//===----------------------------------------------------------------------===//
// SclDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
SclDialect::SclDialect(mlir::MLIRContext *ctx) : mlir::Dialect("scl", ctx) {
  addOperations<
#define GET_OP_LIST
#include "sclang/Ops.cpp.inc"
      >();
  addTypes<ArrayType>();
  addTypes<StructType>();
}

//===----------------------------------------------------------------------===//
// Scl Types
//===----------------------------------------------------------------------===//
// MARK: ArrayType

namespace mlir {
namespace scl {
namespace detail {

struct ArrayTypeSpec {
  /// one dimension, with first and last valid index
  using DimTy = std::pair<int32_t, int32_t>;

  llvm::ArrayRef<DimTy> dimensions;
  mlir::Type elementType;

  ArrayTypeSpec(llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType)
      : dimensions(dimensions), elementType(elementType) {}
};

/// This class represents the internal storage of the SCL `ArrayType`.
struct ArrayTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = ArrayTypeSpec;
  using DimTy = KeyTy::DimTy;

  /// A constructor for the type storage instance.
  ArrayTypeStorage(llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType)
      : arrayType(dimensions, elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const {
    return key.elementType == arrayType.elementType && key.dimensions == arrayType.dimensions;
  }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key.elementType);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType) {
    return KeyTy(dimensions, elementType);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {

    // Copy the dimensions from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<DimTy> dimensions = allocator.copyInto(key.dimensions);
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(dimensions, key.elementType);
  }

  /// The following field contains the element types of the struct.
  ArrayTypeSpec arrayType;
};
} // end namespace detail
} // end namespace scl
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
ArrayType ArrayType::get(llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType) {
  assert(!dimensions.empty() && "expected an array with at least one dimension");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, SclTypes::Array, dimensions, elementType);
}

/// Returns the element types of this struct type.
mlir::Type ArrayType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->arrayType.elementType;
}

// Returns the element types of this struct type.
llvm::ArrayRef<ArrayType::DimTy> ArrayType::getDimensions() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->arrayType.dimensions;
}

namespace {

void print(ArrayType type, mlir::DialectAsmPrinter &printer) {
  // Print the struct type according to the parser format.
  printer << "array<" << type.getElementType();
  for (const auto dim : type.getDimensions()) {
    printer << ", " << dim.first << ":" << dim.second;
  }
  printer << '>';
}

/// Parse an array type instance.
mlir::Type parseArrayType(mlir::DialectAsmParser &parser) {
  // Parse a array type in the following form:
  //   array-type ::= `array` `<` type, `,`, range (`,` range)* `>`

  // Parse: `array` `<`
  if (parser.parseKeyword("array") || parser.parseLess())
    return Type();

  // Parse: type `,`
  mlir::Type elementType;
  if (parser.parseType(elementType) || parser.parseComma())
    return Type();

  // Parse the dimensions
  SmallVector<mlir::scl::detail::ArrayTypeStorage::DimTy, 1> dimensions;
  do {
    int32_t low, high;
    if (parser.parseInteger(low) || parser.parseColon() || parser.parseInteger(high))
      return Type();

    dimensions.push_back(std::make_pair(low, high));

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();

  return ArrayType::get(dimensions, elementType);
}

} // namespace


// MARK: StructType

namespace mlir {
namespace scl {
namespace detail {
/// This class represents the internal storage of the SCL `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
} // end namespace detail
} // end namespace scl
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, SclTypes::Struct, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

namespace {

void print(StructType type, mlir::DialectAsmPrinter &printer) {
  // Print the struct type according to the parser format.
  printer << "struct<";
  mlir::interleaveComma(type.getElementTypes(), printer);
  printer << '>';
}

/// Parse a struct type instance.
mlir::Type parseStructType(mlir::DialectAsmParser &parser) {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

} // namespace


// MARK: parse and print

/// Parse an instance of a type registered to the SCL dialect.
mlir::Type SclDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "array")
    return parseArrayType(parser);
  if (keyword == "struct")
    return parseStructType(parser);

  parser.emitError(parser.getNameLoc(), "unknown SCL type: ") << keyword;
  return Type();
}

/// Print an instance of a type registered to the SCL dialect.
void SclDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  switch(type.getKind()) {
  default:
    llvm_unreachable("Unhandled SCL type");
  case SclTypes::Array:
    print(type.cast<ArrayType>(), printer);
    break;
  case SclTypes::Struct:
    print(type.cast<StructType>(), printer);
    break;
  }
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sclang/Ops.cpp.inc"
