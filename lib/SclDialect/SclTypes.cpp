//===- SclTypes.cpp - SCL IR Dialect registration in MLIR -----------------===//
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

#include "sclang/SclDialect/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::scl;


//===----------------------------------------------------------------------===//
// Scl Types
//===----------------------------------------------------------------------===//

// MARK: AddressType

namespace mlir {
namespace scl {
namespace detail {

/// This class represents the internal storage of the SCL `AddressType`.
struct AddressTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = Type;

  /// A constructor for the type storage instance.
  AddressTypeStorage(Type elementType) : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const {
    return key == elementType;
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static AddressTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AddressTypeStorage>())
      AddressTypeStorage(key);
  }

  /// The following field contains the element types of the struct.
  Type elementType;
};
} // end namespace detail
} // end namespace scl
} // end namespace mlir

/// Create an instance of a `AddressType` with the given element type.
AddressType AddressType::get(mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// Returns the element types of this struct type.
mlir::Type AddressType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementType;
}

namespace {

void print(AddressType type, mlir::DialectAsmPrinter &printer) {
  // Print the struct type according to the parser format.
  printer << "address<" << type.getElementType() << '>';
}

/// Parse an array type instance.
mlir::Type parseAddressType(mlir::DialectAsmParser &parser) {
  // Parse a array type in the following form:
  //   array-type ::= `address` `<` type `>`

  // Parse: type `>`
  mlir::Type elementType;
  if (parser.parseLess() ||
      parser.parseType(elementType) ||
      parser.parseGreater())
    return Type();

  return AddressType::get(elementType);
}

} // namespace


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
    return key.elementType == arrayType.elementType &&
           key.dimensions == arrayType.dimensions;
  }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_value(key.elementType);
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
ArrayType ArrayType::get(llvm::ArrayRef<DimTy> dimensions,
                         mlir::Type elementType) {
  assert(!dimensions.empty() &&
         "expected an array with at least one dimension");

  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, dimensions, elementType);
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
  for (const auto & dim : type.getDimensions()) {
    printer << ", " << dim.first << ":" << dim.second;
  }
  printer << '>';
}

/// Parse an array type instance.
mlir::Type parseArrayType(mlir::DialectAsmParser &parser) {
  // Parse a array type in the following form:
  //   array-type ::= `array` `<` type, `,`, range (`,` range)* `>`

  // Parse: `<`
  if (parser.parseLess())
    return Type();

  // Parse: type `,`
  mlir::Type elementType;
  if (parser.parseType(elementType) || parser.parseComma())
    return Type();

  // Parse the dimensions
  SmallVector<mlir::scl::detail::ArrayTypeStorage::DimTy, 1> dimensions;
  do {
    int32_t low, high;
    if (parser.parseInteger(low) || parser.parseColon() ||
        parser.parseInteger(high))
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
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

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

  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
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
  llvm::interleaveComma(type.getElementTypes(), printer);
  printer << '>';
}

/// Parse a struct type instance.
mlir::Type parseStructType(mlir::DialectAsmParser &parser) {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // Parse: `<`
  if (parser.parseLess())
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

// MARK: integer and logical types

namespace mlir {
namespace scl {
namespace detail {

/// This class represents the internal storage of the SCL `IntegerType` and `LogicalType`.
struct BitWidthStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = int;

  /// A constructor for the type storage instance.
  BitWidthStorage(KeyTy width)
      : width(width) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const {
    return key == width;
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static BitWidthStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {

    // Copy the the provided `KeyTy` into the allocator.
    return new (allocator.allocate<BitWidthStorage>()) BitWidthStorage(key);
  }

  /// The bit-width of the type.
  KeyTy width;
};
} // end namespace detail

IntegerType IntegerType::get(mlir::MLIRContext *ctx, int width) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in.
  // The other parameters are forwarded to the storage instance.
  return Base::get(ctx, width);
}

int IntegerType::getWidth() {
  return getImpl()->width;
}

LogicalType LogicalType::get(mlir::MLIRContext *ctx, int width) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in.
  // The other parameters are forwarded to the storage instance.
  return Base::get(ctx, width);
}

int LogicalType::getWidth() {
  return getImpl()->width;
}

} // end namespace scl
} // end namespace mlir


// MARK: parse and print

/// Parse an instance of a type registered to the SCL dialect.
mlir::Type SclDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "address")
    return parseAddressType(parser);
  if (keyword == "array")
    return parseArrayType(parser);
  if (keyword == "struct")
    return parseStructType(parser);

  if (keyword == "char")
    return IntegerType::get(getContext(), 8);
  if (keyword == "int")
    return IntegerType::get(getContext(), 16);
  if (keyword == "dint")
    return IntegerType::get(getContext(), 32);

  if (keyword == "bool")
    return LogicalType::get(getContext(), 1);
  if (keyword == "byte")
    return LogicalType::get(getContext(), 8);
  if (keyword == "word")
    return LogicalType::get(getContext(), 16);
  if (keyword == "dword")
    return LogicalType::get(getContext(), 32);

  if (keyword == "real")
    return RealType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown SCL type: ") << keyword;
  return Type();
}

/// Print an instance of a type registered to the SCL dialect.
void SclDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
  .Case<AddressType>([&](AddressType type) {
    print(type, printer);
  })
  .Case<ArrayType>([&](ArrayType type) {
    print(type, printer);
  })
  .Case<StructType>([&](StructType type) {
    print(type, printer);
  })
  .Case<IntegerType>([&](IntegerType type) {
    switch (type.getWidth()) {
    default:
      llvm_unreachable("Unhandled IntegerType bit width");
    case 8:
      printer << "char";
      break;
    case 16:
      printer << "int";
      break;
    case 32:
      printer << "dint";
      break;
    }
  })
  .Case<LogicalType>([&](LogicalType type) {
    switch (type.getWidth()) {
    default:
    case 1:
      printer << "bool";
      break;
    case 8:
      printer << "byte";
      break;
    case 16:
      printer << "word";
      break;
    case 32:
      printer << "dword";
      break;
    }
  })
  .Case<RealType>([&](RealType) {
    printer << "real";
  })
  .Default([&](Type) { llvm_unreachable("Unhandled SCL type"); });
}
