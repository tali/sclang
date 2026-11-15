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

#ifndef SCLANG_DIALECT_SCLTYPES_H_
#define SCLANG_DIALECT_SCLTYPES_H_

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace scl {
namespace detail {
struct AddressTypeStorage;
struct ArrayTypeStorage;
struct BitWidthStorage;
struct DirectTypeStorage;
struct InstanceDbTypeStorage;
struct StructTypeStorage;
} // end namespace detail

/// This class defines the SCL addres type.
class AddressType : public mlir::Type::TypeBase<AddressType, mlir::Type,
                                                detail::AddressTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of `AddressType` with the given element types. There
  /// *must* be at least one element type.
  static AddressType get(mlir::Type elementType);

  /// Returns the element types of this struct type.
  mlir::Type getElementType();
};

/// This class defines the SCL array type.
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              detail::ArrayTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;
  using DimTy = std::pair<int32_t, int32_t>;

  /// Create an instance of `ArrayType` with the given dimensions and element
  /// type.
  static ArrayType get(llvm::ArrayRef<DimTy> dimensions,
                       mlir::Type elementType);

  /// Returns the dimensions of this array type.
  llvm::ArrayRef<DimTy> getDimensions();
  /// Returns the element types of this array type.
  mlir::Type getElementType();
};

class DateType
    : public mlir::Type::TypeBase<DateType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class DateAndTimeType
    : public mlir::Type::TypeBase<DateAndTimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

/// This class defines the SCL type for an instance DB
class InstanceDbType : public mlir::Type::TypeBase<InstanceDbType, mlir::Type,detail::InstanceDbTypeStorage> {
public:
  using Base::Base;

  /// Create an instance of `InstanceDbType` given a reference to the function block.
  static InstanceDbType get(mlir::MLIRContext *ctx, StringRef fbSymbol);

  StringRef getFbSymbol();
};

class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type,
                                                detail::BitWidthStorage> {
public:
  using Base::Base;

  static IntegerType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();
};

class LogicalType : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                                detail::BitWidthStorage> {
public:
  using Base::Base;

  static LogicalType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();
};

class RealType
    : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
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

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::StringAttr> elementNames,
                        llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the elements of this struct type.
  llvm::ArrayRef<mlir::StringAttr> getElementNames();

  /// Returns the elements of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the type of the given element.
  mlir::Type getElementType(mlir::StringAttr name);
  mlir::Type getElementType(llvm::StringRef name) {
    return getElementType(StringAttr::get(getContext(), name));
  }

  /// Returns the number of elements held by this struct.
  size_t getNumElements() { return getElementTypes().size(); }
};

class S5TimeType
    : public mlir::Type::TypeBase<S5TimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class TimeType
    : public mlir::Type::TypeBase<TimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class TimeOfDayType
    : public mlir::Type::TypeBase<TimeOfDayType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

/// Area of a directly-represented variable (S7 process image / data block).
/// The enum values are the ASCII letters used in SCL (`%I`, `%Q`, ...) and
/// in the runtime C ABI (ADR 002).
enum class DirectArea : uint8_t {
  Input = 'I',     // %I / AWL E
  Output = 'Q',    // %Q / AWL A
  Marker = 'M',    // %M / AWL M
  Local = 'L',     // %L / AWL LW/LB/LD — per-call L-stack frame (see ADR)
  DataBlock = 'D', // %DBn / AWL DBn (db = DB number)
  Timer = 'T',     // %T / AWL T
  Counter = 'C',   // %C / AWL Z
};

/// This class defines the SCL type of a directly-represented variable, e.g.
/// `%I0.0`, `%MW10` or `%DB1.DBW4`. Printed as
/// `!scl.direct<M, 0, 10, 3, 1>` = area, db, byte, bit, width-in-bits.
/// Loads/stores through `address<direct<...>>` lower to runtime calls
/// (ADR 002), never to inline memref access.
class DirectType : public mlir::Type::TypeBase<DirectType, mlir::Type,
                                               detail::DirectTypeStorage> {
public:
  using Base::Base;

  /// Create a direct type. `width` is the access width in bits
  /// (1 = bit, 8 = byte, 16 = word, 32 = dword).
  static DirectType get(mlir::MLIRContext *ctx, DirectArea area, int32_t db,
                        int32_t byte, int32_t bit, int32_t width);

  DirectArea getArea() const;
  int32_t getDb() const;
  int32_t getByte() const;
  int32_t getBit() const;
  int32_t getWidth() const;

  /// The SCL value type of a load from this address: bool for bit access,
  /// otherwise an integer of the access width.
  mlir::Type getValueType() const;

  /// The area name used in SCL/AWL source and in the MLIR type syntax
  /// ("I", "Q", "M", "DB", "T", "C").
  static llvm::StringRef areaName(DirectArea area);
  /// Parse an area name; nullopt if not a valid area.
  static std::optional<DirectArea> areaFromName(llvm::StringRef name);
};

} // end namespace scl
} // end namespace mlir

#endif // SCLANG_DIALECT_SCLTYPES_H_
