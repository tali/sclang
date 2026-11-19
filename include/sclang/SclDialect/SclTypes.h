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

#include "llvm/ADT/SmallSet.h"
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
  static constexpr ::llvm::StringLiteral name = "scl.address";

  /// Create an instance of `AddressType` with the given element types. There
  /// *must* be at least one element type.
  static AddressType get(mlir::Type elementType);

  /// Verifier hook (called by TypeBase::getChecked): the element type must
  /// be present.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   mlir::Type elementType) {
    if (!elementType)
      return emitError() << "scl.address requires a non-null element type";
    return mlir::success();
  }

  /// Returns the element types of this struct type.
  mlir::Type getElementType();
};

/// This class defines the SCL array type.
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              detail::ArrayTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.array";
  using DimTy = std::pair<int32_t, int32_t>;

  /// Create an instance of `ArrayType` with the given dimensions and element
  /// type.
  static ArrayType get(llvm::ArrayRef<DimTy> dimensions,
                       mlir::Type elementType);

  /// Verifier hook: at least one dimension, lower <= upper per dimension,
  /// non-null element type.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<DimTy> dimensions, mlir::Type elementType) {
    if (dimensions.empty())
      return emitError() << "scl.array requires at least one dimension";
    for (auto [lo, hi] : dimensions)
      if (lo > hi)
        return emitError() << "scl.array dimension [" << lo << ".." << hi
                           << "] has lower > upper";
    if (!elementType)
      return emitError() << "scl.array requires a non-null element type";
    return mlir::success();
  }

  /// Returns the dimensions of this array type.
  llvm::ArrayRef<DimTy> getDimensions();
  /// Returns the element types of this array type.
  mlir::Type getElementType();
};

class DateType
    : public mlir::Type::TypeBase<DateType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.date";
};

class DateAndTimeType
    : public mlir::Type::TypeBase<DateAndTimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.dateandtime";
};

/// This class defines the SCL type for an instance DB
class InstanceDbType : public mlir::Type::TypeBase<InstanceDbType, mlir::Type,detail::InstanceDbTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.idb";

  /// Create an instance of `InstanceDbType` given a reference to the function block.
  static InstanceDbType get(mlir::MLIRContext *ctx, StringRef fbSymbol);

  /// Verifier hook: the FB symbol must be non-empty.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   StringRef fbSymbol) {
    if (fbSymbol.empty())
      return emitError() << "scl.idb requires a non-empty FB symbol";
    return mlir::success();
  }

  StringRef getFbSymbol();
};

class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type,
                                                detail::BitWidthStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.int";

  static IntegerType get(mlir::MLIRContext *ctx, int width);

  /// Verifier hook: only the SCL integer widths are valid.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   int width) {
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      return emitError() << "invalid scl integer width " << width
                         << " (must be 1, 8, 16, 32 or 64)";
    return mlir::success();
  }

  /// Return the bit width of this type.
  int getWidth();
};

class LogicalType : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                                detail::BitWidthStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.bool";

  static LogicalType get(mlir::MLIRContext *ctx, int width);

  /// Verifier hook: only the SCL logical widths are valid (bool/byte/word/
  /// dword and the 64-bit direct-load width).
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   int width) {
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      return emitError() << "invalid scl logical width " << width
                         << " (must be 1, 8, 16, 32 or 64)";
    return mlir::success();
  }

  /// Return the bit width of this type.
  int getWidth();
};

class RealType
    : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.real";
};

/// IEC 61131-3 LREAL: binary64 real (ADR 004). Lowers to f64.
class LrealType
    : public mlir::Type::TypeBase<LrealType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.lreal";
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
  static constexpr ::llvm::StringLiteral name = "scl.struct";

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::StringAttr> elementNames,
                        llvm::ArrayRef<mlir::Type> elementTypes);

  /// Verifier hook: names/types counts match, at least one element, unique
  /// names, non-null types.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<mlir::StringAttr> elementNames,
                   llvm::ArrayRef<mlir::Type> elementTypes) {
    if (elementNames.empty())
      return emitError() << "scl.struct requires at least one element";
    if (elementNames.size() != elementTypes.size())
      return emitError() << "scl.struct name/type count mismatch";
    llvm::SmallSet<llvm::StringRef, 8> seen;
    for (auto name : elementNames) {
      if (!seen.insert(name.getValue()).second)
        return emitError() << "duplicate scl.struct member '" << name << "'";
    }
    for (auto t : elementTypes)
      if (!t)
        return emitError() << "scl.struct member with null type";
    return mlir::success();
  }

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
  static constexpr ::llvm::StringLiteral name = "scl.s5time";
};

class TimeType
    : public mlir::Type::TypeBase<TimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.time";
};

class TimeOfDayType
    : public mlir::Type::TypeBase<TimeOfDayType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.timeofday";
};

/// Area of a directly-represented variable (S7 process image / data block).
/// The enum values are the ASCII letters used in SCL (`%I`, `%Q`, ...) and
/// in the runtime C ABI (ADR 002).
enum class DirectArea : uint8_t {
  Input = 'I',     // %I / AWL E
  Output = 'Q',    // %Q / AWL A
  Marker = 'M',    // %M / AWL M
  Local = 'L',     // %L / AWL LW/LB/LD — per-call L-stack frame (see ADR)
  VLocal = 'V',    // AWL 0x87 VL — the *parent* frame: FC parameters live in
                   // the caller's L frame (awlsim callstack: MEM_VL)
  DataBlock = 'D', // %DBn / AWL DBn (db = DB number)
  Timer = 'T',     // %T / AWL T
  Counter = 'C',   // %C / AWL Z
};

namespace detail {
/// Uniquing key of DirectType (defined here so verifyInvariants can name it).
struct DirectTypeKey {
  DirectArea area;
  int32_t db, byte, bit, width;
  bool operator==(const DirectTypeKey &key) const {
    return area == key.area && db == key.db && byte == key.byte &&
           bit == key.bit && width == key.width;
  }
};
} // namespace detail

/// This class defines the SCL type of a directly-represented variable, e.g.
/// `%I0.0`, `%MW10` or `%DB1.DBW4`. Printed as
/// `!scl.direct<M, 0, 10, 3, 1>` = area, db, byte, bit, width-in-bits.
/// Loads/stores through `address<direct<...>>` lower to runtime calls
/// (ADR 002), never to inline memref access.
class DirectType : public mlir::Type::TypeBase<DirectType, mlir::Type,
                                               detail::DirectTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "scl.direct";

  /// Create a direct type. `width` is the access width in bits
  /// (1 = bit, 8 = byte, 16 = word, 32 = dword).
  static DirectType get(mlir::MLIRContext *ctx, DirectArea area, int32_t db,
                        int32_t byte, int32_t bit, int32_t width);

  /// Verifier hook (args match the KeyTy passed to Base::get): sane
  /// db/byte/bit offsets and a valid access width.
  static mlir::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   detail::DirectTypeKey key);

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
