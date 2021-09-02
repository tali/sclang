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

#include "sclang/SclDialect/Dialect.h"

#include <ctime>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

#include "sclang/SclDialect/SclTypes.h"

using namespace mlir;
using namespace mlir::scl;


// MARK: helper functions

namespace {
void printTimeLiteral(raw_ostream &name, int t) {
  if (!t) {
    name << "0s";
    return;
  }
  int tt = t / (24 * 60 * 60 * 1000);
  if (tt) {
    name << tt << 'd';
    t = t % (24 * 60 * 60 * 1000);
    if (t)
      name << '_';
  }
  tt = t / (60 * 60 * 1000);
  if (tt) {
    name << tt << 'h';
    t = t % (60 * 60 * 1000);
    if (t)
      name << '_';
  }
  tt = t / (60 * 1000);
  if (tt) {
    name << tt << 'm';
    t = t % (60 * 1000);
    if (t)
      name << '_';
  }
  tt = t / (1000);
  if (tt) {
    name << tt << 's';
    t = t % (1000);
    if (t)
      name << '_';
  }
  if (t) {
    name << t << "ms";
  }
}

const int DAYS_FROM_EPOCH_TO_1990 = (1990-1970) * 365 + 5; // 5 leap years

int32_t getDaysSince1990(int year, int month, int day) {
  struct tm tm;
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = tm.tm_min = tm.tm_sec = 0;
  tm.tm_isdst = tm.tm_gmtoff = 0;
  time_t time = mktime(&tm) - timezone;
  assert(time >= 0);

  time_t days_from_epoch = time / (time_t)(24*3600);
  return days_from_epoch - DAYS_FROM_EPOCH_TO_1990;
}

void getDateFromDaysSince1990(int32_t days, int *year, int *month, int *day) {
  struct tm tm;
  time_t days_from_epoch = days + DAYS_FROM_EPOCH_TO_1990;
  time_t time = days_from_epoch * (24*3600);
  gmtime_r(&time, &tm);
  *year = tm.tm_year + 1900;
  *month = tm.tm_mon + 1;
  *day = tm.tm_mday;
}

uint64_t to_bcd2(int offset, int value) {
  uint64_t bcd = 0;
  bcd += (value / 10 % 10) << 4;
  bcd += (value % 10);
  bcd <<= offset;
  return bcd;
}
uint64_t to_bcd3(int offset, int value) {
  uint64_t bcd = to_bcd2(4, value / 10);
  bcd += (value % 10);
  bcd <<= offset;
  return bcd;
}

uint64_t to_datetime(int year, int month, int day,
                     int hour, int min, int sec, int msec) {
  uint64_t value = 0;
  value += to_bcd2(56, year % 100);
  value += to_bcd2(48, month);
  value += to_bcd2(40, day);
  value += to_bcd2(32, hour);
  value += to_bcd2(24, min);
  value += to_bcd2(16, sec);
  value += to_bcd3( 4, msec);
  return value;
}

int from_bcd2(int offset, uint64_t bcd) {
  bcd >>= offset;
  bcd &= 0xff;
  return (bcd >> 4) * 10 + (bcd & 0x0f);
}
int from_bcd3(int offset, uint64_t bcd) {
  bcd >>= offset;
  return from_bcd2(4, bcd) * 10 + (bcd & 0x0f);
}

void from_datetime(uint64_t value, int *year, int *month, int *day,
                   int *hour, int *min, int *sec, int *msec) {
  int y = from_bcd2(56, value);
  if (y >= 90)
    *year = 1900 + y;
  else
    *year = 2000 + y;
  *month = from_bcd2(48, value);
  *day   = from_bcd2(40, value);
  *hour  = from_bcd2(32, value);
  *min   = from_bcd2(24, value);
  *sec   = from_bcd2(16, value);
  *msec  = from_bcd3( 4, value);
}
} // namespace



//===----------------------------------------------------------------------===//
// MARK: ConstantOp
//===----------------------------------------------------------------------===//

/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.
///
OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return value(); }

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (auto intCst = value().dyn_cast<IntegerAttr>()) {
    LogicalType logTy = getType().dyn_cast<LogicalType>();

    // Sugar BOOL constants with 'true' and 'false'.
    if (logTy && logTy.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, use the value.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getInt();
    setNameFn(getResult(), specialName.str());

  } else {
    setNameFn(getResult(), "cst");
  }
}

//===----------------------------------------------------------------------===//
// MARK: ConstantS5TimeOp
//===----------------------------------------------------------------------===//
// time as BCD coded floating point
// first digit (bits 15..12): scale factor (10ms..10s)
// next three digits (bits 11..0): significant

namespace {
uint16_t getS5TimeBcdCoded(unsigned int timeMS)
{
  uint16_t scale;
  unsigned time;

  // determine best time base, round up time according to that base
  if (timeMS >= 1000000) {
    // time base 10s
    time = (timeMS+9999) / 10000;
    scale = 0x3000;
  } else if (timeMS >= 100000) {
    // time base 1s
    time = (timeMS+999) / 1000;
    scale = 0x2000;
  } else if (timeMS >= 10000) {
    // time base 100ms
    time = (timeMS+99) / 100;
    scale = 0x1000;
  } else {
    // time base 10ms
    time = (timeMS+9) / 10;
    scale = 0x0000;
  }
  if (time > 999) {
    // TBD error too large
    time = 999;
  }

  return scale + to_bcd3(0, time);
}
} // namespace


OpFoldResult ConstantS5TimeOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantS5TimeOp::build(OpBuilder &builder, OperationState &state,
                             unsigned int timeMS) {

  Type type = S5TimeType::get(builder.getContext());
  build(builder, state, type, getS5TimeBcdCoded(timeMS));
}

unsigned ConstantS5TimeOp::getTimeMS() {
  unsigned v = value();
  unsigned scale = v & 0x3000;
  unsigned t = from_bcd3(0, v);

  switch (scale) {
  case 0x0000: return 10 * t;
  case 0x1000: return 100 * t;
  case 0x2000: return 1000 * t;
  case 0x3000: return 10000 * t;
  }
  assert(false);
}

static mlir::ParseResult parseConstantS5TimeOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  uint64_t secs, msecs;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseInteger(secs) || parser.parseKeyword("s") ||
      parser.parseInteger(msecs))
    return failure();

  auto builder = parser.getBuilder();
  auto s5t = getS5TimeBcdCoded(secs * 1000 + msecs);
  auto attr = IntegerAttr::get(builder.getIntegerType(16, /*isSigned=*/false),
                               APInt(16, s5t, /*isSigned=*/false));
  result.attributes.append("value", attr);
  result.addTypes(S5TimeType::get(builder.getContext()));
  return success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantS5TimeOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  unsigned t = op.getTimeMS();
  int secs = t / 1000;
  int msecs = t % 1000;
  printer << secs << " s " << msecs;
}

void ConstantS5TimeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  name << "s5t_";
  printTimeLiteral(name, getTimeMS());

  setNameFn(getResult(), name.str());
}

//===----------------------------------------------------------------------===//
// MARK: ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantTimeOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantTimeOp::build(OpBuilder &builder, OperationState &state,
                           int timeMS) {
  Type type = TimeType::get(builder.getContext());
  build(builder, state, type, timeMS);
}

static mlir::ParseResult parseConstantTimeOp(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result) {
  uint64_t secs, msecs;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseInteger(secs) || parser.parseKeyword("s") ||
      parser.parseInteger(msecs))
    return failure();

  auto builder = parser.getBuilder();
  auto value = builder.getSI32IntegerAttr(secs * 1000 + msecs);
  result.attributes.append("value", value);
  result.addTypes(TimeType::get(builder.getContext()));
  return success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantTimeOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  int32_t t = op.value();
  int secs = t / 1000;
  int msecs = t % 1000;
  printer << secs << " s " << msecs;
}

void ConstantTimeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  name << "t_";
  printTimeLiteral(name, value());

  setNameFn(getResult(), name.str());
}


//===----------------------------------------------------------------------===//
// MARK: ConstantDateOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantDateOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantDateOp::build(OpBuilder &builder, OperationState &state,
                           int year, int month, int day) {
  Type type = DateType::get(builder.getContext());
  int days = getDaysSince1990(year, month, day);
  assert(days >= 0 && days < 65535);
  build(builder, state, type, days);
}

static mlir::ParseResult parseConstantDateOp(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result) {
  int year, month, day;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseInteger(year) ||
      parser.parseInteger(month) ||
      parser.parseInteger(day))
    return failure();

  int date = getDaysSince1990(year, month, day);
  if (date < 0 || date > 65535)
    return failure();
  auto builder = parser.getBuilder();
  auto attr = IntegerAttr::get(builder.getIntegerType(16, /*isSigned=*/false),
                               APInt(16, date, /*isSigned=*/false));
  if (!attr)
    return failure();
  result.attributes.append("value", attr);
  result.addTypes(DateType::get(builder.getContext()));
  return success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantDateOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  int year, month, day;
  getDateFromDaysSince1990(op.value(), &year, &month, &day);
  printer << year << " " << month << " " << day;
}

void ConstantDateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);

  int year, month, day;
  getDateFromDaysSince1990(value(), &year, &month, &day);
  name << "d_" << year << '_' << month << '_' << day;

  setNameFn(getResult(), name.str());
}


//===----------------------------------------------------------------------===//
// MARK: ConstantTimeOfDayOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantTimeOfDayOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantTimeOfDayOp::build(OpBuilder &builder, OperationState &state,
                                uint32_t timeMS) {
  Type type = TimeOfDayType::get(builder.getContext());
  build(builder, state, type, timeMS);
}

static mlir::ParseResult parseConstantTimeOfDayOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  uint64_t secs, msecs;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseInteger(secs) || parser.parseKeyword("s") ||
      parser.parseInteger(msecs))
    return failure();

  auto builder = parser.getBuilder();
  auto value = builder.getUI32IntegerAttr(secs * 1000 + msecs);
  result.attributes.append("value", value);
  result.addTypes(TimeOfDayType::get(builder.getContext()));
  return success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantTimeOfDayOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  int32_t t = op.value();
  int secs = t / 1000;
  int msecs = t % 1000;
  printer << secs << " s " << msecs;
}

void ConstantTimeOfDayOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  name << "tod_";
  printTimeLiteral(name, value());

  setNameFn(getResult(), name.str());
}


//===----------------------------------------------------------------------===//
// MARK: ConstantDateAndTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantDateAndTimeOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantDateAndTimeOp::build(OpBuilder &builder, OperationState &state,
          int year, int month, int day, int hour, int min, int sec, int msec) {
  Type type = DateAndTimeType::get(builder.getContext());
  uint64_t value = to_datetime(year, month, day, hour, min, sec, msec);
  build(builder, state, type, value);
}

static mlir::ParseResult parseConstantDateAndTimeOp(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  int year, month, day, hour, min, sec, msec;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseInteger(year) || parser.parseColon() ||
      parser.parseInteger(month) || parser.parseColon() ||
      parser.parseInteger(day) ||
      parser.parseInteger(hour) || parser.parseColon() ||
      parser.parseInteger(min) || parser.parseColon() ||
      parser.parseInteger(sec) ||
      parser.parseInteger(msec))
     return failure();
  uint64_t bcd = to_datetime(year, month, day, hour, min, sec, msec);
  auto builder = parser.getBuilder();
  auto attr = IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/false),
                               APInt(64, bcd, /*isSigned=*/false));
  result.attributes.append("value", attr);
  result.addTypes(DateAndTimeType::get(builder.getContext()));
  return success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantDateAndTimeOp op) {
  printer << " ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  int year, month, day, hour, min, sec, msec;
  from_datetime(op.value(), &year, &month, &day, &hour, &min, &sec, &msec);
  printer << year << ':' << month << ':' << day << ' ';
  printer << hour << ':' << min << ':' << sec << ' ' << msec;
}

void ConstantDateAndTimeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  int year, month, day, hour, min, sec, msec;
  from_datetime(value(), &year, &month, &day, &hour, &min, &sec, &msec);
  name << "dt_" << year << '-' << month << '-' << day << '_';
  name << hour << '-' << min << '-' << sec << '_' << msec;

  setNameFn(getResult(), name.str());
}


//===----------------------------------------------------------------------===//
// MARK: BitCastOp
//===----------------------------------------------------------------------===//

namespace {
int bitWidth(Type type) {
  TypeSwitch<Type>(type)
  .Case<scl::IntegerType>([&](scl::IntegerType type) {
    return type.getWidth();
  })
  .Case<scl::LogicalType>([&](scl::LogicalType type) {
    return type.getWidth();
  })
  .Case<scl::RealType>([&](scl::RealType type) {
    return 32;
  });

  return -1;
}
}

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool BitCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  int inputWidth = bitWidth(inputs.front());
  int outputWidth = bitWidth(outputs.front());
  return inputWidth > 0 && inputWidth == outputWidth;
}


// MARK: DialectCastOp

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool DialectCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  bool inputScl = isa<SclDialect>(inputs.front().getDialect());
  bool outputScl = isa<SclDialect>(outputs.front().getDialect());
  // TBD check dialects
  return inputScl != outputScl;
}


//===----------------------------------------------------------------------===//
// MARK: FunctionOp
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionOp(OpAsmParser &parser, OperationState &state) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, function_like_impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(parser, state, /*allowVariadic=*/false,
                                         buildFuncType);
}

static void print(OpAsmPrinter &printer, FunctionOp fnOp) {
  FunctionType fnType = fnOp.getType();
  function_like_impl::printFunctionLikeOp(printer, fnOp, fnType.getInputs(),
                                  /*isVariadic=*/false, fnType.getResults());
}

LogicalResult FunctionOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  if (getType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult FunctionOp::verifyBody() {
  FunctionType fnType = getType();
  auto walkResult = walk([fnType](Operation *op) -> WalkResult {
    if (auto retOp = dyn_cast<ReturnOp>(op)) {
      if (fnType.getNumResults() != 0)
        return retOp.emitOpError("cannot be used in functions returning value");
    } else if (auto retOp = dyn_cast<ReturnValueOp>(op)) {
      if (fnType.getNumResults() != 1)
        return retOp.emitOpError(
                   "returns 1 value but enclosing function requires ")
               << fnType.getNumResults() << " results";

      auto returnType = retOp.getReturnType();
      auto fnResultType = fnType.getResult(0);
      if (returnType != fnResultType)
        return retOp.emitOpError(" return value's type (")
               << returnType << ") mismatch with function's result type ("
               << fnResultType << ")";
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.addRegion();
}

// CallableOpInterface
Region *FunctionOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface
ArrayRef<Type> FunctionOp::getCallableResults() {
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// MARK: FunctionBlockOp
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionBlockOp(OpAsmParser &parser, OperationState &state) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, function_like_impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(parser, state, /*allowVariadic=*/false,
                                         buildFuncType);
}

static void print(OpAsmPrinter &printer, FunctionBlockOp fnOp) {
  FunctionType fnType = fnOp.getType();
  function_like_impl::printFunctionLikeOp(printer, fnOp, fnType.getInputs(),
                                  /*isVariadic=*/false, fnType.getResults());
}

LogicalResult FunctionBlockOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  if (getType().getNumInputs() != 1)
    return emitOpError("must have one input");
  if (getType().getNumResults() != 0)
    return emitOpError("cannot have results");
  return success();
}

LogicalResult FunctionBlockOp::verifyBody() {
  FunctionType fnType = getType();
  auto walkResult = walk([fnType](Operation *op) -> WalkResult {
    if (auto retOp = dyn_cast<ReturnValueOp>(op)) {
      return retOp.emitOpError(
                "returns 1 value but enclosing function requires ")
             << fnType.getNumResults() << " results";
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

void FunctionBlockOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name) {
  Type idb = InstanceDbType::get(builder.getContext(), name);
  Type selfType = AddressType::get(idb);
  SmallVector<Type, 1> inputs = { selfType };
  SmallVector<Type, 0> results = {};
  FunctionType func_type = builder.getFunctionType(inputs, results);

  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(func_type));

  state.addRegion();
}

// CallableOpInterface
Region *FunctionBlockOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface
ArrayRef<Type> FunctionBlockOp::getCallableResults() {
  return getType().getResults();
}


// MARK: CallFbOp

CallInterfaceCallable CallFbOp::getCallableForCallee() {
  return getOperation()->getAttrOfType<SymbolRefAttr>(fb());
}

Operation::operand_range CallFbOp::getArgOperands() {
  return {getOperation()->operand_begin(), getOperation()->operand_end()}; // only operand is `idb`
}


// MARK: CallFcOp

CallInterfaceCallable CallFcOp::getCallableForCallee() {
  return getOperation()->getAttrOfType<SymbolRefAttr>(callee());
}

Operation::operand_range CallFcOp::getArgOperands() { return arguments(); }


// MARK: GetElementOp

void GetElementOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), name());
}

// MARK: GetGlobalOp

void GetGlobalOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), name());
}

// MARK: GetVariableOp

void GetVariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), sym_name());
}

// MARK: IntegerCastOp

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool IntegerCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // must be a cast between two integer types
  IntegerType input = inputs.front().dyn_cast<IntegerType>();
  IntegerType output = outputs.front().dyn_cast<IntegerType>();
  if (!input || !output)
    return false;
  return true;

}

// MARK: TempVarOp

void TempVariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), name());
}


// MARK: VariableOp

void VariableOp::build(OpBuilder &builder, OperationState &state,
                       Type type, bool isInput, bool isOutput, StringRef name, Value init) {
  auto typeAttr = mlir::TypeAttr::get(type);
  UnitAttr inAttr, outAttr;
  if (isInput)
    inAttr = builder.getUnitAttr();
  if (isOutput)
    outAttr = builder.getUnitAttr();
  auto nameAttr = builder.getStringAttr(name);

  build(builder, state, typeAttr, inAttr, outAttr, nameAttr, init);
}
void VariableOp::build(OpBuilder &builder, OperationState &state,
                       Type type, bool isInput, bool isOutput, StringRef name) {
  build(builder, state, type, isInput, isOutput, name, nullptr);
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sclang/SclDialect/SclOps.cpp.inc"

void SclDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "sclang/SclDialect/SclOps.cpp.inc"
  >();
}
