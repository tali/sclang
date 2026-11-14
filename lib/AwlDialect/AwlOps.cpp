//===- AwlOps.cpp - AWL dialect op implementations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sclang/AwlDialect/AwlOps.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::awl;

// MARK: BlockOp

LogicalResult BlockOp::verify() {
  if (getBody().empty())
    return emitOpError("must have a body");
  if (getBody().getBlocks().size() != 1)
    return emitOpError("must have exactly one flat block");
  Block &b = getBody().front();
  if (b.empty() || !isa<BeOp>(b.back()))
    return emitOpError("body block must end with awl.be");
  return success();
}

// MARK: DirectOp

void DirectOp::build(OpBuilder &builder, OperationState &state,
                     mlir::scl::DirectArea area, int32_t db, int32_t byte,
                     int32_t bit, int32_t width) {
  auto directTy = mlir::scl::DirectType::get(builder.getContext(), area, db,
                                             byte, bit, width);
  build(builder, state, directTy,
        builder.getStringAttr(mlir::scl::DirectType::areaName(area)),
        builder.getI32IntegerAttr(db), builder.getI32IntegerAttr(byte),
        builder.getI32IntegerAttr(bit), builder.getI32IntegerAttr(width));
}

LogicalResult DirectOp::verify() {
  auto dt = getResult().getType().dyn_cast<mlir::scl::DirectType>();
  if (!dt)
    return emitOpError("result must be !scl.direct");
  auto area = mlir::scl::DirectType::areaFromName(getArea());
  if (!area || *area != dt.getArea() || getDb() != dt.getDb() ||
      getByte() != dt.getByte() || getBit() != dt.getBit() ||
      getWidth() != dt.getWidth())
    return emitOpError("attributes do not match the result !scl.direct type");
  if (dt.getWidth() != 1 && dt.getWidth() != 8 && dt.getWidth() != 16 &&
      dt.getWidth() != 32)
    return emitOpError("width must be 1, 8, 16 or 32 bits");
  if (dt.getBit() < 0 || dt.getBit() > 7)
    return emitOpError("bit must be in 0..7");
  return success();
}

ParseResult DirectOp::parse(OpAsmParser &parser, OperationState &state) {
  // awl.direct M, 0, 10, 3, 1 : !scl.direct<...>
  StringRef area;
  int32_t db, byte, bit, width;
  Type type;
  if (parser.parseKeyword(&area) || parser.parseComma() ||
      parser.parseInteger(db) || parser.parseComma() ||
      parser.parseInteger(byte) || parser.parseComma() ||
      parser.parseInteger(bit) || parser.parseComma() ||
      parser.parseInteger(width) || parser.parseColon() ||
      parser.parseType(type))
    return failure();
  auto &builder = parser.getBuilder();
  state.addAttribute("area", builder.getStringAttr(area));
  state.addAttribute("db", builder.getI32IntegerAttr(db));
  state.addAttribute("byte", builder.getI32IntegerAttr(byte));
  state.addAttribute("bit", builder.getI32IntegerAttr(bit));
  state.addAttribute("width", builder.getI32IntegerAttr(width));
  state.addTypes(type);
  return success();
}

void DirectOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getArea() << ", " << getDb() << ", " << getByte() << ", "
          << getBit() << ", " << getWidth() << " : " << getResult().getType();
}

// MARK: LoadOp

void LoadOp::build(OpBuilder &builder, OperationState &state, Value addr) {
  auto dt = addr.getType().cast<mlir::scl::DirectType>();
  Type resTy = dt.getWidth() == 1
                   ? (Type)builder.getI1Type()
                   : (Type)builder.getIntegerType(dt.getWidth());
  build(builder, state, resTy, addr);
}

// MARK: CallOp (CallOpInterface)

void CallOp::setCalleeFromCallable(CallInterfaceCallable callable) {
  (*this)->setAttr(getCalleeAttrName(), callable.get<SymbolRefAttr>());
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  return getOperation()->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
}

Operation::operand_range CallOp::getArgOperands() {
  // no variadic call arguments yet (FC parameters unsupported); the optional
  // rlo operand of CC is not a call argument
  return Operation::operand_range(getOperation()->operand_begin(),
                                  getOperation()->operand_begin());
}

// MARK: TableGen'd op method definitions

#define GET_OP_CLASSES
#include "sclang/AwlDialect/AwlOps.cpp.inc"

void AwlDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "sclang/AwlDialect/AwlOps.cpp.inc"
  >();
}
