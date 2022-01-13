//====- LowerToLLVM.cpp - Lowering from Std+Loop to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"
#include "sclang/SclTransforms/Passes.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

// MARK: DebugPrintOpLowering

/// Lowers `scl.debug.print` to a call to `printf`
struct DebugPrintOpLowering : public OpConversionPattern<scl::DebugPrintOp> {
  DebugPrintOpLowering(MLIRContext *context, SymbolTable &symbolTable)
      : OpConversionPattern<scl::DebugPrintOp>::OpConversionPattern(context),
        symbolTable(symbolTable) {}

  LogicalResult
  matchAndRewrite(scl::DebugPrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Type printfRetType = rewriter.getIntegerType(32);

    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "_debug_print_fmt", StringRef("%s\n\0", 4), parentModule,
        /*reuse=*/true);

    std::string msg = std::string(op.msg());
    // Append `\0` to follow C style string given that
    // LLVM::createGlobalString() won't handle this directly for us.
    msg.push_back('\0');
    Value msgCst = getOrCreateGlobalString(loc, rewriter, "_debug_msg", msg,
                                           parentModule, /*reuse=*/false);

    // Generate call to `printf`.
    SmallVector<Value, 2> printfArgs = {{formatSpecifierCst, msgCst}};
    rewriter.create<CallOp>(loc, printfRef, printfRetType, printfArgs);

    rewriter.eraseOp(op);
    return success();
  }

private:
  SymbolTable &symbolTable;

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                StringRef name, StringRef value,
                                ModuleOp module, bool reuse) const {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global = nullptr;
    if (reuse)
      global = symbolTable.lookup<LLVM::GlobalOp>(name);
    if (!global) {
      // Create a builder without an insertion point.
      // We will insert using the symbol table to guarantee unique names.
      auto context = builder.getContext();
      OpBuilder globalBuilder(context);

      auto type =
          LLVM::LLVMArrayType::get(IntegerType::get(context, 8), value.size());
      global = globalBuilder.create<LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
          builder.getStringAttr(value),
          /*alignment=*/0);
      symbolTable.insert(global);
      // The symbol table inserts at the end of the module, but globals are a
      // bit nicer if they are at the beginning.
      global->moveBefore(&module.front());
    }
    assert(global != nullptr && "could not find or create GlobalOp");

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

// MARK: SclToLLVMLoweringPass

namespace {
struct SclToLLVMLoweringPass
    : public PassWrapper<SclToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void SclToLLVMLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(context);

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `scl`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(context);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  SymbolTable symbolTable(module);

  // The only remaining operation to lower is the DebugPrintOp.
  patterns.add<DebugPrintOpLowering>(&getContext(), symbolTable);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Scl` operations, as
/// well as `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::sclang::createLowerToLLVMPass() {
  return std::make_unique<SclToLLVMLoweringPass>();
}
