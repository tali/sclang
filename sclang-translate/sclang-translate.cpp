//===- sclang-translate.cpp - MLIR translation driver w/ scl/awl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `sclang-translate` translates MLIR files that use the scl/awl dialects:
//
//   sclang-translate module.mlir                  # print MLIR (round-trip)
//   sclang-translate --serialize module.mlir -o module.mlirbc
//   sclang-translate module.mlirbc                # bytecode -> MLIR text
//   sclang-translate --mlir-to-llvmir module.mlir # MLIR -> LLVM IR
//
// Input format (text vs. bytecode) is auto-detected by the MLIR parser.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "sclang/AwlDialect/Dialect.h"
#include "sclang/SclDialect/Dialect.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));
static llvm::cl::opt<bool> mlirToLLVMIR(
    "mlir-to-llvmir", llvm::cl::desc("Translate MLIR to LLVM IR"),
    llvm::cl::init(false));
static llvm::cl::opt<bool> serialize(
    "serialize", llvm::cl::desc("Serialize the module to MLIR bytecode"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "sclang MLIR translation tool (scl/awl dialects)\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<scl::SclDialect>();
  registry.insert<awl::AwlDialect>();

  MLIRContext context(registry);
  context.printOpOnDiagnostic(true);

  // Load the input (text or bytecode — auto-detected).
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "cannot open input: " << ec.message() << "\n";
    return 1;
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  std::error_code ec;
  llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "cannot open output: " << ec.message() << "\n";
    return 1;
  }

  if (serialize) {
    if (failed(writeBytecodeToFile(*module, out.os())))
      return 1;
  } else if (mlirToLLVMIR) {
    registerBuiltinDialectTranslation(context);
    registerLLVMDialectTranslation(context);
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule)
      return 1;
    llvmModule->print(out.os(), nullptr);
  } else {
    module->print(out.os());
  }
  out.keep();
  return 0;
}
