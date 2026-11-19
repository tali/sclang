//===- sclang.cpp - The SCL Compiler
//----------------------------------------===//
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
// This file implements the entry point for the SCL compiler.
//
//===----------------------------------------------------------------------===//

#include "sclang/SclDialect/Dialect.h"
#include "sclang/AwlDialect/Dialect.h"
#include "sclang/AwlDialect/Printer.h"
#include "sclang/SclGen/MLIRGen.h"
#include "sclang/SclGen/Parser.h"
#include "sclang/AwlGen/AwlGen.h"
#include "sclang/FrontendOptions.h"
#include "sclang/SclTransforms/Passes.h"
#include <memory>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace sclang;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input SCL file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Scl, MLIR, Awl, St };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(Scl), cl::desc("Select input file"),
    cl::values(clEnumValN(Scl, "scl", "load the input file as an SCL source")),
    cl::values(clEnumValN(Awl, "awl", "load the input file as an AWL (STL) source")),
    cl::values(clEnumValN(St, "st",
                          "load the input file as IEC 61131-3 ST "
                          "(alias for -x=scl --std=iec)")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

// Source-language mode of the SCL/ST frontend (ADR 004): S7 = Siemens SCL
// dialect (default), IEC = IEC 61131-3 ST surface. `-x=st` forces IEC.
static cl::opt<sclang::LangMode> stdMode(
    "std", cl::init(sclang::LangMode::S7),
    cl::desc("Source-language mode of the SCL frontend (ADR 004)"),
    cl::values(clEnumValN(sclang::LangMode::S7, "s7",
                          "Siemens SCL dialect (default): % directs, "
                          "TIMER/COUNTER, EN/ENO, STRING[n]")),
    cl::values(clEnumValN(sclang::LangMode::IEC, "iec",
                          "IEC 61131-3 ST: CONFIGURATION/VAR_GLOBAL/"
                          "VAR_EXTERNAL, LREAL, bare STRING")));

static sclang::LangMode langMode() {
  return inputType == InputType::St ? sclang::LangMode::IEC : stdMode;
}

// Test-builtin gate (SCL/ST: __ASSERT, __DEBUG_PRINT, ...; AWL: __ASSERT).
static cl::opt<bool> testBuiltinsOpt(
    "test-builtins",
    cl::desc("Enable test builtins: __ASSERT, __DEBUG_PRINT, __BITS, "
             "__UNBITS, __STRTO_M, __LOADM64, __STOREM64 (SCL/ST) and "
             "__ASSERT (AWL)"),
    cl::init(false));

namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRStd, DumpAWL, DumpBytecode };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRStd, "mlir-std",
                          "output the MLIR dump after std lowering")),
    cl::values(clEnumValN(DumpAWL, "awl",
                          "output AWL (STL) text after the LowerToAWL pass")),
    cl::values(clEnumValN(DumpBytecode, "bytecode",
                          "output MLIR bytecode (faster to re-parse than text)")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
// Verify the IR after every pass (default on; disable with
// --verify-each=0 to speed up large compilations, see AGENTS.md).
static cl::opt<bool> verifyEach("verify-each",
    cl::desc("Verify the IR after each pass"), cl::init(true));

/// Returns a SCL AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<sclang::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code EC = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return nullptr;
  }
  auto buffer = FileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename),
                    langMode());
  Parser parser(lexer);
  return parser.ParseModule();
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.scl' / '.awl' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir") &&
      !llvm::StringRef(inputFilename).ends_with(".mlirbc")) {
    bool isAwl = inputType == InputType::Awl ||
                 llvm::StringRef(inputFilename).ends_with(".awl");
    if (isAwl) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
      if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return 6;
      }
      module = sclang::awlGen(context, fileOrErr.get()->getBuffer(),
                              std::string(inputFilename),
                              sclang::g_testBuiltins);
      return !module ? 1 : 0;
    }
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  // Dev-only pass timing: --mlir-timing (see AGENTS.md).
  applyDefaultTimingPassManagerCLOptions(pm);
  pm.enableVerifier(verifyEach);

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToStd = emitAction == Action::DumpMLIRStd;

  if (enableOpt || isLoweringToStd) {
    mlir::OpPassManager &optFC = pm.nest<mlir::scl::FunctionOp>();
    optFC.addPass(mlir::createCanonicalizerPass());
    optFC.addPass(mlir::createCSEPass());
    mlir::OpPassManager &optFB = pm.nest<mlir::scl::FunctionBlockOp>();
    optFB.addPass(mlir::createCanonicalizerPass());
    optFB.addPass(mlir::createCSEPass());
 }

  if (emitAction == Action::DumpAWL) {
    // SCL -> AWL backend (docs/design/scl-to-awl.md): scl dialect -> awl
    // dialect; printed as S7 text by awl::printAwlText in main().
    pm.addPass(mlir::sclang::createLowerToAWLPass());
  }

  if (isLoweringToStd) {
    // Partially lower the SCL dialect with a few cleanups afterwards.
    pm.addPass(mlir::sclang::createLowerToStdPass());

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      // SCCP folds the scan-cycle/timer scaffolding constants; mem2reg
      // promotes the scalar-local memref allocas to SSA registers; LICM
      // hoists loop-invariant ops (Pure ops are speculatable in MLIR 23).
      optPM.addPass(mlir::createSCCPPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(mlir::createMem2Reg());
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
      // Drop functions/blocks that became dead during optimization.
      pm.addPass(mlir::createSymbolDCEPass());
    }
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "SCL compiler\n");
  sclang::g_testBuiltins = testBuiltinsOpt;

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  // MLIR only prints errors to stderr by default; register a handler so
  // warnings and frontend errors are shown too (otherwise mlirGen failures
  // exit silently with rc=1).
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    llvm::errs() << "\n";
    return mlir::success();
  });
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::scl::SclDialect>();
  context.getOrLoadDialect<mlir::awl::AwlDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  if (emitAction == Action::DumpAWL) {
    mlir::awl::printAwlText(*module, llvm::outs());
    return 0;
  }

  if (emitAction == Action::DumpBytecode) {
    if (mlir::failed(mlir::writeBytecodeToFile(*module, llvm::outs())))
      return 1;
    return 0;
  }

  module->print(llvm::outs());
  return 0;
}
