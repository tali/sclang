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
#include "sclang/SclGen/MLIRGen.h"
#include "sclang/SclGen/Parser.h"
#include "sclang/AwlGen/AwlGen.h"
#include "sclang/Runtime/Runtime.h"
#include "sclang/SclTransforms/Passes.h"
#include <memory>

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace sclang;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input SCL file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Scl, MLIR, Awl };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(Scl), cl::desc("Select input file"),
    cl::values(clEnumValN(Scl, "scl", "load the input file as an SCL source")),
    cl::values(clEnumValN(Awl, "awl", "load the input file as an AWL (STL) source")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRStd,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRStd, "mlir-std",
                          "output the MLIR dump after std lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit", "run the code")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Size of a simulated memory area in bytes, e.g. `--area-size M=1024` or
/// `--area-size DB1=4096` (areas I/Q/M/T/C; DBs grow on demand if unset).
static cl::list<std::string> areaSizes(
    "area-size",
    cl::desc("Set the size in bytes of a simulated memory area, e.g. "
             "--area-size M=1024 or --area-size DB1=4096"),
    cl::value_desc("AREA=SIZE"));

namespace {
enum RuntimeKind { Sim, Host };
}
/// Which runtime the JIT binds: `sim` (default, built in) or `host` (a
/// shared library given with --runtime-lib, exporting
/// scl_host_runtime_vtable()).
static cl::opt<enum RuntimeKind> runtimeKind(
    "runtime", cl::init(Sim),
    cl::desc("Runtime bound to the JIT's @scl_rt_* symbols"),
    cl::values(clEnumValN(Sim, "sim", "built-in simulation runtime")),
    cl::values(clEnumValN(Host, "host",
                          "host runtime from --runtime-lib (ADR 002)")));
/// Shared library exporting scl_host_runtime_vtable() for --runtime=host.
static cl::opt<std::string> runtimeLib(
    "runtime-lib", cl::desc("Shared library with a host runtime vtable "
                            "(used with --runtime=host)"),
    cl::value_desc("path.so"));

/// After a JIT run with the sim runtime, dump all non-zero memory-area bytes
/// and the virtual clock to stdout (makes %I/%Q/%M/%DB writes observable in
/// tests, e.g. for SCL-vs-AWL cross-checks).
static cl::opt<bool> dumpAreas(
    "dump-areas", cl::init(false),
    cl::desc("After --emit=jit, dump the sim runtime's memory areas and "
             "clock to stdout (only with --runtime=sim)"));

/// Number of OB1 scans the JIT runs (models repeated PLC scan cycles), and
/// the virtual milliseconds advanced between scans via scl_rt_cycle()
/// (drives timer expiry in tests).
static cl::opt<unsigned> jitCycles(
    "jit-cycles", cl::init(1),
    cl::desc("Number of OB1 scans to run with --emit=jit"));
static cl::opt<uint64_t> cycleMs(
    "cycle-ms", cl::init(100),
    cl::desc("Virtual milliseconds advanced between JIT scans (scl_rt_cycle)"));

/// Returns a SCL AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<sclang::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code EC = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return nullptr;
  }
  auto buffer = FileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.ParseModule();
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.scl' / '.awl' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    bool isAwl = inputType == InputType::Awl ||
                 llvm::StringRef(inputFilename).endswith(".awl");
    if (isAwl) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
      if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return 6;
      }
      module = sclang::awlGen(context, fileOrErr.get()->getBuffer(),
                              std::string(inputFilename));
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

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToStd = emitAction >= Action::DumpMLIRStd;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToStd) {
    mlir::OpPassManager &optFC = pm.nest<mlir::scl::FunctionOp>();
    optFC.addPass(mlir::createCanonicalizerPass());
    optFC.addPass(mlir::createCSEPass());
    mlir::OpPassManager &optFB = pm.nest<mlir::scl::FunctionBlockOp>();
    optFB.addPass(mlir::createCanonicalizerPass());
    optFB.addPass(mlir::createCSEPass());
 }

  if (isLoweringToStd) {
    // Partially lower the SCL dialect with a few cleanups afterwards.
    pm.addPass(mlir::sclang::createLowerToStdPass());

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::sclang::createLowerToLLVMPass());
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

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

/// Bind the runtime C ABI (ADR 002) to the JIT: by default the simulation
/// runtime linked into this binary; a host process can redirect the calls by
/// registering an SclRtVTable via scl_rt_register().
static void bindRuntimeSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([](llvm::orc::MangleAndInterner mangle) {
    llvm::orc::SymbolMap map;
    auto add = [&](const char *name, void *addr) {
      map[mangle(name)] = llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(addr),
          llvm::JITSymbolFlags::Exported);
    };
    add("scl_rt_read", (void *)&scl_rt_read);
    add("scl_rt_write", (void *)&scl_rt_write);
    add("scl_rt_read_bit", (void *)&scl_rt_read_bit);
    add("scl_rt_write_bit", (void *)&scl_rt_write_bit);
    add("scl_rt_update_inputs", (void *)&scl_rt_update_inputs);
    add("scl_rt_update_outputs", (void *)&scl_rt_update_outputs);
    add("scl_rt_cycle", (void *)&scl_rt_cycle);
    add("scl_rt_timer_start", (void *)&scl_rt_timer_start);
    add("scl_rt_counter_op", (void *)&scl_rt_counter_op);
    add("scl_rt_timer_fb", (void *)&scl_rt_timer_fb);
    // memref.copy lowers to a call to the MLIR C runner helper.
    add("memrefCopy", (void *)&memrefCopy);
    return map;
  });
}

/// Apply the `--area-size AREA=SIZE` options to the simulation runtime.
static void configureSimRuntime() {
  scl_rt_sim_reset();
  for (const auto &spec : areaSizes) {
    auto eq = spec.find('=');
    if (eq == std::string::npos || eq == 0) {
      llvm::errs() << "invalid --area-size '" << spec << "', expected AREA=SIZE\n";
      continue;
    }
    uint64_t size = std::strtoull(spec.c_str() + eq + 1, nullptr, 0);
    llvm::StringRef area(spec.data(), eq);
    std::string upperStr = area.upper();
    llvm::StringRef upper(upperStr);
    if (upper.startswith("DB")) {
      // DB<n>=size: pre-size a simulated data block
      uint32_t db = (uint32_t)std::strtoul(upper.data() + 2, nullptr, 0);
      scl_rt_sim_set_db_size(db, size);
      continue;
    }
    auto areaCode = mlir::scl::DirectType::areaFromName(upper);
    if (!areaCode) {
      llvm::errs() << "invalid --area-size area '" << spec << "'\n";
      continue;
    }
    scl_rt_sim_set_area_size((uint8_t)*areaCode, size);
  }
}

/// Bind the runtime selected by `--runtime`/`--runtime-lib` (ADR 002):
/// `sim` clears any registered vtable, `host` loads the vtable exported by
/// the given shared library and registers it.
static bool configureRuntimeBinding() {
  if (runtimeKind == Sim) {
    scl_rt_register(nullptr);
    return true;
  }
  if (runtimeLib.empty()) {
    llvm::errs() << "--runtime=host requires --runtime-lib=<path.so>\n";
    return false;
  }
  std::string err;
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(runtimeLib.c_str(),
                                                            &err);
  if (!lib.isValid()) {
    llvm::errs() << "cannot load runtime lib '" << runtimeLib << "': " << err
                 << "\n";
    return false;
  }
  auto sym = lib.getAddressOfSymbol("scl_host_runtime_vtable");
  if (!sym) {
    llvm::errs() << "runtime lib '" << runtimeLib
                 << "' does not export scl_host_runtime_vtable\n";
    return false;
  }
  auto getVtable = (const SclRtVTable *(*)())sym;
  const SclRtVTable *vt = getVtable();
  if (!vt) {
    llvm::errs() << "runtime lib '" << runtimeLib
                 << "' returned a null vtable\n";
    return false;
  }
  scl_rt_register(vt);
  return true;
}

int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Set the target triple of the module to the host triple.
  module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                  mlir::StringAttr::get(module->getContext(),
                                        llvm::sys::getProcessTriple()));


  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Bind the runtime C ABI (ADR 002): select sim/host runtime, apply
  // --area-size options, and bind the JIT symbols to the runtime functions.
  configureSimRuntime();
  if (!configureRuntimeBinding())
    return -1;
  bindRuntimeSymbols(*engine);

  // Invoke the JIT-compiled function once per scan cycle, advancing the
  // virtual clock between scans (drives %T timers / %C counters).
  for (unsigned i = 0; i < jitCycles; ++i) {
    auto invocationResult = engine->invoke("OB1");
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n" << invocationResult << "\n";
      return -1;
    }
    if (i + 1 < jitCycles)
      scl_rt_cycle(cycleMs);
  }

  if (dumpAreas && runtimeKind == Sim)
    scl_rt_sim_dump_areas();

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "SCL compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::scl::SclDialect>();
  context.getOrLoadDialect<mlir::awl::AwlDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  // Load other target dialects
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->print(llvm::outs());
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
