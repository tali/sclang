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

#include "ProjectCache.h"
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Affine/Transforms/Passes.h"
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
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Target/TargetMachine.h"
#include <atomic>
#include <chrono>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace sclang;
namespace cl = llvm::cl;

static cl::list<std::string> inputFilenames(
    cl::Positional, cl::desc("<input source file(s)> — multiple files are "
                             "compiled as one project (one namespace)"),
    cl::value_desc("filename"), cl::ZeroOrMore);
/// Compile all source files of a project directory as one project (see
/// docs/design/multi-file-incremental.md): sorted *.scl (or *.awl with
/// -x=awl, *.st with -x=st) in the directory.
static cl::opt<std::string> projectDir(
    "project", cl::desc("Compile all source files of this directory as one "
                        "project (sorted, by extension of -x)"),
    cl::init(""), cl::value_desc("dir"));
/// Per-file build cache for SCL projects: each input file's scl MLIR module
/// is stored under SHA-256(content + compiler ABI tag + language mode) and
/// reused while the file is unchanged (Bazel action-cache / ccache analog;
/// docs/design/multi-file-incremental.md phase 2).
static cl::opt<std::string> cacheDir(
    "cache-dir",
    cl::desc("Cache per-file MLIR modules of the input project in this "
             "directory and reuse them while files are unchanged"),
    cl::init(""), cl::value_desc("dir"));
static cl::opt<bool> cacheStats(
    "cache-stats", cl::desc("Print per-file cache hit/miss counts to stderr"),
    cl::init(false));

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

/// Error-tolerant parsing (Parser::tolerantErrors): report all parse errors
/// with panic-mode recovery at unit/statement/declaration boundaries instead
/// of aborting at the first one. On in LSP mode and with --parse-tolerant.
static cl::opt<bool> parseTolerantFlag(
    "parse-tolerant",
    cl::desc("Error-tolerant parsing: collect multiple parse errors per file "
             "(recover at synchronization points) instead of stopping at the "
             "first"),
    cl::init(false));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpAWLMLIR,
  DumpMLIRStd,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT,
  RunJITAwl,
  EmitObj
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpAWLMLIR, "awl-mlir",
                          "output the MLIR dump after lowering to the AWL "
                          "dialect (no AWL text round-trip)")),
    cl::values(clEnumValN(DumpMLIRStd, "mlir-std",
                          "output the MLIR dump after std lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit", "run the code")),
    cl::values(clEnumValN(RunJITAwl, "jit-awl",
                          "run the code via in-memory lowering to the AWL "
                          "dialect (skips the AWL text print/re-parse "
                          "round-trip used by --emit=awl + -x=awl)")),
    cl::values(clEnumValN(EmitObj, "obj",
                          "compile to a host object file (AOT; the scl_rt_* "
                          "runtime symbols stay undefined and are resolved "
                          "when linking against a host runtime, ADR 002)")));

// The Action enum above is ordered by pipeline depth (None < DumpAST < ... <
// RunJITAwl). These predicates are the ONLY places that rely on that order;
// keep them updated when adding a new action so a reordering/insertion can
// never silently change what a comparison means.
static bool actionLowersToStd() { return emitAction >= Action::DumpMLIRStd; }
static bool actionLowersToLLVM() { return emitAction >= Action::DumpMLIRLLVM; }
static bool actionOutputsMLIR() { return emitAction <= Action::DumpMLIRLLVM; }

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename ('-' for stdout)"),
    cl::value_desc("filename"), cl::init("-"));
// mlir-opt-style pipeline override: when given, the hardcoded pipeline in
// processModule() is skipped and this pipeline runs instead (our passes are
// registered, see Passes.td / registerSclangTransformsPasses).
static cl::opt<std::string> passPipelineOpt(
    "pass-pipeline",
    cl::desc("Run this MLIR pass pipeline instead of the built-in one, e.g. "
             "'builtin.module(scl-lower-to-std,scl-lower-to-llvm)'"),
    cl::value_desc("pipeline"));
// Run as a minimal LSP server (JSON-RPC over stdio) instead of a batch
// compile: see sclang/Lsp.cpp and AGENTS.md "LSP mode".
static cl::opt<bool> lspMode("lsp", cl::desc("Run as an LSP server on stdio"),
                             cl::init(false));
// Verify the IR after every pass (default on; disable with
// --verify-each=0 to speed up large compilations, see AGENTS.md).
static cl::opt<bool> verifyEach("verify-each",
    cl::desc("Verify the IR after each pass"), cl::init(true));

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
static cl::list<std::string> dbSyms(
    "db-sym",
    cl::desc("Register a named DB from an external symbol table: NAME=DBNUM "
             "(awlsim .awlpro symbol table), e.g. --db-sym Data=42"),
    cl::value_desc("NAME=DBNUM"), cl::CommaSeparated);

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

/// Per-scan I/O channel for trace-driven JIT tests (IEC 61131 conformance
/// traces, docs/conformance.md): `--scan-io=role:name=AREA/byte[.bit]/width`
/// declares a named slot in a sim memory area (repeat the flag per slot);
/// role is `in` or `out`. With `--scan-inputs=file`, before OB1 scan N the
/// driver writes the values from line N of the file (`N name=value ...`)
/// into the `in` slots, and after each scan it prints
/// `scan N: name=value ...` for the `out` slots.
/// Example: --scan-io=in:x=M/0/2 --scan-io=out:q=M/2.0/1 --scan-inputs=f
static cl::list<std::string> scanIo(
    "scan-io",
    cl::desc("Per-scan I/O slot for trace tests: "
             "role:name=AREA/byte[.bit]/widthBytes (role=in|out, repeat per "
             "slot, only with --runtime=sim)"));
static cl::opt<std::string> scanInputs(
    "scan-inputs",
    cl::desc("File with per-scan input values for --scan-io: one line per "
             "scan, `N name=value ...` (only with --runtime=sim)"));

namespace {
struct ScanIoSlot {
  bool isInput;
  std::string name;
  uint8_t area;
  uint32_t byteOff;
  int bit; // -1 = byte/word access, else bit number
  uint8_t width;
};

/// Parse one `--scan-io` entry: role:name=AREA/byte[.bit]/width.
static bool parseScanIoEntry(const std::string &entry, ScanIoSlot &slot) {
  auto eq = entry.find('=');
  if (eq == std::string::npos)
    return false;
  auto roleName = entry.substr(0, eq);
  auto colon = roleName.find(':');
  if (colon == std::string::npos)
    return false;
  auto role = roleName.substr(0, colon);
  slot.name = roleName.substr(colon + 1);
  slot.isInput = role == "in";
  if (role != "in" && role != "out")
    return false;
  auto spec = entry.substr(eq + 1);
  auto c1 = spec.find('/');
  if (c1 == std::string::npos || c1 == 0)
    return false;
  slot.area = (uint8_t)spec[0];
  auto rest = spec.substr(c1 + 1);
  auto c2 = rest.find('/');
  if (c2 == std::string::npos)
    return false;
  auto addr = rest.substr(0, c2);
  slot.bit = -1;
  auto dot = addr.find('.');
  if (dot != std::string::npos) {
    slot.byteOff = (uint32_t)std::stoul(addr.substr(0, dot));
    slot.bit = std::stoi(addr.substr(dot + 1));
  } else {
    slot.byteOff = (uint32_t)std::stoul(addr);
  }
  slot.width = (uint8_t)std::stoul(rest.substr(c2 + 1));
  // width 0 = string slot: an S7-layout string buffer (maxlen@0, actlen@1,
  // chars@2.., up to 256 bytes) reported as name='...' with IEC $ escapes.
  if (slot.width == 0)
    return slot.bit < 0;
  return slot.width >= 1 && slot.width <= 8 &&
         (slot.bit < 0 || slot.bit < 8);
}

/// Encode a string for the scan output in the IEC 61131-3 ST literal escape
/// form used by the conformance traces ('$' -> $$, quote -> $', tab -> $T,
/// LF -> $L, CR -> $R, other non-printables -> $<2 hex digits>).
static std::string encodeScanString(llvm::StringRef s) {
  std::string out;
  for (char c : s) {
    switch (c) {
    case '\'': out += "$'"; break;
    case '$': out += "$$"; break;
    case '\t': out += "$T"; break;
    case '\n': out += "$L"; break;
    case '\r': out += "$R"; break;
    default:
      if (c <= 0x20 || c >= 0x7F) { // space too: scan lines are whitespace-split
        char buf[4];
        snprintf(buf, sizeof(buf), "$%02X", (unsigned char)c);
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

/// Read the --scan-inputs file into scan index -> [(name, value)].
static bool parseScanInputsFile(
    const std::string &path,
    std::map<unsigned, std::vector<std::pair<std::string, int64_t>>> &out) {
  std::ifstream f(path);
  if (!f)
    return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    unsigned idx;
    if (!(ss >> idx))
      return false;
    auto &vec = out[idx];
    std::string tok;
    while (ss >> tok) {
      auto eq = tok.find('=');
      if (eq == std::string::npos)
        return false;
      // parse unsigned: 64-bit REAL/LREAL bit patterns (via __BITS) can
      // exceed INT64_MAX
      vec.push_back({tok.substr(0, eq),
                     (int64_t)std::stoull(tok.substr(eq + 1))});
    }
  }
  return true;
}
} // namespace

/// Parse SCL source text held in memory (LSP mode); `name` is used for
/// locations. `parserSink`, when set, receives structured parse errors.
std::unique_ptr<sclang::ModuleAST>
parseSourceText(llvm::StringRef text, std::string name,
                std::function<void(int, int, const std::string &, const char *)> parserSink =
                    nullptr) {
  LexerBuffer lexer(text.begin(), text.end(), std::move(name), langMode());
  Parser parser(lexer);
  parser.errorSink = std::move(parserSink);
  // Tolerant mode (LSP / --parse-tolerant): collect many diagnostics per
  // pass; the partial AST is discarded when any error was reported.
  parser.tolerantErrors = parseTolerantFlag || lspMode;
  auto ast = parser.ParseModule();
  if (parser.errorCount > 0)
    return nullptr;
  return ast;
}

std::unique_ptr<sclang::ModuleAST>
parseInputFiles(llvm::ArrayRef<std::string> files,
                std::function<void(int, int, const std::string &,
                                   const char *)> sink);

/// Expand the positional inputs + `--project` into a sorted, de-duplicated
/// file list. With no positional input, reads stdin ("-").
static llvm::SmallVector<std::string, 4> expandInputFiles() {
  llvm::SmallVector<std::string, 4> files;
  if (!projectDir.empty()) {
    std::string ext = inputType == InputType::Awl
                          ? ".awl"
                          : (inputType == InputType::St ? ".st" : ".scl");
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(projectDir, ec), end;
         !ec && it != end; it.increment(ec)) {
      if (llvm::StringRef(it->path()).ends_with(ext))
        files.push_back(it->path());
    }
    if (ec)
      llvm::errs() << "sclang: cannot read project dir '" << projectDir
                   << "': " << ec.message() << "\n";
    llvm::sort(files); // deterministic order
  }
  for (auto &f : inputFilenames)
    files.push_back(f);
  if (files.empty())
    files.push_back("-");
  return files;
}

/// Parse all input files and splice their units into one ModuleAST: a
/// project is one namespace (STEP 7 FC/FB/DB names are globally unique),
/// see docs/design/multi-file-incremental.md. Two real definitions of the
/// same unit are a project error (diagnosed with both sites); a forward
/// declaration plus one real definition is legal.
std::unique_ptr<sclang::ModuleAST>
parseInputFiles(llvm::ArrayRef<std::string> files,
                std::function<void(int, int, const std::string &,
                                   const char *)> sink = nullptr) {
  auto merged = std::make_unique<sclang::ModuleAST>(
      std::vector<std::unique_ptr<sclang::UnitAST>>{}, langMode());
  // upper-cased unit name -> (definition site, is-forward-decl)
  llvm::StringMap<std::pair<std::string, bool>> seen;
  for (auto &f : files) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(f);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return nullptr;
    }
    auto ast = parseSourceText(fileOrErr.get()->getBuffer(), f, sink);
    if (!ast)
      return nullptr;
    for (const auto &unit : *ast) {
      std::string up = unit->getIdentifier().upper();
      const auto &loc = unit->loc();
      std::string site =
          (loc.file ? *loc.file : f) + ":" + std::to_string(loc.line);
      bool fwd = sclang::isForwardDeclaration(unit.get());
      auto it = seen.find(up);
      if (it != seen.end() && !it->second.second && !fwd) {
        // two real definitions of the same unit
        std::string msg = "duplicate program unit '" +
                          unit->getIdentifier().str() + "' (also defined at " +
                          it->second.first + ")";
        llvm::errs() << (loc.file ? *loc.file : f) << ":" << loc.line << ":"
                     << loc.col << ": error: " << msg << "\n";
        if (sink)
          sink(loc.line, loc.col, msg, "SCL1006");
        return nullptr;
      }
      if (it == seen.end() || it->second.second)
        seen[up] = {site, fwd}; // a real definition replaces a fwd decl
    }
    merged->spliceFrom(std::move(*ast));
  }
  return merged;
}

/// Load a module from in-memory source text (LSP mode). `name` selects the
/// language by extension (`.awl` -> AWL, `.mlir` -> MLIR text, else SCL);
/// `parserSink` receives structured parse errors for SCL input.
int loadMLIRFromBuffer(
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    llvm::StringRef text, llvm::StringRef name,
    std::function<void(int, int, const std::string &, const char *)> parserSink = nullptr) {
  if (inputType != InputType::MLIR && !llvm::StringRef(name).ends_with(".mlir")) {
    bool isAwl = inputType == InputType::Awl ||
                 llvm::StringRef(name).ends_with(".awl");
    if (isAwl) {
      for (auto &ds : dbSyms) {
        auto eq = ds.find('=');
        if (eq == std::string::npos)
          continue;
        sclang::awlRegisterNamedDb(ds.substr(0, eq),
                                   (int32_t)atoi(ds.substr(eq + 1).c_str()));
      }
      module = sclang::awlGen(context, text, std::string(name));
      return !module ? 1 : 0;
    }
    auto moduleAST = parseSourceText(text, std::string(name), parserSink);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(text, std::string(name)),
      llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
    return 3;
  return 0;
}

/// Build with the per-file content-hash cache (--cache-dir): load unchanged
/// files' modules from the cache; if anything is missing, parse + codegen the
/// whole project once, extract and store the missed files' modules, then
/// merge all per-file modules in file order (docs/design/multi-file-
/// incremental.md phase 2).
static int loadMLIRCached(mlir::MLIRContext &context,
                          mlir::OwningOpRef<mlir::ModuleOp> &module,
                          llvm::ArrayRef<std::string> files) {
  std::string langTag = std::to_string((int)langMode());
  llvm::SmallVector<std::string, 4> contents;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> mods;
  llvm::SmallVector<unsigned, 4> missIdx;
  unsigned hits = 0;
  for (auto &f : files) {
    auto bufOrErr = llvm::MemoryBuffer::getFileOrSTDIN(f);
    if (std::error_code ec = bufOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return 6;
    }
    contents.push_back(bufOrErr.get()->getBuffer().str());
    std::string key = sclang::projectcache::key(contents.back(), langTag);
    auto m = sclang::projectcache::load(context, cacheDir, key);
    if (m) {
      ++hits;
      mods.push_back(std::move(m));
    } else {
      missIdx.push_back((unsigned)mods.size());
      mods.push_back(mlir::OwningOpRef<mlir::ModuleOp>());
    }
  }
  if (!missIdx.empty()) {
    // Something changed: parse + codegen the whole project once and extract
    // each missed file's module-level ops (recorded by mlirGenPerFile).
    auto ast = parseInputFiles(files);
    if (!ast)
      return 6;
    llvm::StringMap<llvm::SmallVector<mlir::Operation *, 4>> opsByFile;
    auto full = mlirGenPerFile(context, *ast, opsByFile);
    if (!full)
      return 1;
    for (unsigned i : missIdx) {
      auto it = opsByFile.find(files[i]);
      llvm::SmallVector<mlir::Operation *, 4> ops;
      if (it != opsByFile.end())
        ops = it->second;
      auto fm = sclang::projectcache::extractFileModule(*full, ops);
      sclang::projectcache::store(
          cacheDir, sclang::projectcache::key(contents[i], langTag), *fm);
      mods[i] = std::move(fm);
    }
  }
  auto merged = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  for (auto &m : mods) {
    if (!m.get())
      return 1;
    if (mlir::failed(sclang::projectcache::mergeInto(merged, *m)))
      return 1;
  }
  if (mlir::failed(mlir::verify(merged))) {
    merged.emitError("module verification error after cache merge");
    return 1;
  }
  module = std::move(merged);
  if (cacheStats)
    llvm::errs() << "sclang cache: " << hits << " hits, " << missIdx.size()
                 << " misses (" << files.size() << " files)\n";
  return 0;
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::SmallVector<std::string, 4> files = expandInputFiles();
  bool isMlirInput = inputType == InputType::MLIR;
  bool isAwlInput = inputType == InputType::Awl;
  for (auto &f : files) {
    isMlirInput = isMlirInput || llvm::StringRef(f).ends_with(".mlir");
    isAwlInput = isAwlInput || llvm::StringRef(f).ends_with(".awl");
  }
  if (!cacheDir.empty() && !isMlirInput && !isAwlInput)
    return loadMLIRCached(context, module, files);
  if (files.size() == 1) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(files[0]);
    if (std::error_code EC = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << EC.message() << "\n";
      return isMlirInput ? -1 : 6;
    }
    return loadMLIRFromBuffer(context, module, fileOrErr.get()->getBuffer(),
                              files[0]);
  }
  // Multi-file project (docs/design/multi-file-incremental.md).
  if (isMlirInput) {
    llvm::errs() << "sclang: multiple .mlir inputs are not supported (merge "
                    "them into one file first)\n";
    return -1;
  }
  if (inputType == InputType::Awl ||
      llvm::all_of(files,
                   [](const std::string &f) {
                     return llvm::StringRef(f).ends_with(".awl");
                   })) {
    // AWL is a linear instruction list: concatenate the files (locations
    // then refer to the concatenated text; v1 limitation, see design doc).
    std::string text, name;
    for (auto &f : files) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(f);
      if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return 6;
      }
      text += fileOrErr.get()->getBuffer();
      text += '\n';
      if (name.empty())
        name = f;
      else
        name += "+";
    }
    for (auto &ds : dbSyms) {
      auto eq = ds.find('=');
      if (eq == std::string::npos)
        continue;
      sclang::awlRegisterNamedDb(ds.substr(0, eq),
                                 (int32_t)atoi(ds.substr(eq + 1).c_str()));
    }
    module = sclang::awlGen(context, text, name);
    return !module ? 1 : 0;
  }
  auto moduleAST = parseInputFiles(files);
  if (!moduleAST)
    return 6;
  module = mlirGen(context, *moduleAST);
  return !module ? 1 : 0;
}

/// Run the lowering/optimization pass pipeline on an already-loaded module
/// (split out of loadAndProcessMLIR so the LSP server can run it on an
/// in-memory module).
int processModule(mlir::MLIRContext &context,
                  mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  // Dev-only pass timing: --mlir-timing (see AGENTS.md).
  applyDefaultTimingPassManagerCLOptions(pm);
  pm.enableVerifier(verifyEach);

  if (passPipelineOpt.getNumOccurrences()) {
    // Explicit user pipeline (mlir-opt style) — replaces the built-in one.
    // Parse as a rooted pipeline (same as mlir-opt) and run it.
    auto parsed = mlir::parsePassPipeline(passPipelineOpt);
    if (mlir::failed(parsed))
      return 4;
    mlir::PassManager pipelinePm(&context);
    static_cast<mlir::OpPassManager &>(pipelinePm) = std::move(*parsed);
    pipelinePm.enableVerifier(verifyEach);
    if (mlir::failed(pipelinePm.run(*module)))
      return 4;
    return 0;
  }

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToStd = actionLowersToStd();
  bool isLoweringToLLVM = actionLowersToLLVM();

  if (enableOpt || isLoweringToStd) {
    mlir::OpPassManager &optFC = pm.nest<mlir::scl::FunctionOp>();
    optFC.addPass(mlir::createCanonicalizerPass());
    optFC.addPass(mlir::createCSEPass());
    mlir::OpPassManager &optFB = pm.nest<mlir::scl::FunctionBlockOp>();
    optFB.addPass(mlir::createCanonicalizerPass());
    optFB.addPass(mlir::createCSEPass());
 }

  // The AWL-dialect path: lower SCL to awl dialect ops in memory (the same
  // conversion the --emit=awl text backend performs before printing), then
  // let the regular std/LLVM lowering consume the awl blocks — no AWL text
  // print/re-parse round-trip.
  if (emitAction == Action::DumpAWLMLIR || emitAction == Action::RunJITAwl)
    pm.addPass(mlir::sclang::createLowerToAWLPass(
        /*synthCiface=*/emitAction == Action::RunJITAwl));
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

  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::sclang::createLowerToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;
  return processModule(context, module);
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFiles(expandInputFiles());
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

/// Create a TargetMachine for the host so the LLVM optimization pipeline is
/// target-aware (enables e.g. AArch64-specific IR opts) instead of running
/// with a null target machine.
static std::unique_ptr<llvm::TargetMachine> createHostTargetMachine() {
  llvm::Triple triple(llvm::sys::getProcessTriple());
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return nullptr;
  llvm::TargetOptions options;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, llvm::sys::getHostCPUName(), /*features=*/"", options,
      llvm::Reloc::PIC_));
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
  auto targetMachine = createHostTargetMachine();
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/targetMachine.get());
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

/// Emit the fully lowered module as a host object file (AOT path): the
/// scl_rt_* runtime symbols stay undefined and are resolved when the object
/// is linked against a host runtime (--runtime-lib) — see ADR 002.
int emitObjectFile(mlir::ModuleOp module) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto targetMachine = createHostTargetMachine();
  if (!targetMachine) {
    llvm::errs() << "No host target machine\n";
    return -1;
  }
  llvmModule->setTargetTriple(llvm::Triple(targetMachine->getTargetTriple()));
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/targetMachine.get());
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Cannot open output file '" << outputFilename << "': "
                 << ec.message() << "\n";
    return -1;
  }
  llvm::legacy::PassManager pm;
  if (targetMachine->addPassesToEmitFile(pm, out, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target cannot emit an object file\n";
    return -1;
  }
  pm.run(*llvmModule);
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
    add("scl_rt_timer_reset", (void *)&scl_rt_timer_reset);
    add("scl_rt_db_info", (void *)&scl_rt_db_info);
    add("scl_rt_sim_set_db_size", (void *)&scl_rt_sim_set_db_size);
    add("scl_rt_sim_db_ref", (void *)&scl_rt_sim_db_ref);
    add("scl_rt_timer_fr", (void *)&scl_rt_timer_fr);
    add("scl_rt_sleep", (void *)&scl_rt_sleep);
    add("scl_rt_frame_set_fb", (void *)&scl_rt_frame_set_fb);
    add("scl_rt_l_push", (void *)&scl_rt_l_push);
    add("scl_rt_l_pop", (void *)&scl_rt_l_pop);
    add("scl_rt_param_in", (void *)&scl_rt_param_in);
    add("scl_rt_param_out", (void *)&scl_rt_param_out);
    add("scl_rt_sfb", (void *)&scl_rt_sfb);
    add("scl_rt_sfc", (void *)&scl_rt_sfc);
    add("scl_rt_iec", (void *)&scl_rt_iec);
    add("scl_rt_pack_area", (void *)&scl_rt_pack_area);
    add("scl_rt_counter_op_ptr", (void *)&scl_rt_counter_op_ptr);
    add("scl_rt_timer_start_ptr", (void *)&scl_rt_timer_start_ptr);
    add("scl_rt_timer_reset_ptr", (void *)&scl_rt_timer_reset_ptr);
    add("scl_rt_fr_ptr", (void *)&scl_rt_fr_ptr);
    // status word (STW), ABI v8
    add("scl_rt_accu_get", (void *)&scl_rt_accu_get);
    add("scl_rt_accu_set", (void *)&scl_rt_accu_set);
    add("scl_rt_stw_get", (void *)&scl_rt_stw_get);
    add("scl_rt_stw_set", (void *)&scl_rt_stw_set);
    add("scl_rt_stw_reset", (void *)&scl_rt_stw_reset);
    add("scl_rt_timer_lc", (void *)&scl_rt_timer_lc);
    add("scl_rt_stw_set_bit", (void *)&scl_rt_stw_set_bit);
    add("scl_rt_stw_cmp_i", (void *)&scl_rt_stw_cmp_i);
    add("scl_rt_stw_cmp_f", (void *)&scl_rt_stw_cmp_f);
    add("scl_rt_stw_sign_i", (void *)&scl_rt_stw_sign_i);
    add("scl_rt_stw_sign_f", (void *)&scl_rt_stw_sign_f);
    add("scl_rt_arith_f", (void *)&scl_rt_arith_f);
    add("scl_rt_realmath_f", (void *)&scl_rt_realmath_f);
    add("scl_rt_stw_div_i", (void *)&scl_rt_stw_div_i);
    add("scl_rt_stw_div_f", (void *)&scl_rt_stw_div_f);
    add("scl_rt_debug_print", (void *)&scl_rt_debug_print);
    add("scl_rt_debug_print_i", (void *)&scl_rt_debug_print_i);
    add("scl_rt_debug_print_r", (void *)&scl_rt_debug_print_r);
    add("scl_rt_f2i", (void *)&scl_rt_f2i);
    add("scl_rt_ob_start", (void *)&scl_rt_ob_start);
    add("scl_rt_feature", (void *)&scl_rt_feature);
    add("scl_rt_timer_fb", (void *)&scl_rt_timer_fb);
    add("scl_rt_counter_fb", (void *)&scl_rt_counter_fb);
    add("scl_rt_edge_fb", (void *)&scl_rt_edge_fb);
    add("scl_rt_str_to_m", (void *)&scl_rt_str_to_m);
    add("scl_rt_realmath_d", (void *)&scl_rt_realmath_d);
    add("scl_rt_assert_cmp", (void *)&scl_rt_assert_cmp);
    add("scl_rt_assert_cmp_r", (void *)&scl_rt_assert_cmp_r);
    // ABI v5: DB register / address registers / indirect access
    add("scl_rt_set_current_db", (void *)&scl_rt_set_current_db);
    add("scl_rt_set_current_di", (void *)&scl_rt_set_current_di);
    add("scl_rt_current_di", (void *)&scl_rt_current_di);
    add("scl_rt_pack_s7", (void *)&scl_rt_pack_s7);
    add("scl_rt_current_db", (void *)&scl_rt_current_db);
    add("scl_rt_ar_set", (void *)&scl_rt_ar_set);
    add("scl_rt_ar_get", (void *)&scl_rt_ar_get);
    add("scl_rt_ar_raw", (void *)&scl_rt_ar_raw);
    add("scl_rt_ar_add", (void *)&scl_rt_ar_add);
    add("scl_rt_read_indirect", (void *)&scl_rt_read_indirect);
    add("scl_rt_write_indirect", (void *)&scl_rt_write_indirect);
    add("scl_rt_read_bit_indirect", (void *)&scl_rt_read_bit_indirect);
    add("scl_rt_write_bit_indirect", (void *)&scl_rt_write_bit_indirect);
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
    if (upper.starts_with("DB")) {
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

int runJit(mlir::ModuleOp module, llvm::raw_ostream *scanOut = nullptr) {
  llvm::raw_ostream &scanOS = scanOut ? *scanOut : llvm::outs();
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
  auto targetMachine = createHostTargetMachine();
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/targetMachine.get());

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

  // Run the DB initialization block once (DATA_BLOCK initial values from
  // AWL, if the frontend emitted an awl.block "__scl_db_init").
  if (auto initFn =
          engine->lookupPacked("_mlir_ciface___scl_db_init")) {
    if (auto err = engine->invokePacked("_mlir_ciface___scl_db_init", {}))
      llvm::errs() << "JIT db-init invocation failed\n" << err << "\n";
  }

  // S7 startup behavior (awlsim cpu.startup): on a 4-accumulator CPU run
  // OB 102 (cold start) if present, else OB 100 (warm start); then the
  // cyclic scans of OB 1.
  if (scl_rt_feature(0, -1) == 4 && engine->lookupPacked("OB102")) {
    if (auto err = engine->invokePacked("OB102", {}))
      llvm::errs() << "JIT OB102 invocation failed\n" << err << "\n";
  } else if (engine->lookupPacked("OB100")) {
    if (auto err = engine->invokePacked("OB100", {}))
      llvm::errs() << "JIT OB100 invocation failed\n" << err << "\n";
  }

  // Invoke the JIT-compiled function once per scan cycle, advancing the
  // virtual clock between scans (drives %T timers / %C counters).
  // awlsim with no OB 1 just idles — tolerate a missing OB 1 (e.g. FB-only
  // source files) instead of failing the run.
  if (engine->lookupPacked("OB1")) {
    // Per-scan I/O slots for trace-driven tests (--scan-io/--scan-inputs).
    std::vector<ScanIoSlot> scanSlots;
    std::map<unsigned, std::vector<std::pair<std::string, int64_t>>> scanIn;
    bool scanIoOk = true;
    if (!scanIo.empty()) {
      if (runtimeKind != Sim) {
        llvm::errs() << "--scan-io requires --runtime=sim\n";
        return -1;
      }
      for (auto &entry : scanIo) {
        ScanIoSlot slot;
        if (!parseScanIoEntry(entry, slot)) {
          llvm::errs() << "invalid --scan-io entry: " << entry << "\n";
          scanIoOk = false;
        } else {
          scanSlots.push_back(slot);
        }
      }
      if (!scanInputs.empty() &&
          !parseScanInputsFile(scanInputs, scanIn)) {
        llvm::errs() << "cannot parse --scan-inputs file: " << scanInputs
                     << "\n";
        scanIoOk = false;
      }
    }
    if (!scanIoOk)
      return -1;
    for (unsigned i = 0; i < jitCycles; ++i) {
      // Before the scan: push this scan's input values into the in-slots.
      if (auto it = scanIn.find(i); it != scanIn.end())
        for (auto &slot : scanSlots) {
          if (!slot.isInput)
            continue;
          auto vit = llvm::find_if(it->second, [&](auto &p) {
            return p.first == slot.name;
          });
          if (vit == it->second.end())
            continue;
          if (slot.bit >= 0)
            scl_rt_write_bit(slot.area, 0, slot.byteOff, (uint8_t)slot.bit,
                             vit->second != 0);
          else
            scl_rt_write(slot.area, 0, slot.byteOff, slot.width,
                         (uint64_t)vit->second);
        }
      auto invocationResult = engine->invoke("OB1");
      if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n" << invocationResult << "\n";
        return -1;
      }
      // After the scan: report the out-slots as `scan N: name=value ...`.
      if (!scanSlots.empty()) {
        scanOS << "scan " << i << ":";
        for (auto &slot : scanSlots) {
          if (slot.isInput)
            continue;
          if (slot.width == 0) {
            // String slot: S7 layout maxlen@0, actlen@1, chars@2..
            uint32_t maxlen =
                (uint8_t)scl_rt_read(slot.area, 0, slot.byteOff, 1);
            uint32_t actlen =
                (uint8_t)scl_rt_read(slot.area, 0, slot.byteOff + 1, 1);
            if (actlen > maxlen)
              actlen = maxlen;
            std::string s;
            for (uint32_t i = 0; i < actlen; ++i)
              s += (char)(uint8_t)scl_rt_read(slot.area, 0,
                                              slot.byteOff + 2 + i, 1);
            scanOS << " " << slot.name << "='" << encodeScanString(s) << "'";
            continue;
          }
          int64_t v = slot.bit >= 0
                          ? (scl_rt_read_bit(slot.area, 0, slot.byteOff,
                                             (uint8_t)slot.bit)
                                 ? 1
                                 : 0)
                          : scl_rt_read(slot.area, 0, slot.byteOff, slot.width);
          scanOS << " " << slot.name << "=" << v;
        }
        scanOS << "\n";
      }
      if (scl_rt_stop_requested())
        break; // SFC 46 (STOP) / SFC -3 (SHUTDOWN): awlsim halts the CPU
      if (i + 1 < jitCycles)
        scl_rt_cycle(cycleMs);
    }
  }

  if (dumpAreas && runtimeKind == Sim)
    scl_rt_sim_dump_areas();

  // Test oracles (awlsim __ASSERT): any failed assert fails the run.
  if (runtimeKind == Sim && scl_rt_assert_failures() > 0) {
    llvm::errs() << scl_rt_assert_failures() << " assert(s) failed\n";
    return 2;
  }

  return 0;
}

namespace sclang {

/// LSP cancellation state, read by Parser::abortCheck (set in main when
/// --lsp): set on exit, plus a per-compile parse deadline (ms since epoch,
/// 0 = none) so a pathological document cannot hang the server.
std::atomic<bool> g_lspShutdown{false};
std::atomic<int64_t> g_parseDeadlineMs{0};

/// One structured diagnostic (1-based line/col, 0 = unknown).
struct LspDiag {
  int line;
  int col;
  std::string message;
  bool warning;
  std::string code; // stable diagnostic code, e.g. SCL1001 (parse) / MLIR
};

/// Load every dialect the compiler emits or consumes into `context`.
/// Shared by main() and the LSP server so both see the same dialect set.
void registerSclangDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::scl::SclDialect>();
  context.getOrLoadDialect<mlir::awl::AwlDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
}

// Set by lspCompile; the MLIR diagnostic handler appends to it.
static std::vector<LspDiag> *g_lspDiags = nullptr;

static void lspDiagHandler(mlir::Diagnostic &d) {
  if (!g_lspDiags)
    return;
  LspDiag out{0, 0, {}, d.getSeverity() == mlir::DiagnosticSeverity::Warning,
              "MLIR"};
  if (auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(d.getLocation())) {
    out.line = flc.getLine();
    out.col = flc.getColumn();
  }
  llvm::raw_string_ostream os(out.message);
  d.print(os);
  out.message = os.str();
  g_lspDiags->push_back(std::move(out));
}

/// Compile in-memory source text (LSP mode). `name` picks the language by
/// extension. Diagnostics (parse errors via the parser sink + MLIR/verifier
/// diagnostics via the context handler) are appended to `diags`. When
/// `fullPipeline` is set, the lowering pipeline runs too (forced to full
/// depth). Returns 0 on success.
int lspCompile(mlir::MLIRContext &ctx, llvm::StringRef text, llvm::StringRef name,
               bool fullPipeline, std::vector<LspDiag> &diags,
               mlir::OwningOpRef<mlir::ModuleOp> &module) {
  static bool handlerRegistered = false;
  if (!handlerRegistered) {
    ctx.getDiagEngine().registerHandler(lspDiagHandler);
    handlerRegistered = true;
  }
  g_lspDiags = &diags;
  g_parseDeadlineMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
              .count() +
      10000; // 10s parse budget per LSP compile
  int rc = loadMLIRFromBuffer(ctx, module, text, name,
                              [&](int line, int col, const std::string &msg,
                                  const char *code) {
                                diags.push_back(
                                    {line, col, msg, false, code});
                              });
  if (rc == 0 && module)
    rc = mlir::failed(mlir::verify(*module)) ? 1 : 0;
  if (rc == 0 && fullPipeline && module) {
    Action saved = emitAction;
    emitAction = Action::RunJIT; // full lowering depth
    rc = processModule(ctx, module);
    emitAction = saved;
  }
  g_lspDiags = nullptr;
  g_parseDeadlineMs = 0;
  return rc;
}

/// Run the JIT scan loop on a fully lowered module; scan output lines go to
/// `out`. Returns 0 on success.
int lspRunJit(mlir::ModuleOp module, unsigned cycles, llvm::raw_ostream &out) {
  unsigned saved = jitCycles;
  jitCycles = cycles;
  int rc = runJit(module, &out);
  jitCycles = saved;
  return rc;
}

} // namespace sclang

namespace sclang {
// Defined in Lsp.cpp: minimal LSP server (JSON-RPC over stdio).
int runLspServer(unsigned jitCycles);
} // namespace sclang

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  // Make our lowering passes addressable by name in pass pipelines
  // (Passes.td registration; used by sclang-opt and --pass-pipeline).
  mlir::sclang::registerSclangTransformsPasses();

  cl::ParseCommandLineOptions(argc, argv, "SCL compiler\n");

  if (lspMode) {
    // Cancellable parsing for the server: abort on shutdown or when the
    // per-compile deadline (set in lspCompile) has passed.
    sclang::Parser::abortCheck = [] {
      if (sclang::g_lspShutdown.load(std::memory_order_relaxed))
        return true;
      int64_t dl = sclang::g_parseDeadlineMs.load(std::memory_order_relaxed);
      return dl > 0 &&
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                     .count() > dl;
    };
    return sclang::runLspServer(jitCycles);
  }

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  // MLIR only prints errors to stderr by default; register a handler so
  // warnings (e.g. ignored non-constant FB initializers) are shown too.
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    llvm::errs() << "\n";
    return mlir::success();
  });
  // Load our dialects in this MLIR Context.
  sclang::registerSclangDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = actionOutputsMLIR();
  if (isOutputingMLIR) {
    module->print(llvm::outs());
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // AOT: emit a host object file.
  if (emitAction == Action::EmitObj)
    return emitObjectFile(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT || emitAction == Action::RunJITAwl)
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
