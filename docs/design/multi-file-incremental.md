# Multi-file projects and incremental compilation

Status: **phase 1 implemented** (multi-file CLI: multiple positional inputs +
`--project=<dir>`, ModuleAST unit splicing, forward-declaration semantics,
duplicate-POU diagnostics; conformance per-project tier runs natively, 17/17).
Phases 2-4 (per-file codegen split, content-hash cache, LSP project mode) are
plans below.
Related: `docs/design/` (memory-areas, awl-frontend, scl-to-awl), AGENTS.md
"LSP mode", ADR 002 (runtime C ABI), ADR 004 (ST as a mode).

## Current state (facts from the code)

- The CLI takes exactly one positional input (`inputFilename`,
  `sclang/sclang.cpp:81`). The LSP server (`sclang/Lsp.cpp`) already holds
  multiple open documents in memory, but compiles each document standalone.
- `ModuleAST` is just `std::vector<std::unique_ptr<UnitAST>>`
  (`include/sclang/SclGen/AST.h:1361`) — splicing the units of N parsed files
  into one `ModuleAST` is trivial and gives one global namespace.
- MLIRGen already builds its call-signature tables (`functionResultTypes`,
  `lib/SclGen/MLIRGen.cpp:175-205`) from *all* units of the module, and calls
  go out by name via `scl.call_fc` (`MLIRGen.cpp:5163,5175`); a callee not in
  the module gets an on-the-fly **private** stub (`MLIRGen.cpp:5155-5166`).
  So with spliced units, cross-file FC/FB calls get correct signatures and
  resolve inside the single merged module — no MLIR linker needed.
- The conformance harness already validates this semantics: its per-project
  tier concatenates all `.scl` files of a directory into one input
  (`test/conformance/run_conformance.py:244-260`, 17/17 projects pass).
  awlsim likewise loads a whole project into one CPU; TIA Portal compiles
  the whole project into one download image.
- MLIR 23's `ExecutionEngine` takes **one** module per engine (no
  `addModule`; `mlir/.../ExecutionEngine.h:146`). Multi-module JIT therefore
  links either at MLIR level (merge modules — private stub symbols collide
  and would need renaming) or at LLVM IR level
  (`llvm::Linker::linkModules`, which auto-renames colliding private
  symbols). `ExecutionEngineOptions::enableObjectCache` exists for an ORC
  object cache on disk.

## Decisions

1. **Project = one namespace = one MLIR module.** Multi-file input means:
   parse each file, splice the `ModuleAST` unit vectors, run MLIRGen once.
   CLI: accept multiple positional inputs and/or `--project=<dir|manifest>`.
   This matches STEP 7 / awlsim semantics exactly (globally unique FC/FB/DB
   names) and needs no MLIR-level linking for the whole-project build.
2. **Per-file artifacts for incremental builds.** The cacheable unit is the
   *file*: `.sclang-cache/<sha256(source, flags, compiler-version)>.mlir`
   holding the file's scl-dialect module (text MLIR, human-inspectable,
   enables merge + whole-project `--opt`). For the run path we can
   additionally store lowered LLVM bitcode (`<hash>.bc`), which is the
   closest analog to a `.o`/`.rlib`: rebuild = hash sources, reuse hits,
   recompile misses, `llvm::Linker::linkModules` the bitcode, one
   `ExecutionEngine` over the linked module.
3. **LSP project mode.** The server keeps open documents in memory (they
   win over the cache); `sclang/runScans` / didSave compiles only the
   changed document and links it against the cached artifacts of the rest
   of the project, then JITs. Diagnostics are attributed per file via the
   file/line/col already carried in MLIR locations and the parser sink.
4. **No incremental linking.** Linking a handful of small modules is
   milliseconds; lld `--incremental` solves a problem (10k-object game
   binaries) we don't have. What must be avoided is re-*compilation* of
   unchanged files (the hash cache) and, for large projects, re-JIT of
   unchanged machine code (optional: point `enableObjectCache` at the
   cache dir; measure first — lowering 100k lines is ~0.08s and small
   projects re-JIT in ms).

## Alignment with standard tooling

| Tool | Mechanism | Our analog |
|------|-----------|------------|
| make/ninja | depfiles: rebuild only changed TUs | per-file hash check, recompile misses only |
| Bazel/Buck2 | action cache: hash(inputs) -> artifact, optionally remote | `.sclang-cache/<hash>.mlir/.bc`, shareable dir |
| ccache | hash(preprocessed TU + flags + compiler) -> object | hash(source + flags + sclang version) -> module/bitcode |
| rustc | per-crate incremental queries + `.rlib`/`.rmeta` for deps | per-file parse/MLIR cache + cached module as the "rlib" |
| rust-analyzer | LSP server; recompiles only the open crate against cached dep metadata | `sclang --lsp`: open docs in memory, rest from cache, run on save |
| lld --incremental | incremental relink for huge binaries | not needed at PLC project sizes (decision 4) |
| TIA Portal / awlsim | whole project compiled/loaded as one image | one module per project (decision 1) |

## Implementation proposal (phases = self-contained commits)

**Phase 1 — multi-file CLI, whole-project build via AST splice. DONE.**
Notes from the implementation:
- STEP 7 forward declarations are legal: a unit with an empty body
  (e.g. `Forward-declarations.scl`) may share its name with the real
  definition. `parseInputFiles` (sclang.cpp) allows fwd-decl + one real
  definition and rejects two real definitions with both sites. MLIRGen's
  pre-passes must dedup signatures per name (`registerCallableSignature`,
  MLIRGen.cpp) or the call gets the parameters twice; the main loop must
  not re-`push_back` a returned existing op (fwd-decl after the definition
  used to corrupt the module and hang — fixed by checking
  `func->getParentOp() != theModule`).
- Implemented as: multiple positional inputs, `--project=<dir>` (sorted
  `*.scl`/`*.awl`/`*.st` by `-x`), `ModuleAST::spliceFrom`; AWL multi-file
  concatenates texts (v1 limitation: locations refer to the concatenated
  text); multi `.mlir` inputs rejected.
- Tests: `test/sclang/multi_file{,_callee,_dup,_fwddecl}.scl` (cross-file
  call with real result type, both file orders, fwd-decl both orders with
  JIT scan values, duplicate-name negative); conformance per-project tier
  uses explicit positional file lists (and `--project=` on dirs without
  non-SCL files like `makefile.scl`).
- `include/sclang/SclGen/AST.h`: add `ModuleAST::spliceFrom(ModuleAST&&)`
  (units are private; begin/end are const-only today) plus unit-name access
  for duplicate detection (`UnitAST::getIdentifier()` exists, AST.h:1172).
- `sclang/sclang.cpp`: `inputFilename` (sclang.cpp:81) becomes
  `cl::list<std::string> inputFilenames` (Positional, OneOrMore) — touchpoints
  are sclang.cpp:396/400/404/490; new `--project=<dir>` expanding to the
  sorted `*.scl` (or `*.awl` in awl mode) files of the dir; new
  `parseInputFiles()`: parse each file with its own name (per-file
  locations + parser errorSink already exist via `parseSourceText`), splice
  into one `ModuleAST`, diagnose duplicate POU names with both file:line
  (today `functionResultTypes[name]` silently overwrites and symbol-table
  insert of a duplicate name is a release-build hazard).
- MLIRGen needs NO change for this: `mlirGen(const ModuleAST*)`
  (MLIRGen.cpp:121) already runs whole-module pre-passes (types, FB names,
  `functionResultTypes`, MLIRGen.cpp:126-209) before codegen'ing units.
- AWL multi-file v1: concatenate the texts (one synthetic name; locations
  become project-relative — documented limitation). Multi `.mlir` inputs:
  rejected with a clear error in v1.
- Tests: lit — two positional files with a cross-file FC call (check the
  callee's real result type, not an i32 stub), duplicate-name negative
  test; switch the conformance per-project tier from concatenation to
  `--project=dir` and require identical pass counts (17/17).

**Phase 2a — two-pass per-file codegen (prerequisite for per-file cache).**
Extract MLIRGen's two pre-pass loops into `collectModuleGlobals(const
ModuleAST&, Globals&)` and make unit codegen `mlirGenUnit(unit, Globals&)`
into a per-file module. Whole-project gen = collect(all spliced units) + gen
each unit into one module — must produce IR identical to today's single
`mlirGen` call (lit check). Per-file path = parse all files (cheap,
~50ms/100k lines) for globals, codegen only the file being rebuilt.

**Phase 2b — content-hash cache (`sclang/ProjectCache.{h,cpp}`, `--cache-dir=`).**
Key = SHA-256(file content + sclang version/ABI tag + lang mode); artifact
= the file's scl-dialect module as text `<key>.mlir` (parseable via
`parseSourceFile` from a memory buffer). Build flow: parse all project
files → per file: cache hit ? load : `mlirGenUnit` + store → merge modules
with `mergeModulesWithPrivateRename` (small util: on name collision where
both symbols are private, `SymbolTable::replaceAllSymbolUses` + rename in
the incoming module, then splice the body — same trick as `llvm::Linker`)
→ verify → existing pipeline → JIT. No manifest needed: hashing current
files is the invalidation (Bazel action-cache style). Tests: build twice →
all hits; touch one file → one miss; corrupt a cache entry → recompiled;
full suite green.

**Phase 3 — LSP project mode (`sclang/Lsp.cpp`).**
On `initialize`, take `rootUri` and scan the project files (or accept a
custom `sclang/openProject {files[]}`). didOpen documents override disk/
cache. didChange recompiles only that document's module and republishes
that URI's diagnostics (bucket diagnostics by location file name — extend
`LspDiag` with a `file` field). didSave / `sclang/runScans`: merged module =
open docs (recompiled) + other files (cache/disk) → verify → pipeline →
JIT → `sclang/scanResults`. Tests: extend `test/conformance/lsp_smoke.py`
with a two-file project, a cross-file call, an error in one file
(diagnostics only on that URI), and a didSave run with correct scan values.

**Phase 4 — optional, only if measurements demand it**: per-file LLVM
bitcode artifacts + `ExecutionEngineOptions::enableObjectCache` pointed at
the cache dir (skip re-JIT of unchanged code on large projects);
error-tolerant parsing (rust-analyzer style) for multi-error diagnostics.

## Open questions

- Merging per-file MLIR modules directly (instead of AST splicing) hits
  private-stub symbol collisions; AST splicing avoids this for full builds,
  LLVM-IR linking avoids it for per-file artifacts. Don't build an MLIR
  module linker unless a use case needs cross-file MLIR opts on cached
  modules (then rename private symbols on merge, like `llvm::Linker` does).
- Symbol conflicts across files (duplicate FC names) must be diagnosed at
  splice time with both file locations — S7 treats this as a project error.
