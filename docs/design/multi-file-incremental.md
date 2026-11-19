# Multi-file projects and incremental compilation (plan, not yet implemented)

Status: design only. Nothing below is implemented except where noted.
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

## Phased plan

1. **Multi-file CLI**: multiple positional inputs / `--project=`; splice
   `ModuleAST`s; one module through the existing pipeline. Validate against
   the existing per-project conformance tier (make it native instead of
   concatenating).
2. **Build cache**: `--cache-dir=`; sha256(source+flags+version) ->
   per-file MLIR; write-through on build; reuse on rebuild. Manifest of
   project files; missing/changed entries recompile.
3. **LSP project mode**: open docs override cache entries; runScans links
   open (recompiled) + cached (unchanged) modules and JITs; per-file
   diagnostics via locations.
4. **Optional**: per-file LLVM bitcode artifacts + ORC object cache for the
   run path when projects get large; error-tolerant parsing (rust-analyzer
   style) so one broken file doesn't hide other files' diagnostics.

## Open questions

- Merging per-file MLIR modules directly (instead of AST splicing) hits
  private-stub symbol collisions; AST splicing avoids this for full builds,
  LLVM-IR linking avoids it for per-file artifacts. Don't build an MLIR
  module linker unless a use case needs cross-file MLIR opts on cached
  modules (then rename private symbols on merge, like `llvm::Linker` does).
- Symbol conflicts across files (duplicate FC names) must be diagnosed at
  splice time with both file locations — S7 treats this as a project error.
