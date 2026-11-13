# AGENTS.md — working on sclang

MLIR-based SCL (Siemens Structured Control Language) compiler. Repo root is the
project root; branch `main`.

## Architecture / pipeline

```
.scl ──> sclang-gen (SclGen: Lexer → Parser → AST → MLIRGen) ──> scl dialect ops
      ──> LowerToStd pass (lib/SclTransforms/LowerToStd.cpp, partial conversion)
      ──> func/arith/memref/scf/cf
      ──> sclang (sclang/LowerToLLVM.cpp: scf→cf, memref→llvm, arith/func/cf→llvm)
      ──> LLVM IR / ExecutionEngine JIT
```

- `sclang-gen --emit=mlir` prints unlowered scl-dialect IR; `--emit=mlir-std` runs LowerToStd.
- `sclang --emit=jit` runs the full pipeline and JIT-executes. The JIT entry point is
  the `_mlir_ciface_OB1` wrapper, which calls `function OB1` (see test/sclang/jit.scl).
- `--opt` enables canonicalizer/CSE between stages.

## LLVM version

`third_party/llvm-project` is pinned (shallow) to release tag **llvmorg-17.0.6**.
Do not bump blindly. LLVM 17 API facts baked into the code:
`std::optional` (not llvm::Optional), get-prefixed ODS accessors (`getValue()`,
`getResult()`, `getDoBody()`), `cf::CondBranchOp`, `arith::` dialect,
`populateFinalizeMemRefToLLVMConversionPatterns`, `mlir::affine::createLoopFusionPass`,
SCF header `mlir/Dialect/SCF/IR/SCF.h`, CMake target `MLIRMemRefDialect`,
`Pure` (not NoSideEffect), `ExecutionEngineOptions` with `.transformer`,
`registerBuiltinDialectTranslation` + `registerLLVMDialectTranslation` needed for JIT.

## Build & test (arm64 macOS host)

Build dir is `build.release` (already configured):

```
cd build.release
ninja sclang sclang-gen          # build tools
ninja check-sclang              # run all lit tests (background it; poll the log)
./bin/llvm-lit -sv tools/sclang/test/SclTransforms   # run a subset
./bin/sclang --opt --emit=jit test/sclang/jit_for.scl # manual JIT run
```

CMake was configured with `-DCMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja
-DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_HOST_TRIPLE=arm64-apple-darwin
-DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin -DLLVM_ENABLE_ASSERTIONS=OFF
-DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF
-DLLVM_ENABLE_TERMINFO=OFF -DLLVM_DISABLE_ASSEMBLY_FILES=ON`.
Assertions are OFF because the system libc++ rejects `_LIBCPP_ENABLE_HARDENED_MODE`.
cmake is at `/opt/homebrew/bin/cmake`. Host quirks: BSD `sed` needs `sed -i ''` and
has no `\b` — use python3 for regex edits.

## Tests

- lit tests live in `test/SclGen/Parse`, `test/SclGen/MLIR`, `test/SclTransforms`,
  `test/sclang`; BUILD files glob `*.scl`, so new files are picked up automatically.
- RUN conventions: `sclang-gen --emit=mlir-std %s | FileCheck %s` for lowering tests,
  `sclang --opt --emit=jit %s 2>&1 | FileCheck %s` for JIT tests.
- SSA names print as `%alloca`, `%reinterpret_cast`, `%0`… — match with
  `%[a-z_0-9]+` / `%[0-9]+`, not literal names.
- 53 tests currently pass (SclGen/Parse, SclGen/MLIR, SclTransforms, sclang JIT).

## Key design decisions (do not regress)

- `SclTypeConverter` maps `!scl.address<T>` to a **0-d memref with dynamic offset**
  (`memref<T, strided<[], offset: ?>>`) so array element addresses from
  `memref.reinterpret_cast` type-check. `memref.alloca` rejects non-identity
  layouts, so TempVariableOpLowering allocas a plain 0-d memref and
  reinterpret_casts to the address type with offset 0.
- `scl.condition` / `scl.until` are **terminators with successor operands**
  (`trueDest`/`falseDest`). Dialect conversion visits region parents before their
  children (pre-order), so a loop lowering must never feed a region-internal value
  into a newly created *legal* op — the framework would materialize a cast back to
  the unconverted type. Instead the loop lowering moves the terminator into the new
  block, sets its successors, and dedicated `ConditionOpLowering`/`UntilOpLowering`
  patterns create the `cf.cond_br` from the **adaptor** (properly converted i1).
- Loops lower to `cf` blocks directly (not scf), because scf has no break op in
  LLVM 17; `continue`/`exit`/`end` in loop bodies are mapped by
  `lowerLoopTerminators`.
- Arrays lower to memref + flat row-major offset (lower bound subtracted), computed
  in `index` type (`arith.index_cast`, not `extsi` — extsi rejects index results).
- FB/DB/struct memory model: an instance DB or struct is a **flat byte buffer**
  `memref<Nxi8>`; the FB layout (member name -> byte offset/size/type) is computed by
  `computeFbLayouts` in `runOnOperation` *before* conversion (ops are erased during
  conversion) and cached in `SclTypeConverter::fbLayouts`. `scl.function_block` ->
  `func.func(memref<Nxi8>)`, `scl.data_block` of FB type -> `memref.global` **with a
  zero ElementsAttr initializer** (a global without initializer lowers to an
  `external` LLVM global and the JIT cannot resolve its symbol), `scl.call.fb` ->
  `func.call`, `scl.var` in an FB is erased (declaration only).
- Member addresses (`scl.getvariable`/`scl.getelement`) are **`!scl.address<!scl.char>`**
  in MLIRGen; `SclTypeConverter` maps `address<char>` to a **rank-1 byte view**
  `memref<?xi8, strided<[1], offset: ?>>` created by a *single*
  `memref.reinterpret_cast` from the base buffer at the member's byte offset.
  **Never chain reinterpret_casts**: `ReinterpretCastOp::fold` rewrites
  `reinterpret_cast(reinterpret_cast(x))` -> `reinterpret_cast(x)` keeping only the
  outer offset, silently dropping the member offset. Loads/stores of members
  byte-assemble directly from that view (`loadFromByteBuffer`/`storeToByteBuffer`
  index `view[%i]`); loads use the *declared member type*, not the i8 address type.
- `case` -> new `scl.case` op (selector + per-region `array<i32>` case values +
  variadic regions, last = else) -> `cf.switch`. `goto`/labels -> real blocks +
  `cf.br` emitted in MLIRGen (forward gotos create the label block on demand; code
  after a goto lands in a fresh unreachable block). OB -> `func.func` returning void.
- `--emit=mlir-std` always runs canonicalizer+CSE after LowerToStd, so
  single-predecessor blocks (e.g. a forward goto) are folded away; test irreducible
  CFGs (backward gotos) if you want `cf.br` to survive in CHECK lines.

## Known gaps (parsed but not lowered / not code-generated)

- Composite members (arrays/structs as FB or struct members): their member address
  is an i8 byte view, so `getindex`/nested element access on them is not lowered
  correctly yet. Scalar members (int/real/bool/char/date/time) work end to end,
  including JIT (test/sclang/jit_fb.scl).
- FB member initializers (`var state : int := 5` inside an FB) are dropped —
  the IDB global is zero-initialized, so only `:= 0` is correct today.
- Strings (`scl.string` constants / string vars) are not implemented.
- `case` ranges (`1..5`) and CASE on non-integer selectors are not supported;
  only literal value lists on int selectors.
- Nested `continue`/`exit` inside `if` regions within loops are not remapped
  (only terminators directly in loop-body blocks are).
- `goto` following another terminator is a compile error by design.

## Repo hygiene

- Commit message format (always):
  ```
  git commit -m "<message>"

  Generated by Mistral Vibe.
  Co-Authored-By: Mistral Vibe <vibe@mistral.ai>
  ```
- Do NOT `git add -A`: untracked and not meant to be committed (yet):
  `sclang-opt/`, `sclang-translate/`, `cmake/`, `REFERENCES.md`, `.clang-format`.
  `.gitignore` covers `build.*`, `*.swp`, `.DS_Store`, `.ijwb/`, `xcode/`, `bazel-*`.
- History on `main` was rewritten once (filter-branch removed accidentally committed
  .swp files), so local hashes differ from any previously pushed state; pushing
  will need a force-push (nothing was pushed after the rewrite).
- Session notes from the initial review: scratchpad `sclang-review-notes.md`.
