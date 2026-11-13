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
- 62 tests currently pass (SclGen/Parse, SclGen/MLIR, SclTransforms, sclang JIT).

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
  `memref<Nxi8>`; the FB layout (member name -> byte offset/size/type + optional
  constant cold-start init bits) is computed by `computeFbLayouts` in
  `runOnOperation` *before* conversion (ops are erased during conversion) and
  cached in `SclTypeConverter::fbLayouts`. `scl.function_block` ->
  `func.func(memref<Nxi8>)`, `scl.data_block` of FB type -> `memref.global` whose
  **ElementsAttr initializer bakes in the FB members' cold-start values** (a
  global without initializer lowers to an `external` LLVM global and the JIT
  cannot resolve its symbol; initializers must NOT be emitted in the FB body —
  that would re-initialize on every call). `scl.call.fb` -> `func.call`,
  `scl.var` in an FB is erased (declaration only).
- `address<composite>` (array/struct/IDB) converts **directly to the buffer
  memref** (identity layout; `memref.alloca` for locals, `memref.copy` for
  whole-buffer assignment of inputs). Do NOT wrap composite values in a 0-d
  `memref<memref<...>>` slot: a local array/struct var would load a descriptor
  from an uninitialized alloca and crash at runtime.
- Scalar FB/struct member addresses (`scl.getvariable`/`scl.getelement`) are
  **`!scl.address<!scl.char>`** in MLIRGen; `SclTypeConverter` maps `address<char>`
  to a **rank-1 byte view** `memref<?xi8, strided<[1], offset: ?>>`. MLIRGen tracks
  `byteViewTypes` (byte-view value -> declared SCL member type) so loads use the
  declared member type; `getindex`/`getelement` on a byte view carry the declared
  array/struct type in the `arrayType`/`baseType` attrs. Lowerings resolve any
  view chain with `resolveByteViewRoot` and create **one** cast from the root
  buffer. **Never chain reinterpret_casts**: `ReinterpretCastOp::fold` rewrites
  `reinterpret_cast(reinterpret_cast(x))` -> `reinterpret_cast(x)` keeping only the
  outer offset. Loads/stores byte-assemble from the view directly
  (`loadFromByteBuffer`/`storeToByteBuffer` index `view[%i]`; i1 bool members are
  special-cased — `extui`/`trunci` between i8 and i1 are invalid).
- `continue`/`exit` (also nested inside if/case in a loop body) lower via
  `ContinueOpLowering`/`ExitOpLowering` to branches recorded in
  `LoopTerminatorTargets` by the loop lowerings (dialect conversion is pre-order,
  so loop patterns run before the nested ops; rewriting/erasing nested ops inside
  the loop pattern does NOT work — deferred erasure skips them and direct
  `erase()` dangles the worklist).
- Assignments insert an implicit `scl.int.cast` between integer types (target =
  declared member type for byte-view lvalues). `call.fc` results are typed from
  the callee's signature (pre-pass `functionResultTypes`), not hardcoded int.
  Boolean literals are i1 (`true`/`false`), and loop conditions go through
  `mlirGenRValue` so a bool variable condition is loaded, not passed as an address.
- `string[n]` is an array of n chars; `s := 'lit'` stores the literal bytes
  zero-padded. Case labels support ranges (`1..3:` expands in the parser);
  `scl.case` lowers to `cf.switch` for any integer selector width.
- `case` -> new `scl.case` op (selector + per-region `array<i32>` case values +
  variadic regions, last = else) -> `cf.switch`. `goto`/labels -> real blocks +
  `cf.br` emitted in MLIRGen (forward gotos create the label block on demand; code
  after a goto lands in a fresh unreachable block). OB -> `func.func` returning void.
- `--emit=mlir-std` always runs canonicalizer+CSE after LowerToStd, so
  single-predecessor blocks (e.g. a forward goto) are folded away; test irreducible
  CFGs (backward gotos) if you want `cf.br` to survive in CHECK lines.

## Known gaps (parsed but not lowered / not code-generated)

- Array-of-struct members (`a[i].x`): the byte-view chain has a dynamic offset
  at the inner cast, which `resolveByteViewRoot` cannot fold statically.
- String comparison, string var-to-var assignment, and string literals outside
  assignments are not implemented. Strings are zero-padded (SCL pads with
  spaces).
- Non-constant FB member initializers are ignored (the IDB global is zeroed).
- CASE on real/bool selectors is rejected (integer selectors of any width work).
- `goto` following another terminator is a compile error by design.
- Whole-array var-to-var assignment (`a := b`) is not lowered (only
  whole-buffer *input argument* copies via `memref.copy`).

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
