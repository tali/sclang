# AGENTS.md — working on sclang

MLIR-based SCL (Siemens Structured Control Language) compiler. Repo root is the
project root; branch `main`.

## Maintenance rule for this file

**Keep this file up to date at all times.** Whenever you learn something
important in a session — a design decision that must not regress, a build or
host quirk, a new pitfall, a change in test count or pipeline behavior, a new
known gap — add it to the appropriate section below *in the same session*,
automatically, without being asked. AGENTS.md is the memory between sessions:
if it is not written here, it is forgotten. Design rationale that is expensive
to reverse goes into an ADR under `docs/decisions/` (see below) and gets a
one-line pointer here.

## Documentation

- `docs/decisions/` — ADRs (Context/Decision/Consequences + Status/Date) for
  hard-to-reverse decisions: 001 big-endian storage (implemented), 002 runtime
  C ABI (sim vs host; sim implemented), 003 AWL as a parallel dialect
  converging at std IR.
- `docs/design/` — evolving design plans: memory-areas (phase 1 implemented),
  awl-frontend, scl-to-awl. Update these when the plan changes; add new ADRs
  when a decision becomes fixed.
- README.md is user-facing (status + build); do not put design detail there.
- In-code rationale lives next to the code: `.td` description fields, and the
  runtime ABI contract in the header comments of `include/sclang/Runtime/Runtime.h`.

## Architecture / pipeline

```
.scl ──> sclang-gen (SclGen: Lexer → Parser → AST → MLIRGen) ──> scl dialect ops
.awl ──> sclang-gen -x awl (AwlGen: Lexer → Parser → codegen) ──> awl dialect ops
      both ──> LowerToStd pass (lib/SclTransforms/LowerToStd.cpp, partial conversion)
      ──> func/arith/memref/scf/cf (+ calls to the runtime C ABI for % areas)
      ──> sclang (sclang/LowerToLLVM.cpp: scf→cf, memref→llvm, arith/func/cf→llvm)
      ──> LLVM IR / ExecutionEngine JIT (sim runtime bound via registerSymbols)
```

- `sclang-gen --emit=mlir` prints unlowered scl-dialect IR; `--emit=mlir-std` runs LowerToStd.
- `sclang --emit=jit` runs the full pipeline and JIT-executes. The JIT entry point is
  the `_mlir_ciface_OB1` wrapper (`engine->invoke("OB1")` resolves
  `_mlir_ciface_OB1`), which calls `function OB1` (see test/sclang/jit.scl) —
  JIT tests must define that wrapper.
- `--opt` enables canonicalizer/CSE between stages.
- Memory areas: the JIT binds `@scl_rt_*` to the sim runtime
  (`lib/Runtime/SimRuntime.cpp`) via `ExecutionEngine::registerSymbols`;
  `--area-size M=1024` / `--area-size DB1=4096` configure simulated area/DB
  sizes (default 65536 bytes for I/Q/M/T/C, OOB aborts; DBs grow lazily unless
  sized). `--runtime=host --runtime-lib=path.so` dlopens a host runtime
  exporting `scl_host_runtime_vtable()` and registers its vtable via
  `scl_rt_register` (LLVM 17: `DynamicLibrary::getPermanentLibrary` returns a
  value, check `isValid()`; `llvm::JITSymbolFlags`, not `llvm::orc::…`).
  `%T`/`%C` are accepted and treated as plain areas (IEC timer semantics and
  callers of `scl_rt_cycle` are a known gap).

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
  `test/AwlGen`, `test/sclang`; BUILD files glob `*.scl` (sclang dir also `*.awl`),
  so new files are picked up automatically. `test/lit.cfg.py` suffixes include
  `.awl`; the tools list includes `not` (negative tests need `not … | FileCheck`,
  lit's internal shell has pipefail so a bare failing pipeline fails the RUN).
- Run a subset via the *binary* test dirs:
  `cd build.release/tools/sclang/test && ../../bin/llvm-lit -sv AwlGen sclang`
  (running llvm-lit on `../test/...` directly misses the site-config tool
  substitutions). `ninja check-sclang` covers everything.
- RUN conventions: `sclang-gen --emit=mlir-std %s | FileCheck %s` for lowering tests,
  `sclang --opt --emit=jit %s 2>&1 | FileCheck %s` for JIT tests.
- SSA names print as `%alloca`, `%reinterpret_cast`, `%0`… — match with
  `%[a-z_0-9]+` / `%[0-9]+`, not literal names. FileCheck does NOT strip `//`
  comments from CHECK lines — never put `// ...` inside a CHECK pattern.
  Constants are hoisted to the top of the block by canonicalizer/CSE, so check
  `arith.constant` lines with CHECK-DAG, not ordered CHECK.
- 74 tests currently pass (SclGen/Parse, SclGen/MLIR, SclTransforms, AwlGen,
  sclang JIT), incl. direct_address (Parse), direct (MLIR), direct + big_endian
  (SclTransforms), AwlGen ops/lower/areas/err_label, jit_areas + jit_tc +
  jit_awl (.scl/.awl cross-check, identical JIT output) in sclang.

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
- **AWL frontend (ADR 003, implemented)**: `.awl` / `-x awl` in both drivers ->
  `sclang::awlGen` (`lib/AwlGen/AwlGen.cpp`) -> `awl` dialect (`awl.block @OB1`
  with flat ops ending in `awl.be`). The two accumulators are explicit SSA
  values in codegen: `awl.load` pushes, arith pops 2/pushes 1, RLO is an i1
  (`U/O/NOT/A(/O()`, `=`/`S`/`R`, jumps are `awl.jump*` with string labels).
  LowerToStd's `AwlBlockOpLowering` converts `awl.block` to `func.func` and
  splits the flat block at every label/jump/be (pass 1 registers labels, pass 2
  wires `cf.br`/`cf.cond_br`, pass 3 adds fallthroughs, `be` -> `func.return`).
  If `OB 1` exists, a `_mlir_ciface_OB1` wrapper calling `@OB1` is synthesized so
  the JIT entry convention is unchanged. Operand syntax `E/A/M/T/Z [B|W|D]
  byte[.bit]` and `DBn.DB[XBWD] byte[.bit]`; real literals `A 0.0` split into
  byte/bit. **Label = fresh statement**: at a label the acc stack and RLO are
  reset (next U/O/NOT uses the neutral value), otherwise the RLO SSA value from
  a predecessor block would not dominate the label's block after splitting.
  Pitfalls hit: lexer needs 2-token lookahead with per-token payload structs
  (a single shared `identVal` is clobbered by the lookahead); do NOT
  `module.push_back(op)` after creating an op with an insertion point inside the
  module — double insertion corrupts the ilist and `mlir::verify` loops forever
  (sample the hung process to see this).
- `--dump-areas` (sclang driver, sim runtime only): after `--emit=jit`, calls
  `scl_rt_sim_dump_areas()` (`lib/Runtime/SimRuntime.cpp`), which prints all
  non-zero area/DB bytes (`M[20]=1`, `DB1[3]=7`, `clock=…ms`) — makes memory
  writes observable in JIT tests (used by the SCL-vs-AWL cross-check
  `test/sclang/jit_awl.{scl,awl}`, which must produce identical output).
- **Big-endian storage (ADR 001, implemented)**: byte `i` of an N-byte value in a
  byte buffer holds bits `8*(N-1-i)..` (most significant byte first, S7 layout).
  Applied in `loadFromByteBuffer`/`storeToByteBuffer`, in the IDB cold-start
  initializer baking (`DataBlockOpLowering`), and in the sim runtime's
  `scl_rt_read`/`scl_rt_write`. Verified by `test/SclTransforms/big_endian.scl`
  and `test/sclang/jit_areas.scl`.
- **Direct addresses / memory areas (phase 1 of docs/design/memory-areas.md)**:
  - `!scl.direct<area, db, byte, bit, width>` type (area letters `I`/`Q`/`M`/`DB`/
    `T`/`C` as ASCII codes in `DirectArea`; width in bits 1/8/16/32) and the
    `scl.getdirect` op, printed `scl.getdirect M, 0, 10, 3, 1 :
    !scl.address<!scl.direct<M, 0, 10, 3, 1>>` (custom parser/printer in
    SclOps.cpp; verifier ties attrs to the result type). Load value type =
    bool for width 1, else integer of the width (`DirectType::getValueType`).
  - Lexer: `%…` is one `tok_direct_address` token (raw text without `%`);
    `Parser::ParseDirectAddressExpression` parses `%I0.0`, `%IB2`, `%IW4`,
    `%ID0`, `%DB1.DBX0.0`…`%DB1.DBD8` into a `DirectVariableAST`. Plain `%I0`
    without a bit number is a parse error; `%T`/`%C` parse but MLIRGen rejects
    them ("timers/counters not yet supported").
  - LowerToStd: `address<direct<…>>` converts to a packed-i64 placeholder
    (never used); `scl.load`/`scl.store` through a direct address lower to
    `@scl_rt_read/write[_bit](area: i8, db: i32, byteOff: i32, widthBytes|bit:
    i8[, value: i64])` — private `func.func` declarations are inserted at
    module start on demand (`getOrInsertRtFunc`). `scl.getdirect` lowers to an
    `arith.constant` i64 packing the fields.
  - OB scan cycle: `organization_block` sets the `scl.organization_block` unit
    attr on its `scl.function` (prints as `attributes {scl.organization_block}`);
    FunctionOpLowering emits `@scl_rt_update_inputs()` at entry and
    `@scl_rt_update_outputs()` before every `scl.return`.
  - Sim runtime: `include/sclang/Runtime/Runtime.h` (fixed C ABI, ADR 002) +
    `lib/Runtime/SimRuntime.cpp` (CMake target `SclRuntime`). The driver binds
    the JIT symbols with `ExecutionEngine::registerSymbols` — in LLVM 17 that
    is `llvm::JITSymbolFlags` (NOT `llvm::orc::JITSymbolFlags`) and
    `llvm::orc::ExecutorAddr::fromPtr`. Host integration: `scl_rt_register()`
    installs an `SclRtVTable` that the sim forwards to.

## Known gaps (parsed but not lowered / not code-generated)

Planned next features (designs in `docs/design/`, ADRs in `docs/decisions/`):
the SCL→AWL text backend (`docs/design/scl-to-awl.md`). The AWL frontend is
implemented (see above).

AWL frontend known gaps: CC (conditional call), CALL FB n / CALL DB m,
IEC timer instructions (SD/SI/…), FC parameter passing.

Memory areas `%I`/`%Q`/`%M`/`%DB…`/`%T`/`%C` are implemented (see above):
`!scl.direct` type, `scl.getdirect`, `%`-address parsing, lowering to the
`scl_rt_*` C ABI, sim runtime + JIT binding, host binding via
`--runtime=host --runtime-lib=…`, OB scan-cycle hooks, big-endian.

Remaining technical gaps:
- IEC timer semantics (S5TIME encoding, timer start/reset) and anything
  calling `scl_rt_cycle` — `%T`/`%C` are plain byte areas in the sim.

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
