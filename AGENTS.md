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
  `%T`/`%C` are accepted; the sim runtime implements IEC timer/counter
  semantics (phase 3): `scl_rt_timer_start(i32 t, i8 mode, i64 durS5)` and
  `scl_rt_counter_op(i32 c, i8 op, i64 val)` (ABI v2 — appended to
  `SclRtVTable`, NULL-able for older host libs). T/C reads return the timer
  value in ms (remaining while running, preset when stopped; writing `%T`
  sets the preset) / counter value; bit-0 reads return status (timer Q /
  counter>0). AWL side: `SP/SE/SS/SV/SI T n`, `ZV/ZR/S/R C n`, `S5T#1S`
  literals (encoded (base<<12)|BCD; `S5T#vMS` = v*10ms), `L T0` = value
  (width 16), `U T0` = status (width 1). The JIT driver runs
  `--jit-cycles=N` OB1 scans with `scl_rt_cycle(--cycle-ms)` between them
  (drives timer expiry). Deviations: `L Tn` returns plain ms (not
  S5TIME-BCD); timer Q for SS = expired, else = running.

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
- 89 tests currently pass (SclGen/Parse, SclGen/MLIR, SclTransforms, AwlGen,
  sclang JIT), incl. direct_address (Parse), direct (MLIR), direct + big_endian
  (SclTransforms), AwlGen ops/lower/areas/err_label/roundtrip/roundtrip_timer/
  timer/timer_lower/call_fb/cc, to_awl + to_awl_areas + to_awl_fb (SclTransforms),
  jit_areas + jit_tc + jit_awl (.scl/.awl cross-check) + jit_awl_backend
  (SCL->AWL->JIT) + jit_timer (AWL timers/counters via --jit-cycles/--cycle-ms)
  + jit_fb_awl (CALL FB n, DB m via --dump-areas) + jit_cc (CC via
  --dump-areas) + jit_string (string compare/copy via --dump-areas) in
  sclang; string_compare in SclGen/MLIR.

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
  zero-padded. String comparison (`s1 = s2`, `<>`, and lexicographic
  `</<=/>/>=`, also against a literal) is an unrolled byte-wise compare
  (AND of cmpeq / OR of cmpne, prefix-eq chain for ordering) in MLIRGen
  (`mlirGenStringCompare`); string var-to-var and whole-array assignment
  (`s1 := s2`, `a := b`) is an element-wise copy loop. Lexer pitfall fixed:
  `getStringValue()` returns a StringRef into the lexer's reused buffer —
  ParseStringLiteralExpression must copy it BEFORE `consume()` (an identifier
  token after the literal clobbered it: 'hi' became 'th'). Case labels
  support ranges (`1..3:` expands in the parser); `scl.case` lowers to
  `cf.switch` for any integer selector width.
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
- **SCL→AWL text backend (docs/design/scl-to-awl.md, implemented)**:
  `sclang-gen --emit=awl` runs `createLowerToAWLPass()`
  (`lib/SclTransforms/LowerToAWL.cpp`): each OB/FB/FC-n `scl.function` becomes
  an `awl.block`; locals are allocated in the M area (bit-packed bools),
  `%`-directs stay direct, expressions are emitted in accumulator order
  (operands materialized recursively), if/while/repeat/for/case/goto become
  labels+jumps, bool logic becomes rlo chains (temps for computed operands).
  `_mlir_ciface_*` JIT wrappers and data blocks are dropped. The result is
  printed as S7 text by `awl::printAwlText` (`lib/AwlDialect/Printer.cpp`),
  which simulates the ACC stack/RLO and reconstructs `A(`/`O(` brackets.
  Verified by awl→awl round-trip stability and by
  `test/sclang/jit_awl_backend.scl` (SCL→AWL→JIT == SCL→JIT memory writes).
  Printer pitfalls: i1 consts must print unsigned (`L 1`, not `L -1`);
  rlo-init `const true` (incl. before `=`) is not printed; `awl.cast` is
  implicit in the mnemonic letter; the CASE selector is re-loaded per
  compare (AWL `L` shifts ACC1 into ACC2).
  AWL parser pitfall: a label branch in the instruction loop must `continue`,
  not fall through to parseInstruction (two consecutive labels misparse).
  AWL backend gaps: FC params, FB arrays/structs/strings, cross-width
  assignment, DB initializers. (CALL FB n, DB m, CC and timers are done —
  see below.)
- **CALL FB n, DB m + FB instance DBs (implemented)**: AWL `CALL FB n, DB m`
  / `UC FB n, DB m` -> `awl.call @FBn, db m`; an FB block lowers to
  `func.func @FBn(i32 %arg0)` where arg0 = the instance DB number, and the
  call site passes the db attr as `%c_m_i32`. Inside an FB, a DB operand
  **without a number** (`DB.DBW0`, db=0) means the instance DB and lowers to
  `scl_rt_read/write('D', %arg0, …)` via the `directDbValue` helper in
  LowerToStd.cpp (reads the func arg when the direct has area DB, db 0, and
  the enclosing func name starts with "FB" and takes 1 arg; null-guarded).
  In the SCL->AWL backend, `scl.call.fb @fb(%idb)` becomes `awl.call` with the
  callee uppercased (`FB1`) and the db number resolved from the IDB symbol's
  data block (names like `DB m` keep m, others get sequential numbers from 1);
  FB member accesses (`scl.getvariable` on an IDB) become `DB.DBW<byteOff>`
  directs at the byte offsets of the precomputed per-FB layout
  (`AwlFbLayout`, keyed by UPPERCASED fb name). SCL symbol names are lowercase
  but the AWL printer expects `FB`/`FC` prefixes uppercase — uppercase when
  creating the callee attr. The sim runtime owns DB storage, so
  `--dump-areas` shows instance DBs (`DB2:[1]=2`) and host integration stays
  uniform. Tests: `test/AwlGen/call_fb.awl` (parse/lower/round-trip),
  `test/sclang/jit_fb_awl.awl` (JIT), `test/SclTransforms/to_awl_fb.scl`.
  FileCheck pitfall: `CHECK-LABEL: FB 1` also matches inside the line
  `CALL FB 1, DB 2` — anchor block labels with `{{^}}FB 1`.
- **CC (conditional call, implemented)**: `CC FB n, DB m` / `CC FC n` parses
  to `awl.call` with an optional `rlo` operand (the current RLO SSA value;
  CC leaves RLO unchanged, CALL/UC reset the chain). In AwlBlockOpLowering a
  call with an rlo is a cut op like a jump: an empty head block is split in
  front of the call block and gets `cf.cond_br(rlo, callBlock, fall)`; the
  call itself stays in callBlock for AwlCallOpLowering. The printer emits
  `CC FB n, DB m` (and counts a CallOp rlo use as an rlo-chain use so the
  neutral const-true is still not printed). `awl.call` has NO variadic
  argument operand anymore (FC params unsupported): ODS rejects
  Variadic+Optional in one op without segment sizes, so the only operand is
  the optional rlo; printed form is `awl.call @FB1, db 2` (no `()` suffix).
  Tests: `test/AwlGen/cc.awl` (parse/lower/round-trip), `test/sclang/jit_cc.awl`.
  Canonicalizer note: `L 1; L 1; ==I; CC …` folds the cmpeq to a constant and
  the dead call disappears — use M-area loads in tests to keep cond_br.
  C++ pitfall hit: `llvm::StringRef::upper()` returns a temporary
  `std::string` — never bind `StringRef x = s.upper()` (dangling); go through
  a `std::string` local.
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

AWL known gaps (both directions): CALL DB m, FC parameter passing. (IEC
timer instructions SP/SE/SS/SV/SI, counters ZV/ZR, CALL FB n / DB m with FB
instance DBs, and CC conditional calls are implemented — see the bullets
above.)

Memory areas `%I`/`%Q`/`%M`/`%DB…`/`%T`/`%C` are implemented (see above):
`!scl.direct` type, `scl.getdirect`, `%`-address parsing, lowering to the
`scl_rt_*` C ABI, sim runtime + JIT binding, host binding via
`--runtime=host --runtime-lib=…`, OB scan-cycle hooks, big-endian.

Remaining technical gaps:
- SCL-side timer FBs (TON/TOF/TP): CALL FB now works (see above), but the
  TON/TOF/TP semantics themselves are still only modeled via raw %T/%C areas
  + AWL timer instructions.
- Array-of-struct members (`a[i].x`): the byte-view chain has a dynamic offset
  at the inner cast, which `resolveByteViewRoot` cannot fold statically.
- String literals outside assignments/comparisons are not implemented.
  Strings are zero-padded (SCL pads with spaces — documented deviation).
- Non-constant FB member initializers are ignored (the IDB global is zeroed).
- CASE on real/bool selectors is rejected (integer selectors of any width work).
- `goto` following another terminator is a compile error by design.
- Whole-array var-to-var assignment now works for 1-d arrays with scalar
  element types (element-wise copy in MLIRGen); array-of-struct elements are
  still not copied.

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
