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
  converging at std IR, 004 IEC ST as a `--std=iec` mode of the SCL frontend
  (NO new dialect; LangMode threaded through Lexer/Parser/MLIRGen, dialect
  only gains LREAL f64 type + `scl.config` op).
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
- Source-language mode (ADR 004): `--std=s7` (default) / `--std=iec` on both
  `sclang` and `sclang-gen`, or `-x=st` (= `-x=scl --std=iec`). The mode is a
  `LangMode` (include/sclang/SclGen/LangMode.h) carried by the Lexer into the
  Parser and the ModuleAST, read by MLIRGen; the scl dialect stays
  mode-agnostic. IEC mode so far: PROGRAM aliases FUNCTION_BLOCK, bare
  `STRING` = `STRING[254]`, CONFIGURATION/RESOURCE/TASK + VAR_GLOBAL/
  VAR_EXTERNAL (VAR_GLOBAL -> global `scl.db` named after the configuration
  with init values baked big-endian; bindings kept as a `scl.config` metadata
  op, erased by LowerToStd; VAR_EXTERNAL resolves via `externalVars` into
  that DB — load with the declared type in mlirGenRValue, NOT the address
  element type). IEC keywords are only keywords in IEC mode. S7-only
  constructs are diagnosed in IEC mode and vice versa. fbNames is keyed
  UPPER-CASE (fbNameOrig keeps the original spelling; MLIR symbols are
  case-sensitive) — a user FB named e.g. "counter" wins over the
  TIMER/COUNTER builtin aliases in MLIRGen AND in LowerToStd's CallFbOp.
  INSERT semantics differ: IEC inserts AFTER 1-based P, S7 AT P (mode-gated
  in mlirGenStringBuiltin). The ST trace harness runs .st with -x=st and
  generates the task schedule (INTERVAL/PRIORITY/SINGLE) into the OB1
  wrapper for CONFIGURATION tests.
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
- 112 tests currently pass (SclGen/Parse, SclGen/MLIR, SclTransforms, AwlGen,
  sclang JIT, conformance), incl. direct_address (Parse), direct (MLIR), direct + big_endian
  (SclTransforms), AwlGen ops/lower/areas/err_label/roundtrip/roundtrip_timer/
  timer/timer_lower/call_fb/cc, to_awl + to_awl_areas + to_awl_fb (SclTransforms),
  jit_areas + jit_tc + jit_awl (.scl/.awl cross-check) + jit_awl_backend
  (SCL->AWL->JIT) + jit_timer (AWL timers/counters via --jit-cycles/--cycle-ms)
  + jit_fb_awl (CALL FB n, DB m via --dump-areas) + jit_cc (CC via
  --dump-areas) + jit_string (string compare/copy) + jit_case_bool +
  jit_case_real (CASE on bool/real selectors) + jit_array_of_struct
  (a[i].x) in sclang; string_compare in SclGen/MLIR; array_of_struct in
  SclTransforms (to_awl_fc), SclGen/MLIR (string_compare); jit_fb_init +
  jit_awl_fc (FC params via SCL->AWL->JIT) + jit_byte_trunc (int->byte
  truncation, direct and via AWL) + jit_char_lit (char := 'A') +
  jit_ton (builtin TON) in sclang, and test/conformance/conformance.mlir
  (external corpora runner — see "External conformance" below).
- External conformance: `test/conformance/run_conformance.py` compiles public
  SCL/AWL corpora (default root $SCLANG_CONFORMANCE_CORPORA or
  /Users/martin/scl-external; SKIPs when absent) and enforces pass-rate floors.
  Three tiers: compile (--emit=mlir), JIT smoke (--emit=jit --jit-cycles=2
  on OB1 files; rc 2 = __ASSERT oracles failed still counts as smoke pass),
  oracle (--emit=jit --jit-cycles=5 on OB1 files containing __ASSERT; pass
  iff rc == 0 — awlsim __ASSERT directives lower to awl.assert/assert_r ->
  scl_rt_assert_cmp[_r], Runtime ABI v4; malformed assert forms skip the
  line). sclang driver exits 2 when scl_rt_assert_failures() > 0 after the
  cycle loop.
  Current: AWL 116/116 (100%), SCL single 15/15, SCL per-project 17/17,
  JIT smoke 103/105 (98%), oracle (__ASSERT value-level, OB1 + bare
  instruction files) 70/71 (99%). Known-open oracle failure:
  startup_multiple (expects a 2-accu CPU; awlsim's own runner uses
  --twoaccu for it — permanent under our 4-accu default).
  dbpointer.awl line 99 (P#"Data".DTVAR1 quoted-DB-name extension) is
  also open; awlsim itself needs a symbol table for that file.
  ST/SCL trace tier: `test/conformance/run_st_traces.py` runs the 92
  s-celles IEC 61131-3 `.st`/`.trace` pairs through the SCL JIT engine and
  compares per-scan outputs (PROGRAM rewritten to FB + instance DB + OB1
  wrapper; driver flags --scan-io/--scan-inputs feed M-area slots per scan,
  see test/sclang/jit_scan_io.scl; --cross diffs SCL->JIT vs SCL->AWL->JIT).
  Current: 75/75 pass (100%), 17 unsupported (CONFIGURATION/LREAL/STRING/...).
  Frontend additions (committed): enum types (`TYPE C : (RED, GREEN);` ->
  INT storage + named constants resolved in codegen incl. CASE labels),
  subrange types (`INT (5..10)` -> base type, unchecked), multi-declaration
  TYPE blocks, array literals `[...]` incl. `n(v)` repetition, nested/2-D and
  struct-element `(x := 1, ...)` literals (FB members folded to a big-endian
  `scl.init_bits` cold-start attr consumed by the lowering's byte-exact
  baking; temp/local vars get per-element stores), IEC ed.3 METHODs
  (`instance.method(...)` -> FunctionOp `<FB>.<METHOD>` with $self first
  arg), implicit FB EN/ENO (EN gates the call, ENO stored in a per-instance
  cell hoisted to the function entry so `inst.ENO` / bare `ENO` in an FB
  read it back; outside FBs ENO reads FALSE as before). ENO is no longer
  lexed as a false literal. TIMER/COUNTER map to builtin TON/CTU unless the
  project declares its own FB under that name. bool<->int assignment now
  converts (zext/trunc). sclang-gen prints diagnostics (registered handler).
  FC calling convention (committed): VAR_IN_OUT/VAR_OUTPUT of FCs are
  by-address args; call sites reorder args into declared order, fill omitted
  inputs from declared defaults (zero if none), stage by-ref params through
  temp cells with copy-in/copy-out write-backs (write-back targets may be
  arbitrary lvalue paths, e.g. `DONE => "DB".x.y`); FunctionCallAST results
  are cached so argument expressions are never evaluated twice.
  Date/time (committed): TOD fraction lexing fixed (`.5` = 500 ms); IEC ed.3
  table 30 arithmetic expanded in MLIRGen into integer ops on the decoded
  values (DATE = days since 1990-01-01 u16; DT = 8 big-endian BCD bytes in an
  i64, S7 layout incl. weekday nibble; civil-calendar math via Hinnant's
  algorithms): DATE-DATE -> TIME ms, DT+/-TIME -> DT, DT-DT -> TIME ms;
  DT_TO_DATE/DT_TO_TOD/DT_YEAR..DT_WEEKDAY extractors; DATE/DT ordering
  flips the sign bit first (unsigned order). BCD byte extraction uses
  floor-div + mod fixup (signed div of the negative top byte corrupts).
  NOTE: never build IntegerAttr with an scl type (scl::IntegerType is not
  mlir::IntegerType) — use builder.getIntegerType(w) for the attr; the old
  pattern crashed the attr uniquer non-deterministically (APInt memcmp on a
  mismatched width) in DT code paths. lit: test/sclang/jit_date_time.scl.
  FUNCTION/OB/METHOD VAR sections (unlike VAR_TEMP) are default-initialised
  to the type's zero value at every invocation (ed.3 6.6.2.1: functions have
  no internal state) — mlirGenTempVar(decls, /*defaultInit=*/isVar).
  FOR loops re-read the TO bound on every iteration (ed.3 7.3.3.4.2 leaves
  modifying it undefined; the reference traces record the re-read behavior):
  ForDoOp's `to` operand is the ADDRESS of a $for_bound temp cell that the
  body re-stores at the end of each iteration, and the lowering reloads it
  in the cond block (AWL emitFor loads it from its direct too).
  Builtin TON/TOF/TP (scl_rt_timer_fb): the end-of-time test (elapsed >= PT)
  only applies when the timer was already running at the start of the
  evaluation (one-cycle lag at PT=0, as recorded in the impl-defined traces);
  TON uses the live PT for compare and ET clamp and latches Q once TRUE until
  IN falls; TOF keeps the standard power-up Q=FALSE (the timer_pt_zero trace's
  TRUE is a documented IronPLC deviation conflicting with conformant
  timers/tof). run_st_traces.py counts `status: implementation-defined`
  mismatches as pass-with-DIFF-note, not failures (the behavior is
  non-normative by definition).
  The ST tier is wired into run_conformance.py as a tier with floor 0.95
  (FLOOR_ST_TRACES; prints `ST traces (JIT)` and propagates FAIL/DIFF lines).
  --cross mode runs each passing wrapper SCL->AWL->JIT and requires identical
  scan lines: the wrapper is renamed to the AWL naming convention by
  awl_compat() (prog->FB1, inst->DB1, other FUNCTIONs->FCn), DB1 is sized via
  --area-size DB1=8192 (the AWL backend drops DATA_BLOCK declarations), and
  constructs the AWL backend cannot represent are cross-skips, not failures.
  Current cross: 46/75 identical; remaining diffs are AWL-backend gaps
  (builtin FB calls TON/CTU/R_TRIG inside SCL FBs, CASE expr, whole-struct
  expression materialization, ...).
  sclang-gen now registers a diagnostic handler (MLIRGen errors were
  previously swallowed, giving a silent rc=1).
  Builtin standard FBs (TON/TOF/TP/CTU/CTD/CTUD/R_TRIG/F_TRIG/SR/RS) are
  real now: forced instance layouts in the lowering's FbLayoutCache match
  scl_rt_timer_fb/counter_fb/edge_fb (runtime ABI v11); member access on a
  builtin-FB instance goes through scl.getelement's fb attr; local
  FB-typed instances work (instance-DB addresses are strided byte views).
  Update this and docs/conformance.md ("ST/SCL trace tier") when it changes.
  Fixed since last writing: runtime ACCU1..4 registers
  (awl.accu_get/accu_set, values now persist across labels and calls),
  which unblocked insn_4ACCU, the insn_JUMP SPZ/LOOP sections and
  stw_operators (= now mirrors OR=0/STA=VKE/NER=0 like awlsim
  insn_assign); SPN tests A1^A0 without state reset (awlsim insn_spn);
  LOOP reads ACCU1 from the runtime at the jump site; PUSH/POP are
  ACCU1<->ACCU2 copies (awlsim insn_push/insn_pop); RLDA/RRDA rotate
  through the A1 status bit (awlsim insn_rlda/insn_rrda); the AWL
  printer emits SPBN for jump-if-RLO==0 (SPN means A1^A0 in awlsim).
  Real arith is awlsim-exact too: scl_rt_arith_f (double math rounded to
  f32, *R inf special case, /R inf/zero-divisor cases, NaN = 0xFFFFFFFF)
  and scl_rt_realmath_f (sin/cos/tan/.../sqr with NaN->pNaN, epsilon
  snapping, STW classification incl. denormal-as-f32); __ASSERT==
  compares raw 32-bit patterns (awlsim asScalar); TRUNC/RND fallback
  keeps the original dword bits, RND is half-to-even. insn_ARITH_R now
  passes fully. __ASSERT sets NER=0 after comparing (awlsim
  insn_assert_*), __FEATURE pushes the feature value into ACCU1, and
  CALL SFC 46/-3 terminate the block (awlsim STOP/SHUTDOWN abort the
  scan) — this unblocked insn_CALL, insn_SE/insn_SS, sfb0-5, tc200
  array/pointer/udt, shutdown and startup_OB100/102. JIT-tier failures:
  the two Quine AWL files
  (intra-block jump loop timeout; awlsim needs a symbol table for those
  too). The JIT driver tolerates a missing OB 1 (FB-only files idle like
  awlsim), which fixed the zustimmtaster project invocation.
  When these numbers change, refresh them here AND in docs/conformance.md
  ("Current results") AND in the run_conformance.py header comment.
  String handling: `VAR := 'lit'` in DB-init/statement stores len @+1, chars
  @+2.., zero-fills to maxlen; STRING[n] size is (n+2) even-aligned;
  ARRAY OF STRING[n] keeps the dim in elemType; STRING[i] = char at +2+(i-1)
  width 8 (stringIndexed flag in resolveDbMember).
  The former LogMsg flake (~4% of runs, two distinct bugs) is FIXED:
  (1) intermittent `error: unknown element '...' in struct` —
  StructTypeStorage::operator== used `||` instead of `&&`, so uniquing
  could return a struct with matching names OR types but not both;
  (2) deterministic SIGSEGV in the parallel canonicalizer on
  `scl.call.fc` inside a `scl.function_block`: CallFcOp's callee lookup
  used lookupNearestSymbolFrom, which stops at the FB (a symbol table
  holding the FB's variables) and never sees module-level functions —
  the null result was only guarded by an assert (compiled out in
  release). The canonicalizer now looks the callee up in the module and
  bails out (failure()) when it is missing or has no named args (stub
  FCs). Related latent gaps fixed in the same area: func visibility is
  propagated when lowering scl.function(_block) to func.func (a public
  func declaration is rejected by the verifier), DB globals are always
  created with a plain identity-layout memref type (the LLVM lowering of
  memref.global requires identity maps) with memref.get_global +
  memref.cast producing the strided byte-window address type, and
  CallFbOp lowering reinterpret_casts the instance DB to the callee's
  parameter type (stub FBs have a zero-sized buffer). LogMsg now passes
  100/100 runs in --emit=mlir and 50/50 in mlir-std/llvm/jit.
  lit is 117/117 (check-sclang). OB-temp presets (OB1 header) are written at OB entry
  (scl_rt_ob_start); the JIT driver runs the startup OB (OB 102 on a
  4-accu CPU, else OB 100) before the OB 1 cycles; __FEATURE t [, v] is
  implemented (awl.feature op -> scl_rt_feature: nrAccus default 4,
  OB-temp presets flag).
  Corpora + licenses + results + gap list:
  `docs/conformance.md`.
- Spec traceability: every language feature carries a `// SPEC[<lang>]
  <spec ref> [(full|approx|gap)]` comment (lang = AWL/STL or SCL/ST) at its
  implementation site (AwlGen dispatch, AwlOps.td ops, lowerings, runtime
  fns, tests); a reason may follow the status (`(approx: reason)`) and
  still counts as that status, never as full. The status MUST be the
  trailing parenthesized token of the tag — an `(approx: ...)` in the middle
  of the text parses as full (put `-> where` before it). `docs/spec-conformance.md`
  contains a matrix GENERATED from
  those tags by `test/conformance/gen_spec_matrix.py` — after adding or
  changing a SPEC tag, ALWAYS regenerate it (`python3
  test/conformance/gen_spec_matrix.py`); the lit test
  `test/conformance/spec_matrix.mlir` (--check) fails check-sclang when the
  checked-in matrix is stale. Doc anchors: IEC 61131-3 (STL=AWL, ST=SCL),
  Siemens AWL/SCL manuals (links in README REFERENCES), awlsim as the
  executable reference semantics.
  - SCL const symbols in rvalue position fold via mlirGenConstSymbol
    (mlirGenVariable/mlirGenSymbol): int consts -> ConstantOp at the
    declared width, real/time consts -> the constant handlers, const
    expressions via foldConstIntAST, string consts warn + 0 (previously an
    unlinked operand / verification error, e.g. `stack.size_max :=
    STACK_SIZE` in Formelparser).
  - SCL forward declarations (FUNCTION x ... BEGIN END_FUNCTION in a
    separate file): mlirGenFunctionLike replaces an existing FunctionOp
    with the real definition, and skips a forward decl arriving after the
    definition (was: "redefinition of symbol" verification error).
  - String functions (LEN/CONCAT/LEFT/RIGHT/MID/FIND/DELETE/INSERT/REPLACE,
    <type>_TO_STRING, STRING_TO_<inttype>) are implemented inline in
    mlirGenStringBuiltin (lib/SclGen/MLIRGen.cpp): strings are flat
    zero-padded !scl.char buffers, length = index of first zero byte
    (C-string semantics), functions expand to unrolled byte loops and
    return a String[254] temp buffer ADDRESS; string-to-string assignment
    with different declared sizes copies the common prefix and zero-fills
    (truncating). STRING_TO_REAL/REAL_TO_STRING/VAL_STRG remain external
    stubs. Gotchas hit: runtime indices into a source buffer MUST be
    clamped (clampIdx) — the mux evaluates both sides, so an unclamped
    index reads OOB and segfaults the JIT; and optional args (P/L) are null
    mlir::Value for functions that don't take them — never build an op on
    them unconditionally (SubOp on null -> crash). Tests: SclGen/MLIR/
    string_builtins.scl (codegen), sclang/jit_strings.scl (execution).
  - __FEATURE 0 reports nrAccus = 4 (4-accu CPU default). startup_multiple
    expects 2 (awlsim runs it with --twoaccu) — permanent oracle failure;
    startup_OB100/OB102 expect 4 and pass.
  `check-sclang` includes `test/conformance` (added to test/CMakeLists.txt).

## Key design decisions (do not regress)

- IEC 61131-3 ST is a mode of the SCL frontend, not a dialect (ADR 004):
  `--std=iec` (default `--std=s7`, `-x=st` alias) sets a `LangMode` threaded
  through Lexer/Parser/MLIRGen. IEC-only constructs (CONFIGURATION/RESOURCE/
  TASK, VAR_GLOBAL/VAR_EXTERNAL, LREAL, bare STRING→STRING[254]) are parsed
  in IEC mode; S7-only ones (% directs, TIMER/COUNTER, EN/ENO, "DB".member)
  in S7 mode. VAR_GLOBAL lowers to a global DATA_BLOCK named after the
  CONFIGURATION; VAR_EXTERNAL resolves into it; RESOURCE/TASK bindings are
  kept as a `scl.config` metadata op (the trace harness generates the scan
  schedule from them). Do NOT create a parallel `st` dialect.
- AWL scalar semantics follow awlsim exactly: 8/16-bit values are UNSIGNED
  at their width (`AwlMemoryObject_asScalar` masks 0xFF/0xFFFF), so loads
  zero-extend into the 32-bit ACCU (`zextAcc`) and integer literals in the
  int16 range are zero-extended too (`parseConst` emits i32 `v & 0xFFFF`;
  `B#`/`W#`/`L#` typed literals carry `litWidth` 8/16/32). Assert oracles
  compare as u32 bit patterns (`assertCheck` in SimRuntime), matching
  awlsim's width-masked compare. DT#/DATE_AND_TIME is 8 BCD bytes (width-64
  direct), packed big-endian with awlsim's float-msec + weekday rules.
- `[ARn, P#x.y]` addressing is a TEMPORARY offset: `scl_rt_ar_add` must NOT
  write back into the AR register (except explicit `+ARn` which does).
- AR/pointer semantics (awlsim-exact): `LARn`/`TARn` P#x.y store the full S7
  32-bit pointer (area<<24|db<<16|byte<<3|bit); `+ARn off` adds only the low
  24 bits (byte*8+bit) — a bare `+ARn` adds ACCU1's low 24 bits (new
  `awl.ar_add` op takes a runtime i64 offset); bare `TAR` swaps AR1/AR2;
  `POP`/`LEAVE` = ACCU1 := ACCU2 (ACCU2 unchanged); `TAK` swaps ACCU1/2.
  `CALL FB n, DB m` sets AR2 = 0x84-packed (AREA_DB_S | base offset, db
  field 0 = current DI), NOT 0x85. The P# lexer must not treat whitespace
  before the byte number as a symbolic member (ptrSym) — skipWs before
  symBegin; and Token.quoteChar must be captured BEFORE lexer.next().
  Single-quoted literals 'AB' pack chars big-endian ('ABCD' = 0x41424344);
  double-quoted are names/symbols. DATA_BLOCK "Name" gets a stable DB number
  (namedDbs, from 100) so "Name".member resolves symbolically.
- awlsim status/logic details: after CALL/UC returns, OS=0 OR=0 STA=1 NER=0
  (OV preserved); CALL FB n,DB m saves+restores the DB/DI registers around
  the call (awlsim prevDb/prevDiRegister); CLR sets NER=0, so the next U/O
  computes from the neutral accumulator (rloFresh in AwlGen) — U x after CLR
  yields x, not 0. LC T n loads the timer value in S5TIME format
  (awl.timer_lc -> scl_rt_timer_lc, ABI v10 vtable entry appended at END).
  ANY headers for FB statics use area 0x84 + runtime DINO as DB word and
  bake the AR2 multi-instance base into the pointer (awlsim
  _trans_anyPointerFromDI); ARRAY operands set the element count as
  repetition factor. SFC 21 fails with E_DBNOTEXIST only when the DB is
  really not loaded (DB numbers are NOT limited to 0xFF — tests use DB 900).
- AR/accu semantics (awlsim insn_larN/insn_tarN): LARn bare = ARn := ACCU1,
  accus UNCHANGED (peekAcc, not popAcc); TARn bare = ACCU1 := ARn with the
  accus shifting down (accu2 := old accu1); bare TAR swaps AR1/AR2; TARn
  <operand> STORES ARn into the operand (opposite of LARn). ANY header
  pointer dwords use P#DBX style (db field 0; real DB in word @+4), and
  ".DBX"-style suffixes in P# literals are width keywords, not members.
  OUT params of runtime-called blocks are staged with their current value
  so a callee that leaves RET_VAL untouched copies the old value back.
  DI-area S7 pointers (area 0x85) decode to
  db=0xFFFF which resolves to the DI register; FB calls emit
  `scl_rt_set_current_di(db)` so `DIW…`/`P##var` work inside FBs.
- Array members track per-dimension bounds (`MemberInfo.dims/lo/elemSize`);
  `consumeArrayIndices` returns a BIT offset (ARRAY OF BOOL is bit-packed).
  STRING[n] members in a DATA_BLOCK get their max-len byte (byte 0)
  initialized to n in `__scl_db_init` (`emitStringMaxlenInits`, recursing
  into STRUCT/UDT/ARRAY members); a declaration-only DATA_BLOCK (no BEGIN
  section) is still sized and initialized via parseDbInitSection. BOOL
  RET_VAL occupies a full byte (bitCursor fix) so paramBytes > 0.
  Declaration initializers (`VAR : ARRAY[..] := 1, 2, 4 (…)`) are captured
  raw in `Layout.memberInit` and replayed per-element into `__scl_db_init`
  via a sub-lexer (the sub-lexer's input string must outlive it — dangling
  StringRef pitfall). CALL params with array RHS copy element-wise
  (`copyArrayParam`, incl. FC L-area `kLocalBase` offset and by-reference
  copy-back).
- `SPL` is a jump table (low byte of ACCU selects among following `SPA`
  labels, default = SPL operand); `BEB`/`BEC` lower to the new `awl.be_if`
  op (conditional block end); `BEU` = `BE`. S5T encoding picks the FINEST
  timebase covering the value (awlsim `Timer_seconds_to_s5t`), units in ms.
- Block-call model (awlsim semantics, all in `lib/AwlGen/AwlGen.cpp` +
  `lib/Runtime/SimRuntime.cpp` + `AwlCallOpLowering` in LowerToStd.cpp):
  - Every `awl.block` pushes an L frame (`scl_rt_l_push`) at entry and pops
    before every return. `bufferFor` adjusts `byteOff` IN PLACE for the L
    area (callers must index with the adjusted offset — the old bug was
    returning the buffer but indexing with the unadjusted offset). Frames
    advance by >= 2 bytes even when empty, so TEMP-less blocks still get a
    separate frame. l_push/l_pop also save/restore AR1/AR2 and the DB/DI
    registers (S7 restores them on block return).
  - FC interface params live in the CALLER's L frame at `kLocalBase`
    (8192, = runtime `kParamBase`), reached from inside the callee via the
    VL area (0x87 = parent frame; awlsim MEM_VL). The callee's `paramBytes`
    attr triggers `scl_rt_param_in` (after its l_push) / `scl_rt_param_out`
    (before l_pop), which copy the param area between the absolute MARKER
    staging area (call-site stores/read-backs, `kLocalBase` in M) and the
    caller frame (`lFrames.back() + kParamBase`). The callee accesses params
    as `!scl.direct<V, 0, kLocalBase+byte, ...>` (`DirectArea::VLocal = 'V'`);
    V and L share ONE runtime buffer (`areaBytes['L']`), only the frame base
    differs (`frameBaseOf`). L frames TILE the L-stack: `l_push` advances
    `lBase` by the CURRENT (caller's) frame size (`curFrameSize`), so a
    callee frame never overlaps the caller's param/VL-literal area. The
    callee's own `lframe` attr still covers `kLocalBase + paramBytes` so a
    nested callee's VL (= this frame) has room for its params.
    String literals for by-ref FC STRING params are copied into the caller's
    frame past the callee's whole param area (`blockVlLitOff/End`, grows
    `lframe`) with a `{0, 0x87|off}` header staged in M (awlsim
    `_FC_trans_copyToVLWithDBPtr`).
    FC interface layout = canonical order VAR_INPUT, VAR_OUTPUT (+ implicit
    RET_VAL right after OUT), VAR_IN_OUT, VAR — independent of source order
    (awlsim `blockinterface.getAllFields`). The pre-scan runs TWICE
    (`preScan()` clears and recollects) so FB/SFB-typed members
    (multi-instance vars) get the right size via `gFbLayouts`/`sfbLayoutOf`.
  - FB locals use DB-area directs with db = -1 (0xFFFF): the runtime maps
    that to the DI register AND adds the AR2 byte offset (awlsim cpu.py
    getArea_DI/setArea_DI). `CALL FB n, DB m` sets the DI register and,
    via the `instdb` attr, AR2 := P#DBX 0.0 (saved/restored around the call).
    `CALL #var` (multi-instance FB/SFB member) sets db = -1 + `ar2off` attr;
    AR2 += ar2off (relative, nests correctly) and DBR := DI (awlsim
    __call_MULTI_FB). SFB0-5 calls (`sfc` attr, callee SFBn) lower to
    `scl_rt_sfb(n)` in the runtime (counters CTU/CTD/CTUD, timers TON/TOF/TP
    at DI DB + AR2 base; layouts in `sfbLayoutOf`).
  - `UC/CC FC [MW x]` (indirect call) builds a jump-table dispatch over all
    FC/FB blocks known in the file (`__ind<n>_...` labels + `awl.call` per
    case), like SPL.
  - TIMER/COUNTER-typed block variables store the timer/counter NUMBER in
    the member's storage (FB: instance DB at mi->byte, db=-1; FC: L frame at
    kLocalBase+mi->byte). Accesses indirect through the T/C area at that
    number: parseOperand loads the number and returns an `awl.pack_area`
    dynamic i64 pointer (area 'T'/'C', byte = number); `L #cv` ->
    load_indirect, `U/X #cv` -> load_bit_indirect, ZV/ZR/S/R/FR/SP.. #tv ->
    CounterOp/TimerStartOp/TimerResetOp/FrOp with the i64 operand, lowered
    to `scl_rt_*_ptr` runtime fns (vtable v7). Note: awl.direct ops lower
    to packed i64 too, so the _ptr variants are used uniformly for T/C.
  - `P##name` pointers: FB interface -> area 0x85 (DI), FC interface -> 0x87
    (VL, parent L frame, at kLocalBase+byte), TEMP -> 0x86 (own frame).
    `P#DIX...` lexes as area 0x85, `P#V...` as 0x87. REAL-typed locals load
    as f32 (bitcast in
    `lowerDirectLoad`; `lastOperandType` tracks the member type).
    VL (0x87) pointers decode in `rawToPacked` with a 21-bit byte field
    (bits 3..23 — the DB field is unused for VL), because the FC param area
    starts at kLocalBase = 8192 > 8191 (13-bit S7 byte field max).
  - By-ref literal staging: a string or DT# literal for an FC by-ref
    compound param is copied into the caller's L frame past the callee's
    param area and a {0, 0x87|off} header staged (DT# packs byte0=year in
    the HIGH byte of the i64). ANY headers staged for CALL params map
    L-area r-values to VL (0x87); the same header staged for a runtime
    SFC/IEC call (emitRtCall) keeps L (0x86) — the runtime resolves it in
    the CURRENT frame (`toVl` flag on emitAnyParamHeader).
  - By-ref compound params (STRING[n]/DATE_AND_TIME/ANY — `isByRefType`)
    carry a pointer header in their slot: 6 bytes {db-nr word @+0, S7 ptr
    dword @+2}, 10 bytes for ANY. The header size is RESERVED in the layout
    in `parseStructBody` (declaration order!) — FC: all interface sections;
    FB: only VAR_IN_OUT (awlsim overrides just those with a DB-pointer).
    `paramBytes` therefore covers the headers; `param_in/out` copy them.
    Call-site staging: `emitPtrParamHeader` (L RHS -> 0x87 VL 21-bit byte
    field, db word 0; DB RHS -> 0x84 with db field 0, DB number in the word
    — awlsim drops the pointer db field for indirect access, uses the DB
    register; instance RHS -> 0x84 + DINO word), `emitAnyParamHeader`
    (10-byte ANY header). Callee-side `#param` / `#param[i]` on an FC
    compound param derefs the header: load ptr dword at VL slot+2, add the
    member offset in bits, return an i64 indirect address (STRING[i] is a
    1-based char at byte 2+(i-1), width 8). FB IN/OUT compound params are
    copied BY VALUE into the instance DB slot (byte-wise `copyBlockParam`,
    or the literal directly).
  - POINTER-typed params are passed by VALUE as a 6-byte DB-pointer header
    in the slot: P# literal RHS stages {db word, ptr dword db-field-0}
    directly; an addressable operand RHS (DB50.DBX 10.1, M 99.7) goes
    through `emitPtrParamHeader` like a by-ref param; OUT POINTER params
    (and POINTER RET_VAL) get the header staged before the call so the
    callee can write through it. `T #ptrParam` (ptrParamSlot flag from
    parseOperand) writes the header {db word, ptr dword} into the VL slot.
  - P# literal rules (awlsim): a P#DBn.../P#DIn... literal loaded as a
    32-bit value has db field 0 in the dword (the DB number lives only in
    the 48-bit DBPointer) — masked in `parseConst` for ptrLit tokens, which
    also resolve symbolic members via `consumePtrLiteral` (P#DB42.DTVAR1).
    `consumePtrLiteral` also resolves quoted DB names P#"Data".VAR via
    `namedDbs` or the CLI-registered `gCliNamedDbs` (--db-sym Name=DB,
    fed by the conformance harness from sibling .awlpro symbol tables,
    both XML-CDATA and legacy base64 INI formats).
  - `P#DBn.SYM[.SUB][[i]]` symbolic pointer literals: the lexer captures the
    member part in `Token.ptrSym` (+ `ptrTypeCode`/`ptrCount` for ANY
    suffixes like `P#M 10.0 BYTE 7`); `consumePtrLiteral()` resolves the
    byte/bit offset via `resolveDbMember` (sub-lexer over "DB<n>.<sym>").
    Used at all P# consumption sites (LAR/TAR/`+AR`/indirect offsets).
  - LARn/TARn semantics (awlsim insn_larN/insn_tarN): LARn <mem> loads the
    operand's 32-bit CONTENT as the pointer value (ar.set(fetch(op,32))),
    LARn P#x.y loads the literal, bare LARn takes ACCU1. TARn <mem> stores
    the AR as a RAW 32-bit S7 pointer (awl.ar_raw -> scl_rt_ar_raw), TARn
    LD/LW/LB uses the mnemonic width (LD=32). TARn ARm does ARm := ARn.
    `[LD n]`/`[LW n]`/`[#var]`/`[MD 4]` as an indirect base load the
    operand's VALUE as the base pointer; a raw P#x.y value (area byte 0)
    takes the ACCESS instruction's area byte (DIX -> 0x85, DBX -> 0x84,
    M -> 0x83, L -> 0x86 ...), T/Z [..] treat the value as the block NUMBER
    (awl.pack_area). An indirect RHS in a CALL param is loaded through
    (load_indirect), not passed as the address (lastOperandIndirect flag).
    AUF/OPN DI [LW 0] sets the DI register (awl.set_div ->
    scl_rt_set_current_di); AUF DB [LW 0] reads the L area (not M).
  - Call-depth guard: scl_rt_l_push exits(2) with a clear message at depth
    256 (awlsim specs.callStackSize default) — infinite recursion used to
    SIGSEGV the JIT (rc 139); rc 2 counts as a clean JIT-tier pass.
  - AR2 multi-instance base is added to DI-area accesses ONLY inside FB
    frames (awlsim __fetchDI/__storeDI check callStackTop.block.isFB):
    runtime state curIsFb, set per L-frame via scl_rt_frame_set_fb from
    the block's `fb` attr; ar2ByteOff() returns 0 outside FBs (an AR2 set
    by LAR2 in an OB/FC must not leak into DID/DBD offsets — pointer.awl
    FC2 DB3 OOB).
  - `+ARn P#x.y` (also `+ ARn P#x.y` after an arith operator) emits
    `awl.indirect` (ar_add) + `awl.ar_set` — ARn += offset in bits.
    LARn/TARn with no operand pop ACCU1 ZERO-extended to i64 (`zextTo64`):
    S7 pointers are unsigned (0x84010000 must not sign-extend).
  - ANY header type code/count follows awlsim `ANYPointer.makeByAutoType`:
    `anyTypeCodeOf` (type name -> S7 type code, INT=0x05 distinct from
    WORD=0x04) and `anyAutoType` (ARRAY with supported elem type ->
    (elem code, elem count); otherwise makeByTypeWidth(total bits):
    DWORD %32, WORD %16, BYTE %8, else BOOL). Symbolic P# literals carry
    the declared type/array info via `lastOperandType`/`lastPtrCount`/
    `lastPtrElemType`/`lastPtrElemBits` set by `resolveDbMember`.
  - FB/SFB VAR_IN ANY params (`isDbParam && section != 2`): the whole
    10-byte ANY header is copied into the instance-DB slot with
    `emitAnyParamHeader(..., toVl=false)` — NOT just the raw pointer
    dword (that was the anypointer.awl FB-section bug: header bytes read
    back as 0x83/0/0x50/...).
  - FC ARRAY params are by-ref too (awlsim: compound r-values get a
    6-byte DB-pointer header in the VL slot via _FC_trans_dbpointerInVL,
    regardless of the param's declared type): call site stages the header
    (byRef includes m->type == "ARRAY" for FC), the slot reserves
    max(array size, 6) bytes, and the callee derefs like other by-ref
    params. The callee deref uses the ROOT param's slot/type (saved as
    rootByRefParam/rootParamByte/rootParamType before descending into
    .SUB members — otherwise #ARR[1].SUB reads the header as data) and
    combines the header's DB-number word into the raw pointer ONLY for
    DB/DI-area headers (VL 0x87 bits 16-23 are the 21-bit byte field —
    kLocalBase=8192 needs them; masking them broke string literals).
    FB array params stay value-copied into the instance DB; forwarding a
    by-ref FC array param to an FB param (`VAR0 := #VAR00`) copies the
    elements through the deref'd pointer (copyArrayParamInd).
  - By-reference CALL params (`isByRefParam`: ANY/STRING/DATE_AND_TIME):
    ANY params stage a 10-byte ANY header (`emitAnyParamHeader`, incl.
    `P#... TYPE n` literals via ptrTypeCode/ptrCount); STRING/DT params
    stage a 6-byte POINTER (`emitPtrParamHeader`: DB-number word @+0, S7
    pointer dword @+2) which the callee reads via `P##param`
    (`L W [AR1, P#0.0]` = DB nr, `L D [AR1, P#2.0]` = ptr). Literal RHS
    (`callRhsAddressable`/`byRefRhs`) falls back to by-value. No read-back
    for by-ref OUT params (the callee writes through the pointer).
  - `CALL "NAME"` of a user block (namedBlockNums) routes through the
    common call path with the quoted symbol as callee and namedLayouts
    layout (goto common_call_path); runtime builtins win first
    (rtBuiltins: LEN/EQ_DT/GE_DT/DELETE/IEC + TIME_TCK->SFC64 -> emitRtCall,
    callee IECn with `iec` attr, lowered to `scl_rt_iec(n)` in the runtime;
    awlsim library/iec semantics: LEN = string act-len byte, EQ_DT = 8-byte
    compare, GE_DT = year-BCD-validated BCD field compare with 1900/2000
    year correction (FAIL leaves RET_VAL untouched), DELETE = substring
    removal with awlsim's clamping, negative L/P -> empty result).
    `awl.block` has a `fb` unit attr so NAMED FBs also get the i32
    instance-DB argument (lowering checks `startswith("FB") || getFb()`).
  - `awl.call` attrs: `callee`, optional `rlo` (CC), `db` (FB instance DB;
    -1 = current DI for multi-instance), `instdb` (unit: explicit instance
    DB -> AR2 set), `ar2off` (multi-instance base byte), `sfc` (SFB/SFC
    number), `iec` (IEC library FC number).
- UDT/inline-STRUCT handling: type names may contain a space (`TYPE UDT 1`);
  `parseStructBody` must compute `layout.size` even when it exits via an
  END keyword (goto done), must skip `STRUCT`/header-keyword lines, and
  inline `STRUCT` members get a nested Layout (`MemberInfo.nested` /
  `elemNested`) resolved in all dot-chains. Declaration initializers are
  collected recursively (`collectInitAssigns`, incl. per-element for arrays
  of UDTs) and replayed into `__scl_db_init` before the BEGIN section.
  `__STW <bit>` is a real bit operand (A0/A1 = bits 6/7); `LARn/TARn ARm`
  copies the other AR. The status word (STW) is fully modelled now
  (awlsim statusword bit numbering NER=0 VKE/RLO=1 STA=2 OR=3 OS=4 OV=5
  A0/CC0=6 A1/CC1=7 BIE/BR=8): `awl.stw_*` ops in AwlOps.td lowered to
  `scl_rt_stw_*` runtime calls (Runtime ABI v8/v9, SimRuntime `stw` field,
  reset at OB entry in scl_rt_ob_start). Comparisons record A0/A1/VKE
  (stw_cmp, VKE = predicate result), +,-,mul record sign + overflow
  (stw_sign, op+range operands so no extra arith ops appear in the IR),
  / records div-by-zero as UO+OV+OS (stw_div), TRUNC/RND/RND+/RND- go
  through awl.f2i -> scl_rt_f2i (out-of-range keeps ACCU, sets OV/OS),
  SAVE -> stw_set_bit(8, rlo), the VKE bit mirrors the RLO at the end of
  each logic statement, CALL SFC/SFB sets BIE=1 (IEC library FCs set BIE
  per their AWL body: success 1, error 0). RLO chain semantics: the RLO
  persists across statements, but a fresh U/O after a non-logic statement
  starts from the operand (awlsim NER=0 => VKE=STA; O-chains start from
  0, U/A from 1 — see parseRloLogic / stmtLogic). `U >=0`/`<=0` are !A0/!A1
  (so UO gives 0), `U >0`/`<0` are !A0&A1 / A0&!A1, UO is A0&A1.
  BIE/BR: set by the runtime in scl_rt_sfc/scl_rt_sfb/scl_rt_timer_fb
  (1 on success, 0 on error e.g. negative SFB PT or bad ANY magic in
  SFC 21) and by scl_rt_iec per the library body; OS (bit 4) is cleared
  at every awl.call entry (AwlCallOpLowering). P#DBX packs with area
  0x84 and db field 0 (current DB register), P#DIX with 0x85.

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
- `address<composite>` (array/IDB) converts **directly to the buffer memref**
  (identity layout; `memref.alloca` for locals, `memref.copy` for
  whole-buffer assignment of inputs). `address<struct>` converts to a
  **rank-1 i8 window** `memref<Nxi8, strided<[1], offset: ?>>` (static size N
  = byteSize of the struct), so that `a[i].x` on an array-of-structs chains
  through `resolveByteViewRoot` with a dynamic offset. An **array of
  composites** (array of struct/array) also converts to a flat i8 byte
  buffer, and GetIndexOpLowering produces a static-size window for composite
  elements (`makeByteView(..., staticSize=true)`); scalar elements keep the
  dynamic-size byte view. TempVariableOpLowering allocas the full window
  (`memref<Nxi8>`) for static-size rank-1 strided types; storeToByteBuffer
  casts the buffer to the window type before `memref.copy` when types differ.
  Do NOT wrap composite values in a 0-d `memref<memref<...>>` slot: a local
  array/struct var would load a descriptor from an uninitialized alloca and
  crash at runtime.
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
- Assignments insert an implicit `scl.int.cast` on WIDTH mismatch; byte/word/
  dword are LogicalType (not IntegerType), so the cast goes to an IntegerType
  of the same width — otherwise storing an int into a byte var writes 2 bytes
  into the 1-byte view and reads back the high byte (bug fixed; see
  jit_byte_trunc). storeToByteBuffer additionally truncates to the view's
  static size when known. `call.fc` results are typed from the callee's
  signature (pre-pass `functionResultTypes`), not hardcoded int.
  Boolean literals are i1 (`true`/`false`), and loop conditions go through
  `mlirGenRValue` so a bool variable condition is loaded, not passed as an address.
- Binary expressions with mixed-width integer operands (e.g. char + int,
  byte + word) now coerce both to an IntegerType of the wider width via
  `scl.int.cast` (see jit_mixed_width). `scl.int.cast` accepts LogicalType
  (byte/word/dword/bool) operands (ODS + areCastCompatible relaxed) and has
  an optional `zext` unit attr: LogicalType sources widen with ExtUI
  (unsigned — byte 200 + 100 = 300), IntegerType sources with ExtSI
  (signed int/dint). Lowering: IntegerCastOpLowering in LowerToStd.cpp.
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
  `cf.switch` for integer selectors of any width and bool selectors
  (`scl.case` accepts a LogicalType selector; the lowering ZERO-extends i1
  to i32 — extsi would turn true into -1 and never match case value 1);
  real selectors are lowered in MLIRGen to a nested if-chain of float
  cmpeq (`mlirGenRealCase`), since cf.switch needs integers. When emitting a
  nested if-chain manually, reset the insertion point to the else block
  before its createEndOp (recursion leaves the IP inside the nested if).
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
  (a single shared `identVal` is clobbered by the lookahead). `lexer.line()`
  is the SCANNED position (lookahead is ahead) — use `lexer.curLine()` (the
  current token's own line, stored in `Token.line`) for diagnostics/oracle
  ids. Base-16 typed literals accept a C-style `0x` prefix (`DW#16#0xFF…`).
  `convertScalarTo` now does real float width changes (ExtF/TruncF; it used
  to return the value unchanged for float->float, which broke f32->f64).
  do NOT
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
  AWL backend gaps: DB initial values (AWL text cannot express them; a
  warning is emitted and they are dropped — in the std path they ARE
  supported: `scl.db` carries an `init` dict attr (member -> i64) folded
  from the DB begin-section assignments by `foldConstIntAST` in MLIRGen
  and baked into the IDB global cold start in DataBlockOpLowering, see
  jit_db_init), dynamic array indices (AWL has
  no indirect addressing — constant indices are folded into the operand's
  byte offset). FB arrays/structs/strings ARE supported: composite FB
  members expose a byte direct at the member offset and
  getindex/getelement with constant indices fold into DB/M byte offsets
  (`widthOf` covers arrays/structs; `AwlFbMember` keeps the declared type;
  `declaredTypeOf` resolves it; verified by jit_awl_backend_fb_array).
  FC string arguments are copied byte-wise into the M param window slot
  (emitCallFc), and char (i8) values are promoted to i16 for
  arithmetic/comparison because S7 AWL has no byte mnemonic
  (jit_awl_backend_fc_string).
  (CALL FB n, DB m, CC, timers, FC params and cross-width assignment via
  awl.cast are done — see below.)
- **FC parameters in the SCL->AWL backend (implemented)**: passed through a
  reserved window at the top of the M area (the AWL analog of the S7 L
  stack): `kParamBaseByte = 65280`, 8 bytes per param; the FC return value
  uses `kResultByte = 65272`. The caller emits `L arg; T MW<slot>` per
  argument (mapped by name via `argNames` to the callee's `scl.name` arg
  order, collected in `fcSignatures` BEFORE any function is erased — the
  symbol lookup fails afterwards), then `UC FC n`, then loads the result
  slot. The FC's input tempvars map to the param slots (`paramDirects`),
  and the entry-block `store tempvar = blockarg` copies are skipped (the
  caller already stored them). An FC with a result stores its output var
  (tempvar named like the function) to the result slot before BE (insertion
  point must be reset before the trailing `awl.be`). A `call.fc` whose
  result is used is emitted by `materialize()` at the use site, not by
  emitOp (which only emits statement-level calls with unused results) —
  otherwise the call is emitted twice. AWL locals are allocated in M
  starting AFTER the bytes used by direct %M addresses in that function
  (nextBit initialized from a walk over GetDirectOp), so locals and %M
  directs don't collide. Known limitation: all blocks share the M area, so
  locals of different blocks alias each other (S7 uses a per-block L stack;
  our param window is the only reserved region).
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
    without a bit number is a parse error; `%T`/`%C` timers/counters are
    supported (`%TW1`/`%TD1`/`%CW1` word/dword forms work through the same
    `scl_rt_*` C ABI; the sim runtime owns the T/C areas).
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

AWL: shift/rotate instructions (SSI/SLD/SRD/RLD/RRD) and real math
(ABS/TAN and friends; RND* truncates) are NOT implemented — the awl dialect
has no shift/float-math ops (see docs/conformance.md; 7 corpus files fail on
this). Everything else the awlsim corpus uses is handled, incl. awlsim
extensions (headers, quoted names, `#locals`, `L/LD/LW`, symbolic operands,
indirect `[AR1, P#…]` addressing, S5T#/T#/D#/TOD# literals, CALL SFC47 /
CALL "name", …) — see test/AwlGen/awlsim-extensions.awl. CALL DB m is N/A —
S7 AWL has no CALL DB instruction (DBs are opened with OPN, which our
explicit `DBn.` operands make unnecessary). IEC timer instructions
SP/SE/SS/SV/SI (`awl.timer_start` + `scl_rt_timer_start`) and counters
ZV/ZR (`awl.counter` + `scl_rt_counter_op`), CALL FB n / DB m with FB
instance DBs, CC conditional calls, and FC params via the M param window
are all implemented — see the bullets above.

Memory areas `%I`/`%Q`/`%M`/`%DB…`/`%T`/`%C` are implemented (see above):
`!scl.direct` type, `scl.getdirect`, `%`-address parsing, lowering to the
`scl_rt_*` C ABI, sim runtime + JIT binding, host binding via
`--runtime=host --runtime-lib=…`, OB scan-cycle hooks, big-endian.

Remaining technical gaps:
- SCL-side timer FBs TON/TOF/TP work as builtins (see the builtin bullet
  above); raw %T/%C areas + AWL timer instructions also work.
- Array-of-struct members (`a[i].x`) work (see above).
- String literals work in assignments to a string var (byte copy,
  zero-padded) or to a char (first byte), and as named FC/FB call arguments
  (materialized into a temp buffer of the declared parameter type via
  `mlirGenStringLiteralValue`, using the `callableVarTypes` pre-pass; whole-
  array load/store through dynamic i8 byte views in LowerToStd).
  Strings are zero-padded (SCL pads with spaces — documented deviation).
- FB member initializers that are constant EXPRESSIONS (e.g. `40 + 2`) are
  folded by `foldConstBits` and baked into the IDB global's cold start;
  genuinely non-constant initializers are ignored (zeroed) but now emit a
  warning (MLIR drops warnings without a registered handler — `sclang` main()
  registers one that prints all diagnostics to stderr).
- CASE on real/bool selectors works (see above).
- `goto` following another terminator is a compile error by design.
- Whole-array var-to-var assignment works for 1-d arrays, including
  array-of-struct elements (copied through a char-array view of the flat i8
  buffer in MLIRGen; see `byteSizeOf`).
- FB-member array-of-struct elements (`pts[1].x` on an FB member) work in
  the std path: GetIndexOpLowering only emits a static-size window when the
  result is address<composite> (plain array variable); the byte-view path
  (arrayType attr set, result address<i8>) must stay a dynamic-size view or
  the conversion framework inserts an invalid DialectCastOp materialization
  (fixed; see jit_fb_array_of_struct).

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
