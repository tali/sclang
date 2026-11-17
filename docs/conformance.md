# External conformance testing

For *which* specifications this is tested and how to trace a spec item to
code and tests, see [spec-conformance.md](spec-conformance.md) (generated
from `SPEC[...]` tags in the sources).

Publicly available SCL/AWL sources are used as a conformance corpus to test
the compiler end to end. The corpora are **not** committed to this repo
(licenses); they live outside the tree and the test suite skips gracefully
when they are absent.

## Running

```
python3 test/conformance/run_conformance.py --sclang build.release/bin/sclang
# or via lit (runs in `ninja check-sclang`):
#   test/conformance/conformance.mlir
```

The corpora root defaults to `$SCLANG_CONFORMANCE_CORPORA` or
`/Users/martin/scl-external`. When the root is missing the runner prints
`SKIP` and exits 0; when present it enforces minimum pass-rate floors
(currently AWL >= 0.90, SCL single >= 0.55, SCL projects >= 0.60, see the
constants in the script) and exits 1 on regression.

What is run (two tiers):

- **compile tier** (`--emit=mlir`, frontend + MLIR verifier only):
  - every `*.awl`: dialect sniffed per file (awlsim project markers
    `ORGANIZATION_BLOCK` / quoted block names / `DATA_BLOCK` => `-x=awl`,
    SCL markers `FUNCTION` / unquoted `DATA_BLOCK` / `VAR_INPUT|OUTPUT|IN_OUT`
    => `-x=scl`);
  - every `*.scl` (except `makefile.scl`) individually with `-x=scl`;
  - every directory of `.scl` files concatenated (newline-separated) as one
    project with `-x=scl` (TIA exports are multi-file);
- **JIT smoke tier** (`--emit=jit --jit-cycles=2`) on every file defining
  OB 1: the full pipeline — LowerToStd, LowerToLLVM, ExecutionEngine — and
  two scan cycles of `_mlir_ciface_OB1`. This is a *smoke test*: it proves
  the program lowers and executes without crashing or missing symbols.
  Exit code 2 (awlsim `__ASSERT` oracles failed, see below) still counts as
  a smoke pass — it proves lowering and execution worked.
- **Oracle tier** (`--emit=jit --jit-cycles=5`) on every OB 1 file that
  contains awlsim `__ASSERT<cmp>[R] lhs, rhs` directives: the frontend
  lowers them to `awl.assert` / `awl.assert_r` runtime oracle calls
  (`scl_rt_assert_cmp[_r]`, Runtime ABI v4), evaluated at every scan cycle
  exactly where awlsim evaluates them. The tier passes iff the driver exits
  0, i.e. every executed oracle held — this checks *computed values*
  against the expectations embedded in the publicly available test files.
  `__STW`/status-bit asserts (`BIE`, `OV`, `A0`, `__STW 0`, `==0`/`UO`/…
  conditions) are evaluated: the sim models the full status word
  (awlsim statusword bits; see AGENTS.md — awl.stw_* ops -> scl_rt_stw_*
  runtime calls, ABI v8/v9). Malformed/unknown assert forms skip the line
  so they can never break the compile tier.

To make the JIT tier work on single files of multi-file projects,
`awl.call` to a callee not defined in the module now lowers to an empty stub
function (documented approximation; system blocks SFC/SFB included).

## Corpora (cloned into /Users/martin/scl-external, licenses as noted)

| directory | contents | license |
|---|---|---|
| `awlsim-test` (github.com/mbuesch/awlsim `tests/`) | ~102 AWL/awlsim-project tests: instruction set (`tc100_instructions`), datatypes, system blocks, projects (`tc999_projects` incl. shiftregister, ringbuffer, trafficlights, acme-*), legacy fileformat-v0 | GPL |
| `lopez-dev-Siemens-SCL-Source-Files` | 15 TIA Portal SCL exports | repo license |
| `thomas-v2-S7-PLC-Snippets` | SCL multi-file project (Formelparser), TIA Put/Get OUC, Quine AWL | repo license |
| `zjx195-TIA-Portal-SCL`, `LeoSunSET-...-Algorithms` | more SCL/AWL samples | repo license |
| `tallakt-awltool`, `PLC-lang-rusty` | AWL tooling | various |
| `s-celles-iec61131-conformance` | 92 IEC 61131-3 ST conformance programs, each with a `.trace` file of per-scan expected outputs (CC0) — ground truth for the ST/SCL trace tier below | CC0 |

Totals: 22 `.scl` files (17 project dirs), 116 `.awl` files.

## Current results (LLVM 17 build, this writing)

- AWL compile: **116/116 pass (100%)**
- SCL single file: **15/15 pass (100%)**
- SCL per project: **17/17 pass (100%)**
- JIT smoke (files with OB 1, full pipeline + execution): **103/105 pass (98%)**
  — failures are the two Quine AWL snippets (intra-block jump loop, timeout;
  awlsim itself needs a symbol table for those). The JIT driver tolerates a
  missing OB 1 (FB-only files idle, like awlsim).
- Oracle (`__ASSERT` files with OB 1 or bare instruction lists, value-level,
  exit code must be 0): **70/71 pass (99%)** — see "Oracle tier" below.
  Known-open: startup_multiple (expects a 2-accu CPU; awlsim's own runner
  uses --twoaccu for it — permanent under our 4-accu default).
  Fixed since last writing: runtime ACCU1..4 registers
  (awl.accu_get/accu_set — accumulators persist across labels/calls),
  which unblocked insn_4ACCU, the insn_JUMP SPZ/LOOP sections and
  stw_operators; plus awlsim-exact SPN (A1^A0, no state reset), LOOP
  reading ACCU1 at the jump site, `=` mirroring OR=0/STA=VKE/NER=0,
  PUSH/POP as ACCU1<->ACCU2 copies, RLDA/RRDA rotating through A1, and
  the AWL printer emitting SPBN for jump-if-RLO==0, and awlsim-exact
  real arith (scl_rt_arith_f / scl_rt_realmath_f in the sim runtime:
  double math rounded to f32, *R inf special case, /R zero/inf divisor,
  NaN results as 0xFFFFFFFF, epsilon snapping in sin/cos/tan, STW
  classification incl. denormal-as-f32; __ASSERT== compares raw 32-bit
  patterns like awlsim asScalar; TRUNC/RND keep the original dword on
  out-of-range, RND rounds half-to-even like Python round()). insn_ARITH_R
  now passes fully. Also fixed: __ASSERT sets NER=0 after comparing
  (awlsim insn_assert_*), __FEATURE loads the feature value into ACCU1,
  and CALL SFC 46/-3 terminate the block like BEA (awlsim STOP/SHUTDOWN
  abort the scan; the driver also stops the scan loop) — this unblocked
  insn_CALL, insn_SE/insn_SS, sfb0-5, tc200 array/pointer/udt, shutdown
  and startup_OB100/102. dbpointer.awl line 99 (P#"Data".DTVAR1) remains
  open — awlsim itself needs a symbol table for that file. Fixed since last
  writing: infinite-recursion exits cleanly (call-stack guard at awlsim's
  callStackSize = 256, exit code 2); the Formelparser SCL project compiles
  (const symbols fold in rvalue position via mlirGenConstSymbol; forward
  declarations are replaced by the real definition in mlirGenFunctionLike;
  the string functions CONCAT/STRING_TO_INT/INT_TO_STRING/LEN/... are
  implemented inline in mlirGenStringBuiltin — see the SPEC matrix and
  test/sclang/jit_strings.scl). Fixed since last writing:
  pointer.awl fully (TARn stores the AR as a raw S7 pointer via the new
  awl.ar_raw/scl_rt_ar_raw; LARn <mem> loads the operand's 32-bit content;
  TARn ARm does ARm := ARn; [LD n]/[#var]/[MD 4] indirect bases load the
  operand's value and raw P#x.y values take the access instruction's area;
  AUF DI [LW 0] sets the DI register; AR2 base applies to DI accesses only
  in FB frames), array.awl (FC ARRAY params by-ref: 6-byte DB-pointer
  header in the VL slot, callee derefs it incl. the header DB word for
  DB/DI areas; FB array params value-copied; forwarding via
  copyArrayParamInd),
  anypointer.awl (ANY header type code/count per awlsim makeByAutoType —
  `anyAutoType`/`anyTypeCodeOf`, incl. ARRAY -> (elem code, elem count) —
  plus FB VAR_IN ANY params now stage the full 10-byte header into the
  instance DB, not just the raw pointer dword) and any-to-instance-db.awl.

## ST/SCL trace tier (s-celles IEC 61131-3, value-level)

`test/conformance/run_st_traces.py` executes the 92 `.st`/`.trace` pairs of
`s-celles-iec61131-conformance` through the SCL engine and compares every
per-scan output. Each `PROGRAM` is rewritten to a `FUNCTION_BLOCK` plus an
instance `DATA_BLOCK`, wrapped in an `OB1` that copies per-scan inputs from
fixed M-area slots into the instance, calls the block and copies the outputs
back. The driver runs it with `--emit=jit --jit-cycles=N --cycle-ms=<period>`
`--scan-io=... --scan-inputs=...` (writes scan inputs before each OB1 scan,
prints `scan i: name=value ...` after it; see test/sclang/jit_scan_io.scl)
and the observed scan lines are compared to the trace. `--cross` also runs
each wrapper SCL->AWL->JIT and requires identical scan lines (differential
check of the SCL and AWL lowering paths; the AWL path itself is
oracle-verified against awlsim above). Tests using constructs outside the
supported subset (CONFIGURATION, LREAL, STRING/array/struct outputs, ...) are
counted as "unsupported", not failures.

Current: **40/75 pass (53%), 17 unsupported** (74 traces are `status:
conformant`, 18 `implementation-defined`). Builtin standard FBs (TON/TOF/TP,
CTU/CTD/CTUD, R_TRIG/F_TRIG/SR/RS) are implemented: fixed instance layouts
(forced in the lowering's FbLayoutCache, mirrored by scl_rt_timer_fb /
scl_rt_counter_fb / scl_rt_edge_fb in the runtime, ABI v11) with member
access via scl.getelement's fb attr. Open clusters: array literals `[...]`,
enum types, SHL/SHR/ROL/ROR, unsigned-source conversions (UINT->DINT
zero-extend), DATE/TOD/DT arithmetic, FUNCTION VAR retention, FB EN/ENO,
named-vs-positional call binding, REAL output observation via __BITS.

## Remaining failures and known approximations

All compile tiers pass 100% (AWL 116/116, SCL single 15/15, SCL projects
17/17) and the JIT smoke tier passes 103/105. The oracle tier is 70/71 —
see the list above; accumulator registers are now modeled at runtime
(awl.accu_get/accu_set), so values persist across labels and calls.
The former LogMsg flake (~4% of runs) is fixed; see AGENTS.md for the two
root causes (StructTypeStorage uniquing compared names OR types instead of
AND; the CallFcOp canonicalizer resolved callees via the nearest symbol
table, which stops at the enclosing FB, and dereferenced the null result in
release builds). LogMsg now passes 100/100 `--emit=mlir` runs and 50/50 in
mlir-std/llvm/jit.

Every remaining spec deviation is an `approx` row in the generated SPEC
matrix ([spec-conformance.md](spec-conformance.md), currently 96 tags, 0
`gap` rows): SF, MCR nesting, TAW, real math via float32 host math,
TRUNC/RND by truncation, TDB not modelled, FP/FN edge memory, symbolic DB
members, and the SCL string functions' flat zero-padded buffers
(STRING_TO_REAL/REAL_TO_STRING/VAL_STRG stay external stubs).

## Oracle tier: what passes and what the failures mean

52 of 71 `__ASSERT` files pass all their executed oracles. Block
calls are now fully modeled: FC params are frame-relative with
param_in/param_out copies (recursion works — factorial passes), RET_VAL is an
implicit interface member, indirect `UC FC [MW x]` dispatches over the
file's blocks, FB multi-instances (`CALL #var`) work via DI+AR2 addressing,
AR1/AR2 and DB/DI registers are call-saved, and SFB0-5 run real counter/timer
semantics in the runtime (`scl_rt_sfb`): SFB0-2 counters (CTU/CTD/CTUD with
R > LOAD > edge precedence per awlsim) and SFB3-5 timers (TP/TON/TOF with
STATE/STIME/ATIME in the instance DB, ATIME only updated while running,
PT==0 resets IN) — all six `sfb*.awl` oracles pass, plus `sfb/basic.awl`
(`CALL SFB -1` is a no-op, BIE stays 1). `insn_CALL` and
`insn_CALL_multiinstance` pass all oracles. SFC 21 (FILL, full ANY-pointer
semantics incl. error codes 0x8124/0x8325/0x8122/0x8323/0x813A/0x833A/
0x8128/0x8329 and BVAL wrap-around), SFC 47 (WAIT, WT in us) and SFC 64
(TIME_TCK) run in the runtime (`scl_rt_sfc`); their params are staged as
full ANY headers / scalars in the marker area at kLocalBase by the frontend
(`CALL SFC` param parsing). IEC library FCs called by name run in the
runtime too (`scl_rt_iec`: LEN 21, EQ_DT 9, GE_DT 12, DELETE 4, plus
TIME_TCK->SFC 64; callee `IECn` with an `iec` attr, params staged by the
shared `emitRtCall`) — `fc9_eq_dt` and `fc4_delete` pass all oracles.
User blocks called by quoted name (`CALL "_STR_PRESET" (...)`) route through
the common call path with staged params; by-reference params (ANY ->
10-byte ANY header, STRING/DATE_AND_TIME -> 6-byte POINTER read via
`P##param`) are passed as pointers. `P#DBn.SYM` symbolic pointers resolve
through the DB layout, VL (0x87) pointers use a 21-bit byte field (the FC
param area starts at 8192), `+ARn P#x.y` adds to the AR, LAR/TAR from ACCU
zero-extend, STRING[n] DB members get their max-len byte initialized, and
BOOL RET_VAL occupies a full byte. ANY is 10 bytes and POINTER 6 bytes in layouts;
symbolic members of undeclared DBs (DB1.VAR0) are auto-assigned offsets in
order of first use, and every DB's referenced extent is registered via
`awl.set_db_ref` in `__scl_db_init` for ANY-pointer bounds checks.
TIMER/COUNTER-typed block variables hold the timer/counter NUMBER and
indirect through the T/C area at it (`awl.pack_area` + `scl_rt_*_ptr`), so
`insn_Z` and `acme-005/006` pass all oracles.

The only remaining oracle failure is `startup_multiple.awl` (nrAccus: we
default to 4 accumulators, the file expects 2 — accepted; each failure
message names the line and actual/expected). The status word IS cleared at
OB entry (`scl_rt_ob_start`, SPEC-tagged), so `startup_OB100`/`startup_OB102`
pass.

The floor in `run_conformance.py` (`FLOOR_ORACLE = 0.15`) guards against
regressions.

## Historical log: what the corpora caught (early rounds)

The AWL frontend previously passed 56/116; the corpora drove these fixes in
`lib/AwlGen/AwlGen.cpp` (all covered by `test/AwlGen/awlsim-extensions.awl`):

- `L <immediate>` (plain ints, `W#16#…`, `DW#16#…`, `L#…`, `P#…`, `T#…`,
  `D#…`, `TOD#…`, `S5T#…` with H/M/S/MS components and `_` separators);
- local data `L/LD/LW`, `#locals`, `STW`, peripheral `PE*/PA*`/`PI*/PQ*`;
- indirect addressing `B/W/D [AR1, P#…]`, `M [AR1, …]`, `U [AR1, …]`,
  `P#"DB".VAR`, `P#DB42.ARR1[1].V1`;
- symbolic operands (`SYM_M_0.0`, `"name"`, `'literal'` → hashed marker bit);
  symbolic DB members (`DB1.VAR0`, `DB1.DBVAR_UDT1.VAR6_BOOL` → DBW 0/DBX 0.0
  of that DB — approximation, DB contents are runtime memory);
- instructions: `OPN`, `TDB`, `LC`, `SA/SD`, `FP/FN` (approximated by U/UN —
  no edge memory), `SAVE`, `BLD`, `MCR(/MCR)`, `ENT/PUSH`, `POP/LEAVE`,
  `INC/DEC`, `TAK/TAD`, `ITD/DTI`, `INVI/INVD`, `RND/RND±` (truncation),
  `BTI/BTD/ITB/DTB` (BCD via arith ops), `MOD`, `TAW`, `X/XN(/XN(`,
  `SPL/SPZ/SPP/SPM/SPS/SPBI/SPBN/B/NBI/SPU/SPO/SOS/JB/JNB`, `LOOP`,
  `BEA/BEB`, `+ <immediate>`, bare `U/O/…`, `A/O` without `(`;
- `CALL SFC47(…)`, `CALL "GE_DT"(…)`, `CALL "FB", "IDB" (…)`, `UC FC [MW 50]`;
- block header lines (`TITLE/AUTHOR/FAMILY/NAME/VERSION/NETWORK/…`), `{…}`
  attribute blocks, quoted block names (`OB "CYCLE"`), awlsim directives
  (`__FEATURE` etc. line-skipped; `__ASSERT*` lowered to runtime oracles —
  see the oracle tier above), `__CNST_PI` etc.;
- accumulator semantics: `T` does not pop ACCU1, short stacks pad with 0.

Bugs found and fixed while doing this: two-token lookahead staleness after
line skips (`Lexer::nextLine`), an infinite loop on lowercase `T#2s` and on
`P#DB42.` literals, and a null-`savedRlo` crash for `A(` as the first
instruction of a chain.
