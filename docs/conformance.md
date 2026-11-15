# External conformance testing

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
  `__STW`/status-bit asserts (`BIE`, `OV`, `A0`, …) are skipped (the sim
  does not track status bits). Malformed/unknown assert forms skip the line
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
| `tallakt-awltool`, `PLC-lang-rusty`, `s-celles-iec61131-conformance` | AWL tooling / IEC ST (ST is not SCL; used only for reference) | various |

Totals: 22 `.scl` files (17 project dirs), 116 `.awl` files.

## Current results (LLVM 17 build, this writing)

- AWL compile: **109/116 pass (94%)**
- SCL single file: **15/15 pass (100%)**
- SCL per project: **16/17 pass (94%)**
- JIT smoke (files with OB 1, full pipeline + execution): **65/77 pass (84%)**
- Oracle (`__ASSERT` files with OB 1, value-level, exit code must be 0):
  **31/53 pass (58%)** — see "Oracle tier" below

## Remaining failures = documented gaps

AWL (7 files) — need new awl dialect ops (not yet added):

- shift/rotate instructions `SSI SLD SRD RLD RRD` (insn_SHIFT, shiftregister,
  shiftregister_tp, quine AWL): the awl dialect has no shift ops and a
  variable shift count cannot be expressed with the existing arith ops.
- real math `ABS`, `TAN` (insn_ARITH_R, benchmark): no float math ops in the
  awl dialect (RND* is approximated by truncation for the same reason).

SCL (9 single / 6 projects) — all external TIA library types or cross-file
references when a file is compiled alone:

- `Variant`, `Program_Alarm`, `DataLogCreate`/`DataLog*` (TIA library types,
  lopez-dev BackupAndRestore/LogMsg/rezept);
- `Remote` (TIA-Put-Get-OUC PUT/GET);
- undefined constant `FC_ID_CHECK_ARRAY` (FC_ID_CHECK references a symbol from
  another TIA library file);
- cross-file types/symbols in Step7-Formelparser-SCL when compiled per file
  (pass as a project except the `CONCAT`/`STRING_TO_INT`/`INT_TO_STRING`
  string builtins, which are not implemented).

## Oracle tier: what passes and what the failures mean

31 of 53 `__ASSERT` files pass all their executed oracles (was 27). Block
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
(`CALL SFC` param parsing). ANY is 10 bytes and POINTER 6 bytes in layouts;
symbolic members of undeclared DBs (DB1.VAR0) are auto-assigned offsets in
order of first use, and every DB's referenced extent is registered via
`awl.set_db_ref` in `__scl_db_init` for ANY-pointer bounds checks.

Remaining oracle failures map to known gaps (each message names the line and
actual/expected):

- pointers/ANY/DB-pointer types (`pointer`, `anypointer`,
  `any-to-instance-db`, `dbpointer`): `P#`-valued variables and ANY
  pointer params are not yet modeled;
- `array`, `string` (compile-tier gaps), `dbpointer` (compile gap);
  `struct`, `udt`, `DB`, `latin1` now pass: FB/FC interface sections are
  re-aligned at section boundaries (BOOL bit-packed, WORD/DWORD even),
  initializer skipping no longer eats members after `END_STRUCT` without
  `;`, DB-init member chains resolve arbitrarily deep
  (VAR10.VAR10_3[1].VAR10_3_2[2]), whole-STRUCT/UDT CALL params are copied
  byte-wise, TYPE declarations are pre-scanned to a fixpoint so forward UDT
  references size correctly, and single-char quoted literals ('ä') yield
  their raw byte value;
- startup info (`startup_*`, `ob_startup_info`: OB number / cycle info in the
  ACCU at OB start not provided);
- `insn_SI`/`insn_Z`: timer/counter NUMBER indirection through interface
  words; `insn_MISC`: BCD/INV edge cases; `latin1`: char literal encoding;
- IEC library FCs (`fc4_delete`, `fc9_eq_dt`, `fc12_ge_dt`, `fc21_len`):
  still host stubs;
- `stw_operators`: STW bit tracking (A0/A1/OV/OS) beyond the skipped
  `__STW` asserts.

The floor in `run_conformance.py` (`FLOOR_ORACLE = 0.15`) guards against
regressions.

## What the corpora caught (fixed this round)

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
