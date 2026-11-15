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

What is run:

- every `*.awl`: dialect sniffed per file (awlsim project markers
  `ORGANIZATION_BLOCK` / quoted block names / `DATA_BLOCK` => `-x=awl`,
  SCL markers `FUNCTION` / unquoted `DATA_BLOCK` / `VAR_INPUT|OUTPUT|IN_OUT`
  => `-x=scl`), compiled with `--emit=mlir`;
- every `*.scl` (except `makefile.scl`) individually with `-x=scl`;
- every directory of `.scl` files concatenated (newline-separated) as one
  project with `-x=scl` (TIA exports are multi-file).

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

- AWL: **109/116 pass (94%)**
- SCL single file: **13/22 pass (59%)**
- SCL per project: **11/17 pass (65%)**

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
  (`__ASSERT==` etc. — skipped as test oracles), `__CNST_PI` etc.;
- accumulator semantics: `T` does not pop ACCU1, short stacks pad with 0.

Bugs found and fixed while doing this: two-token lookahead staleness after
line skips (`Lexer::nextLine`), an infinite loop on lowercase `T#2s` and on
`P#DB42.` literals, and a null-`savedRlo` crash for `A(` as the first
instruction of a chain.
