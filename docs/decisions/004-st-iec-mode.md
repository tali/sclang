# ADR 004: IEC 61131-3 ST as a mode of the SCL frontend — no new dialect

- **Status**: accepted, implementation started (2025 review session)
- **Date**: 2025

## Context

Our `.scl` frontend is Siemens S7-SCL: direct `%I/%Q/%M/%DB` addresses,
`PROGRAM`/`OB`/`FB`/`FC`, `EN`/`ENO`, `METHOD`s, `TIMER`/`COUNTER` aliases,
`STRING[n]`, multi-file TIA projects. IEC 61131-3 ST (what the public
s-celles conformance corpus uses) is the same language family with a
different surface: `CONFIGURATION`/`RESOURCE`/`TASK` declarations,
`VAR_GLOBAL`/`VAR_EXTERNAL`, `LREAL`, unsized `STRING`. We want both to
compile through one engine — the 75/75 passing IEC ST traces already prove
the SCL engine implements the IEC core semantics.

The question was whether ST needs its own MLIR dialect (parallel to `scl`,
like `awl` per ADR 003), or a mode of the existing frontend.

## Decision

**No new dialect. ST is a `LangMode` of the existing SCL frontend**
(`Lexer → Parser → AST → MLIRGen` in `lib/SclGen/`), selected with
`--std=iec` (default `--std=s7`; `-x=st` is an alias for `-x=scl --std=iec`).
A parallel dialect is only justified when the *IR model* differs (that is
why `awl` is its own dialect: accumulator machine, status word, jump list).
ST and SCL share execution semantics; the `scl` dialect ops are already
generic ST ops — the S7-ness lives in *types* (`!scl.direct` with
`DirectArea`, `S5TimeType`, …), not in a parallel op set.

The mode flag is threaded through the frontend, never into the dialect:

| layer | S7 vs IEC difference | sharing mechanism |
|---|---|---|
| Lexer | keyword set, `%` direct token | mode flag, shared tokenizer |
| Parser | S7-only productions (`%` directs, `TIMER`/`COUNTER`, `EN`/`ENO` params, `"DB".member`) vs IEC-only ones (`CONFIGURATION`/`RESOURCE`/`TASK`, `VAR_GLOBAL`/`VAR_EXTERNAL`, `LREAL`, bare `STRING`) | mode-gated branches, one shared AST (+ a `ConfigurationAST` node) |
| AST | `+ ConfigurationAST` | otherwise fully shared |
| MLIRGen | type table (`STRING` → `STRING[254]`, `LREAL` → new 64-bit real type), CONFIGURATION lowering | one MLIRGen, mode consulted in type resolution |
| `scl` dialect | + 64-bit real type (LREAL), + `scl.config` metadata op | one dialect, additive only; ops stay mode-agnostic |
| LowerToStd / LowerToAWL / LowerToLLVM / runtime | f64 arithmetic in the runtime ABI | shared, except the new numeric width |
| tests | TIA exports = S7 corpus; s-celles IEC traces = IEC corpus | same pipeline, `--cross` works in both modes |

Key reuse: IEC `VAR_CONFIG ... AT %I0.0` and S7 `%I0.0` lower to the *same*
`!scl.direct` type, and `CONFIGURATION`'s `VAR_GLOBAL` lowers to an ordinary
global `DATA_BLOCK` (named after the configuration) that `VAR_EXTERNAL`
members resolve into — the runtime memory model (ADR 002 areas) never
notices the difference. `RESOURCE`/`TASK`/`PROGRAM ... WITH task` bindings
are parsed and kept as a `scl.config` op (metadata for tools and for the
trace harness, which generates the scan schedule); execution scheduling is
a runtime/driver concern, not a codegen one.

Dialect additions (the only ones): an f64 `LrealType` in
`include/sclang/SclDialect/SclTypes.h`, and a `scl.config` op carrying the
configuration mapping as attributes.

## Consequences

- One verifier, one LowerToStd, one LowerToAWL, one JIT path for both
  languages; `--cross` (SCL→AWL→JIT differential) applies to IEC sources
  unchanged.
- S7-only constructs in IEC mode (and vice versa) produce a clear
  "S7-only construct in IEC mode" diagnostic, not a silent misparse.
- LREAL is the only real semantic addition: f64 arithmetic/comparison/
  conversion ops plus matching `scl_rt_*` runtime entries (parallel to the
  existing f32 REAL path).
- Implementation order (each closes a share of the 17 unsupported ST
  traces): bare `STRING` → `STRING[254]` + string scan-io slots (6 traces),
  `CONFIGURATION`/`VAR_GLOBAL`/`VAR_EXTERNAL` (6 traces), `LREAL` (5
  traces).
- The trace harness (`test/conformance/run_st_traces.py`) drives IEC mode
  (`--std=iec`) and generates the OB1 scan schedule from the parsed
  `TASK` declarations for configuration tests.
