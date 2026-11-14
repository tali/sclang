# ADR 003: AWL as a parallel source dialect converging at std IR

- **Status**: accepted and implemented (frontend; SCL->AWL backend pending)
- **Date**: 2025 (review session)

## Context

Siemens AWL (STL, "Anweisungsliste") is the assembly-like language for S7 PLCs.
We want to (a) compile AWL programs, and (b) lower SCL to AWL text to compare
with existing implementations (e.g. STEP 7 output).

The question was whether AWL should be an *intermediate* dialect (SCL → AWL →
backend) or a *parallel* one. An intermediate dialect would force every SCL
construct through AWL's accumulator/jump-list shape even when targeting the
host — pointless indirection, and AWL cannot express SCL's structured types
naturally.

## Decision

- AWL gets its own dialect (`awl`, in `include/sclang/AwlDialect/`), **parallel
  to `scl`**, as a *source-level* dialect for the AWL frontend
  (`lib/AwlGen/`). Both frontends converge on the existing std IR
  (func/arith/memref/cf) — MLIR already plays the role of the intermediate
  language; no custom intermediate dialect is needed:

  ```
  .scl ──> scl dialect ──┐
                         ├─> LowerToStd ──> func/arith/memref/cf ──> LowerToLLVM ──> JIT
  .awl ──> awl dialect ──┘
  ```

- The same `awl` dialect is *also* kept as a **backend target** for
  SCL→AWL: a `LowerToAWL` pass maps scl ops to awl ops and a custom printer
  emits S7 mnemonics (`sclang-gen --emit=awl`). The AWL frontend + this backend
  give an `awl → awl` round trip for validating the printer.
- Shared addressing: both dialects use `!scl.direct` (defined in the scl base
  dialect) for memory-area operands; both lower area access to the runtime C
  ABI (ADR 002).

## Consequences

- The LowerToStd conversion target additionally marks the `awl` dialect illegal
  and converts its ops (or a sibling pass does) — backend (LowerToLLVM, JIT)
  unchanged.
- AWL's unstructured jump list maps to cf blocks directly (reuses the goto
  machinery); its accumulator stack is modeled as explicit SSA values in
  MLIRGen (block args at merges), RLO as an `i1`.
- Target family: classic S7-300/400 AWL (two-accumulator). S7-1200/1500 is a
  possible later extension.
