# Design: lowering SCL to AWL (text backend + in-memory execution)

- **Status**: implemented. Two consumers of the LowerToAWL pass
  (`lib/SclTransforms/LowerToAWL.cpp`):
  1. `sclang-gen --emit=awl` prints S7 text via `awl::printAwlText`
     (`lib/AwlDialect/Printer.cpp`) — legacy export for comparison with
     STEP 7 output. The text print/re-parse round-trip has known
     accumulator-model limitations (the printer models T as popping ACCU1;
     awlsim does not), so it is not used for execution.
  2. `sclang --emit=awl-mlir` (dump) and `--emit=jit-awl` (run) keep the
     awl-dialect IR in memory and feed it straight to LowerToStd/LLVM/JIT —
     the primary SCL->AWL execution path, used by the `--cross` differential
     in test/conformance/run_st_traces.py and fuzz_st.py. With
     `synthCiface`, the pass synthesizes the `_mlir_ciface_OB1` awl wrapper
     block (call OB1 + BE) that the JIT driver invokes.
  Tests: test/SclTransforms/to_awl{,_areas,_fb}.scl,
  test/AwlGen/roundtrip.awl, test/sclang/jit_awl_backend.scl,
  test/sclang/jit_awl_direct.scl (--emit=jit-awl).
  Deviations/gaps: FB arrays/structs, strings and DB initializers are
  rejected/dropped with an error or warning; `scl.debug.print` is dropped;
  cross-width assignments insert an awl.cast (with `zext` for unsigned
  sources; signed i16->i32 widening emits SLD 16 + SSD 16 — SRD is a
  *logical* shift in S7/awlsim);
  bool logic is emitted as rlo chains (U/UN/O/ON/=) with temp bits for
  computed operands; the CASE selector is re-loaded before each compare
  (AWL L shifts ACC1). Computed values are spilled to M-area temps right at
  their definition, and operand temps are (re-)stored at every use, so no
  temp read can be dominated by a skipped loop/conditional.
- Related: ADR 003, `awl-frontend.md`

Goal: emit canonical AWL (S7 mnemonic) listings from SCL sources, to compare
with existing implementations (e.g. STEP 7 output), and to execute SCL
through the AWL dialect without a text round-trip. Exposed as
`sclang-gen --emit=awl` (text) and `sclang --emit=awl-mlir` / `--emit=jit-awl`
(in-memory).

## Pipeline position

```
.scl ──> scl dialect ──> LowerToAWL pass ──> awl dialect ──> printer ──> AWL text   (sclang-gen --emit=awl)
                                              └─> LowerToStd ──> LLVM ──> JIT              (sclang --emit=jit-awl)
```

The `awl` dialect is the same one the AWL frontend produces, so `awl → awl`
round-trips validate the printer.

## Pass structure (`lib/SclTransforms/LowerToAWL.cpp`)

1. **Variable → area allocation**: FC/FB-local vars → M area (or L local
   stack); FB instance DBs → `DBn`; direct vars (`%…`) stay direct. Produces
   a symbol → `!scl.direct` map used by all later mappings.
2. **Op mapping**:
   - load/store → `L`/`T` with sized operands (`L MW10`, `T DB1.DBW0`).
   - arithmetic → accumulator ops (`a + b` → `L a; L b; +I`); comparisons →
     `==I` etc. setting RLO.
   - `scl.if` → condition into RLO + `JC`/`JCN` + labels:
     `…; JC else; <then>; SPA end; else: …; end:`.
   - loops → labels + jumps; `scl.case` → compare chain or S7 `SPL` jump list.
   - `scl.call.fb` → `CALL FB n, DB m` (db number resolved from the IDB's
     data block; FB members → `DB.DBW<byteOff>` directs on the instance DB
     inside the FB block); FCs → `UC FC n`; OB → `OB 1` wrapper.
   - FC parameters: passed through a reserved window at the top of the M
     area (byte 65280 + 8*i per param, return value at byte 65272) — the AWL
     analog of the S7 L stack. The caller stores each argument to the
     callee's param slot before `UC FC n`; the FC reads its inputs from
     there and stores its return value to the result slot before `BE`.
     Known limitation: all blocks share the M area (no per-block L stack),
     so locals of different blocks alias.
   - `scl.debug.print` → no AWL equivalent; emit a comment or drop (decision).
3. **Printer**: custom assembly per awl op emitting S7 mnemonics; canonical
   label naming (`NEXT1:`, …); operand format `E 0.0` / `MW 10` / `DB1.DBX 0.0`.

## Byte order

Requires big-endian storage (ADR 001) so that data written by the SCL program
matches what AWL consumers expect.

## Testing

- FileCheck tests on emitted AWL text for each construct (if/for/while/case/
  FB calls/arrays).
- Round-trip: parse emitted AWL with the AWL frontend and re-emit; stable.
- Cross-check against reference listings from existing toolchains.
