# Design: lowering SCL to AWL (text backend)

- **Status**: implemented. `sclang-gen --emit=awl` runs the LowerToAWL pass
  (`lib/SclTransforms/LowerToAWL.cpp`) and prints S7 text via
  `awl::printAwlText` (`lib/AwlDialect/Printer.cpp`). Tests:
  test/SclTransforms/to_awl{,_areas,_fb}.scl, test/AwlGen/roundtrip.awl,
  test/sclang/jit_awl_backend.scl (SCL -> AWL -> JIT gives the same memory
  writes as the SCL original).
  Deviations/gaps: FB arrays/structs, strings, cross-width assignments and
  DB initializers are rejected/dropped with an error or warning;
  `scl.debug.print` is dropped;
  awl.cast is implicit in the mnemonic letter (not printed); bool logic is
  emitted as rlo chains (U/UN/O/ON/=) with temp bits for computed operands;
  the CASE selector is re-loaded before each compare (AWL L shifts ACC1).
- Related: ADR 003, `awl-frontend.md`

Goal: emit canonical AWL (S7 mnemonic) listings from SCL sources, to compare
with existing implementations (e.g. STEP 7 output). Exposed as
`sclang-gen --emit=awl`.

## Pipeline position

```
.scl ──> scl dialect ──> LowerToAWL pass ──> awl dialect ──> printer ──> AWL text
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
