# Design: lowering SCL to AWL (text backend)

- **Status**: planned
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
   - `scl.call.fb` → `CALL FB n, DB m`; FCs → `CALL FC n`; OB → `OB 1` wrapper.
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
