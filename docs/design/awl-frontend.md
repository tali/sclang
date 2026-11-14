# Design: AWL (STL) frontend

- **Status**: implemented (AWL frontend: `.awl`/`-x awl` -> awl dialect ->
  LowerToStd -> JIT; tests in test/AwlGen + test/sclang/jit_awl.{scl,awl}).
  Deviations from the plan below: ACC is a codegen-time SSA stack (no phis —
  labels reset the statement state instead, see AGENTS.md); `awl.block` is
  lowered by splitting the flat block at labels/jumps in AwlBlockOpLowering
  rather than emitting cf blocks during codegen; CC and CALL DB m are
  rejected for now; `_mlir_ciface_OB1` is synthesized when OB 1 exists.
  Implemented since: IEC timers (SP/SE/SS/SV/SI, S5T#… literals) and counters
  (S/R/ZV/ZR), and `CALL FB n, DB m` / `UC FB n, DB m` with instance DBs
  (bare `DB.…` operands inside an FB lower to the instance DB func argument,
  see AGENTS.md). Remaining: CC, CALL DB m, FC parameters.
- Related: ADR 003 (parallel dialect), ADR 002 (runtime ABI), `memory-areas.md`

Target: classic S7-300/400 AWL (two-accumulator model). One instruction per
line, `:` labels, `//` comments.

## Dialect

`awl` dialect in `include/sclang/AwlDialect/` (`AwlOps.td`), parallel to `scl`.
Roughly 20 ops, all operating on an explicit accumulator model:

```
awl.load / awl.transfer / awl.and / awl.or / awl.not / awl.xor
awl.set / awl.reset                      (S / R on bits)
awl.add/sub/mul/div/mod / comparisons     (+I +D …, ==I <>I …) — set ACC/RLO
awl.jump / awl.jump_if / awl.jump_if_not  (SPA/SPB, JC/JCN, JM/JZ/…)
awl.call                                 (CALL FB n, DB m / UC / CC)
awl.nop
```

Operands are `!scl.direct` (shared with SCL, see memory-areas.md).

## MLIRGen (`lib/AwlGen/`)

- Lexer/parser for `.awl`; dispatch by extension or `-x awl` in both drivers.
- **Virtual accumulator stack**: MLIRGen keeps a small stack of SSA values.
  `L MW0` pushes a load, `+I` pops two and pushes `arith.addi`, etc. No
  implicit state. At labels/merges where the stack is non-empty, ACC1/ACC2
  become block arguments (phis); in practice AWL code keeps the stack empty
  at labels.
- **RLO** (result of logic operation) is an `i1` SSA value: `U x` →
  `rlo &= bit(x)`, `O`, `NOT`; `= M0.0` assigns RLO to a bit; `JC label` →
  `cf.cond_br rlo`.
- Jumps → cf blocks, reusing the goto machinery (`labelBlocks`, on-demand
  block creation, dead block after unconditional jump).
- Area access lowers to the runtime C ABI (ADR 002) — shared with SCL.
- Calls → `func.call` (same as `scl.call.fb`).

## Lowering to std

The LowerToStd conversion target additionally marks `awl` illegal and converts
its ops to func/arith/memref/cf (or a sibling pass). LowerToLLVM and the JIT
are unchanged.

## Testing

- lit parse tests for AWL; lowering tests; JIT tests.
- Cross-check: the same program written in SCL and in AWL must produce
  identical JIT output.
- `awl → awl` round trip via the SCL→AWL backend validates the printer.
