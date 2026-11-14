# Design: memory areas in SCL and AWL

- **Status**: planned
- Related: ADR 001 (byte order), ADR 002 (runtime ABI), ADR 003 (awl dialect)

## Addressing model

One shared type for directly-represented variables, defined in the scl base
dialect so both frontends use it:

```
!scl.direct<area: I|Q|M|DB|T|C, db: int, byte: int, bit: int, width: 1|8|16|32>
```

| SCL syntax | AWL operand | meaning |
|---|---|---|
| `%I0.0`, `%Q0.0`, `%M10.3` | `E 0.0`, `A 0.0`, `M 10.3` | bit access, width 1 |
| `%IB0`, `%QB2`, `%MB10` | `EB 0`, `AB 2`, `MB 10` | byte, width 8 |
| `%IW0`, `%QW4`, `%MW10` | `EW 0`, `AW 4`, `MW 10` | word, width 16 |
| `%ID0`, `%QD8`, `%MD4` | `ED 0`, `AD 8`, `MD 4` | double word, width 32 |
| `%DB1.DBX0.0`, `%DB2.DBW4` | `DB1.DBX 0.0`, `DB2.DBW 4` | DB access (db = DB number) |

## Frontend work (SCL)

- Lexer: `%`-prefixed direct-address tokens.
- Parser: new `DirectVariableAST` — an lvalue/rvalue resolving to a constant
  `!scl.direct` address (like `SimpleVariableAST` but symbol-free).
- MLIRGen: `scl.getdirect` (Pure, constant-like) producing
  `!scl.address<!scl.direct<...>>`; loads/stores reuse `scl.load`/`scl.store`.
- Area sizes configurable via CLI (`--area-size M=1024`, …). DB numbering
  follows S7 conventions.

## AWL frontend mapping

AWL operand parsing (`E/A/M/DBn.<B|W|D|X> addr[.bit]`) produces the same
`!scl.direct` values — one lowering for both languages.

## Lowering

In LowerToStd: `scl.load`/`scl.store` whose address element is `!scl.direct`
lower to the runtime C ABI calls (ADR 002) — `@scl_rt_read/write` for
byte/word/dword, `@scl_rt_read_bit/write_bit` for bits — with the width taken
from the direct type. Bit access on a byte-addressed area: byte = byteOff,
bit = bit. No inline memref access for areas.

## Scan cycle

OB entry emits `@scl_rt_update_inputs()`, OB exit (before return) emits
`@scl_rt_update_outputs()` — models S7 cyclic execution (read inputs → run OB1
→ write outputs) and gives host runtimes synchronization points. Timers and
counters (T/C areas) go through `scl_rt_timer_*`/`scl_rt_counter_*` with the
virtual clock advanced by `@scl_rt_cycle(ms)` — scope decision pending.

## Testing

- lit parse tests for `%`-addresses; lowering tests (`scl.load %I0.0` →
  `call @scl_rt_read_bit(...)`).
- JIT sim tests: program writes `%Q…`/reads `%M…`, debug-prints results;
  timer test advancing the virtual clock explicitly.
