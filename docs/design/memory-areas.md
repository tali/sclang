# Design: memory areas in SCL and AWL

- **Status**: phase 1 implemented (SCL side); AWL mapping and host binding pending
- Related: ADR 001 (byte order), ADR 002 (runtime ABI), ADR 003 (awl dialect)

## Implemented (phase 1)

- `!scl.direct<area, db, byte, bit, width>` type in the scl base dialect
  (area letters `I`/`Q`/`M`/`DB`/`T`/`C`, width in bits 1/8/16/32), parsed and
  printed by `SclDialect::parseType`/`printType`.
- `scl.getdirect` op (`scl.getdirect M, 0, 10, 3, 1 : !scl.address<!scl.direct<M, 0, 10, 3, 1>>`),
  `Pure`, with a verifier tying the attrs to the result type.
- Lexer: `%…` is a single `tok_direct_address` token (raw text without `%`);
  `Parser::ParseDirectAddressExpression` parses `%I0.0`, `%IB2`, `%IW4`,
  `%ID0`, `%DB1.DBX0.0`, `%DB1.DBB2`, `%DB1.DBW4`, `%DB1.DBD8` into a
  `DirectVariableAST`. Plain `%I0` without a bit is rejected.
- MLIRGen: `DirectVariableAST` -> `scl.getdirect`; rvalue loads use the
  access width as value type (bool for bit access, else int of that width).
  `%T`/`%C` are rejected for now ("timers/counters not yet supported").
- LowerToStd: `scl.load`/`scl.store` through `address<direct<…>>` lower to
  `@scl_rt_read/write[_bit](area, db, byteOff, widthBytes[, value])` calls
  (private `func.func` declarations inserted on demand at module start);
  `scl.getdirect` lowers to a packed-i64 placeholder (the address type
  converts to i64; the value is never used by the runtime calls).
- Scan cycle: OBs (`organization_block`, marked with the
  `scl.organization_block` unit attr on `scl.function`) emit
  `@scl_rt_update_inputs()` at entry and `@scl_rt_update_outputs()` before
  every `scl.return`.
- Sim runtime: `include/sclang/Runtime/Runtime.h` (fixed C ABI) +
  `lib/Runtime/SimRuntime.cpp` (fixed-size I/Q/M/T/C areas, default 65536
  bytes, OOB aborts; DBs grow lazily; big-endian; virtual clock). The
  `sclang` driver binds the JIT symbols via `ExecutionEngine::registerSymbols`
  and applies `--area-size AREA=SIZE`. A host can redirect all calls by
  `scl_rt_register(&vtable)`.
- Big-endian storage (ADR 001) is implemented in the byte-assembly helpers,
  the IDB initializer baking, and the sim runtime.

## Implemented (phase 2)

- `%T`/`%C` accepted by MLIRGen; the sim treats them as plain byte areas
  (IEC timer semantics — S5TIME encoding, timer instructions — and callers
  of `scl_rt_cycle` remain a gap).
- `--area-size DB<n>=size` pre-sizes simulated DBs (bounds-checked like the
  other areas; unset DBs still grow lazily).
- Host-process binding via driver flags: `--runtime=host --runtime-lib=path.so`
  dlopens the library, resolves `scl_host_runtime_vtable()` and registers the
  returned `SclRtVTable` via `scl_rt_register()` — the sim forwards all ABI
  calls to it. `--runtime=sim` (default) clears the vtable. Verified manually
  with a logging host `.so`.

## Pending

- IEC timer semantics beyond the sim model (see above). SCL-side timer FBs
  are implemented as builtins: `data_block d TON …` + `TON."d"()` lower to
  `@scl_rt_timer_fb(mode, idbPtr)` (ABI v3) with a fixed builtin IDB layout
  (IN @0, PT @4, Q @8, ET @12); the sim keeps per-instance state against the
  virtual clock (see AGENTS.md, test jit_ton.scl).
- AWL operand mapping (below) and the SCL->AWL backend.

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
counters (T/C areas) are implemented (phase 3): the sim runtime keeps timer
state (mode, running, start, duration) and counter values; AWL timer
instructions (`SP/SE/SS/SV/SI T n`) lower to `@scl_rt_timer_start(i32, i8, i64)`
and counter instructions (`ZV/ZR/S/R C n`) to `@scl_rt_counter_op(i32, i8, i64)`
(ABI v2, appended to the vtable, NULL-able for older hosts). Reads of `T`/`C`
return the current timer value in ms (remaining while running, preset when
stopped — writing `%T` sets the preset) / the counter value; bit 0 reads return
the status (timer Q / counter != 0). The virtual clock advances via
`@scl_rt_cycle(ms)`, driven between JIT scans by `--jit-cycles=N --cycle-ms=M`.
S5TIME literals (`S5T#1S`) encode to (base<<12)|BCD; the sim decodes them.
Deviations: `L Tn` returns plain ms, not the S5TIME-BCD word of real S7;
`S5T#vMS` is parsed as v*10ms (unit suffix MS/S), and S5T#100MS means 1s.

## Testing

- lit parse tests for `%`-addresses; lowering tests (`scl.load %I0.0` →
  `call @scl_rt_read_bit(...)`).
- JIT sim tests: program writes `%Q…`/reads `%M…`, debug-prints results;
  timer test advancing the virtual clock explicitly (jit_timer.awl:
  `--jit-cycles=3 --cycle-ms=600` expires a 1s on-delay timer).
