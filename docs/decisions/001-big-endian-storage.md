# ADR 001: Big-endian storage for multi-byte values

- **Status**: accepted (planned; not yet implemented)
- **Date**: 2025 (review session)

## Context

S7 hardware memory is big-endian: `MW10` is the high byte at `M10`, the low
byte at `M11`; `DB1.DBW0` is the high byte at DB byte 0. The compiler's
byte-assembly helpers (`loadFromByteBuffer`/`storeToByteBuffer` in
`lib/SclTransforms/LowerToStd.cpp`) currently assemble little-endian, which is
correct only for the host CPU.

We want SCL programs, AWL programs, the JIT simulator, and real S7 hardware to
agree on data layout, and we want to lower SCL to AWL (see
`docs/design/scl-to-awl.md`) — that is impossible if the internal layout is
host-endian while AWL expects big-endian.

## Decision

All multi-byte values in byte buffers (FB instance DBs, structs, memory areas)
are stored **big-endian**, matching S7. This means reversing the byte order in
the byte-assembly helpers (byte `i` of an N-byte value lives at offset
`N-1-i` from the value's start address).

## Consequences

- `loadFromByteBuffer`/`storeToByteBuffer` and the runtime's read/write of
  word/dword areas all use big-endian.
- Existing lit tests that check byte-assembly CHECK lines must be updated.
- Host-endian is only used where values never cross into S7-visible memory
  (pure SSA integers are unaffected).
