# ADR 002: Peripheral/memory access via a fixed C ABI runtime

- **Status**: accepted and implemented (sim runtime + JIT binding + host vtable via --runtime-lib)
- **Date**: 2025 (review session)

## Context

Memory areas (inputs E/%I, outputs A/%Q, markers M/%M, data blocks DB, timers,
counters) and hardware access need runtime support. Two consumers must work:

1. **JIT simulation** — deterministic, in-process, testable with lit.
2. **A host process** that wants to control the I/O (process image, real or
   virtual clock, event loop).

Baking areas as plain `memref.global`s in the compiled module would make host
integration impossible (the host cannot reach into the JIT's globals sanely),
and compiling them inline would bake in simulation semantics.

## Decision

All access to memory areas / hardware lowers to calls to a **fixed set of
externally-declared C ABI functions** (same pattern as `printf` for
`scl.debug.print`):

```c
// include/sclang/Runtime/Runtime.h
int64_t  scl_rt_read(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t widthBytes);
void     scl_rt_write(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t widthBytes, uint64_t value);
bool     scl_rt_read_bit(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit);
void     scl_rt_write_bit(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit, bool v);
void     scl_rt_update_inputs(void);    // scan-cycle begin: host pushes process image
void     scl_rt_update_outputs(void);   // scan-cycle end: host reads outputs
void     scl_rt_cycle(uint64_t elapsedMs); // advance virtual clock (timers/counters)
// timers/counters: scl_rt_timer_*/scl_rt_counter_* if modeled
```

- The compiled program (MLIR, LLVM IR, JIT) only ever sees these external
  symbols; it never knows which backend is bound.
- **Simulation (default)**: `lib/Runtime/SimRuntime.cpp` implements the ABI
  with static per-area byte buffers (sizes via CLI `--area-size`) and a
  virtual clock. The driver binds these to the JIT via
  `ExecutionEngine::registerSymbols` (MLIR 17).
- **Host integration**: the host calls `scl_rt_register(const SclRtVTable*)`
  before running; the driver then binds the same JIT symbols to C++
  trampolines that forward through the vtable. Supported both when sclang is
  embedded as a library and when the `sclang` binary loads a runtime `.so`
  (`--runtime=sim|host`, `--runtime-lib=path.so`).
- `--emit=llvm`/`--emit=obj` leave the runtime external, so a host can also
  link the compiler's output against its own C implementation — no JIT needed.
- Scan-cycle semantics: OB entry emits `@scl_rt_update_inputs()`, OB exit
  emits `@scl_rt_update_outputs()` (models the S7 cyclic execution and gives
  the host clean synchronization points).
- Widths are in bytes (1/2/4/8); byte order inside the runtime is big-endian
  (ADR 001).

## Consequences

- LowerToStd lowers `scl.load`/`scl.store` on `!scl.direct` addresses to these
  calls instead of memref access.
- Deterministic tests: the sim clock is advanced explicitly via
  `scl_rt_cycle`, so timer behavior is testable without real time.
- The ABI is versioned by the header; changing it is a breaking change for
  host runtimes (hence an ADR).
