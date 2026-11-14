//===- Runtime.h - Fixed C ABI between compiled SCL code and a runtime ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The fixed C ABI between compiled SCL code (MLIR / LLVM IR / JIT) and a
// runtime providing the S7 memory areas and hardware access (ADR 002).
//
// Contract:
// - The compiled program only ever sees these external symbols; it never
//   knows which backend is bound (simulation or host process).
// - Multi-byte values in the areas are stored BIG-ENDIAN (ADR 001): byte 0
//   of a word/dword holds the most significant byte (S7 layout, e.g. MW10 is
//   the high byte of the word at M10).
// - `area` is one of the SclRtArea codes (ASCII letters matching the SCL `%`
//   areas and the AWL operand areas). For SCL_RT_AREA_DB, `db` is the DB
//   number and `byteOff` is the offset within that DB.
// - `widthBytes` is 1, 2, 4 or 8 (byte/word/dword/...). Bit access uses the
//   *_bit variants with `bit` in 0..7.
// - Scan cycle: the compiled OB calls scl_rt_update_inputs() on entry and
//   scl_rt_update_outputs() on exit, giving the runtime synchronization
//   points (host pushes the process image / reads the outputs).
// - scl_rt_cycle() advances the virtual clock (timers/counters).
//
// Host integration: a host process calls scl_rt_register() with its vtable
// before running the JIT code; the simulation runtime forwards every call
// through the registered vtable. The driver binds the JIT symbols to these
// functions via ExecutionEngine::registerSymbols.
//
// This header versions the ABI: changing it is a breaking change for host
// runtimes (hence ADR 002).
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_RUNTIME_RUNTIME_H_
#define SCLANG_RUNTIME_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Area codes: ASCII letters matching the SCL `%` areas / AWL operand areas.
enum SclRtArea {
  SCL_RT_AREA_I = 'I',  ///< inputs      (%I / AWL E)
  SCL_RT_AREA_Q = 'Q',  ///< outputs     (%Q / AWL A)
  SCL_RT_AREA_M = 'M',  ///< markers     (%M / AWL M)
  SCL_RT_AREA_DB = 'D', ///< data blocks (%DBn / AWL DBn), db = DB number
  SCL_RT_AREA_T = 'T',  ///< timers      (%T / AWL T)   (reserved)
  SCL_RT_AREA_C = 'C',  ///< counters    (%C / AWL Z)   (reserved)
};

/// Read a value of `widthBytes` bytes at `byteOff` (big-endian, sign-extended
/// to int64_t).
int64_t scl_rt_read(uint8_t area, uint32_t db, uint32_t byteOff,
                    uint8_t widthBytes);
/// Write the low `widthBytes` bytes of `value` at `byteOff` (big-endian).
void scl_rt_write(uint8_t area, uint32_t db, uint32_t byteOff,
                  uint8_t widthBytes, uint64_t value);
/// Read bit `bit` of the byte at `byteOff`.
bool scl_rt_read_bit(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit);
/// Set/clear bit `bit` of the byte at `byteOff`.
void scl_rt_write_bit(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit,
                      bool value);

/// Scan-cycle begin: the host pushes the process image of the inputs.
void scl_rt_update_inputs(void);
/// Scan-cycle end: the host reads the process image of the outputs.
void scl_rt_update_outputs(void);
/// Advance the virtual clock by `elapsedMs` milliseconds.
void scl_rt_cycle(uint64_t elapsedMs);

/// Host-provided runtime operations. A host registers a vtable with
/// scl_rt_register(); the bound runtime forwards all ABI calls through it.
typedef struct SclRtVTable {
  int64_t (*read)(uint8_t area, uint32_t db, uint32_t byteOff,
                  uint8_t widthBytes);
  void (*write)(uint8_t area, uint32_t db, uint32_t byteOff,
                uint8_t widthBytes, uint64_t value);
  bool (*read_bit)(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit);
  void (*write_bit)(uint8_t area, uint32_t db, uint32_t byteOff, uint8_t bit,
                    bool value);
  void (*update_inputs)(void);
  void (*update_outputs)(void);
  void (*cycle)(uint64_t elapsedMs);
} SclRtVTable;

/// Register a host vtable; all subsequent ABI calls forward through it.
/// Pass NULL to return to the built-in simulation.
void scl_rt_register(const SclRtVTable *vtable);

/// Simulation-only configuration (NOT part of the fixed ABI): set the size
/// in bytes of a simulated area (I/Q/M/T/C). DB areas grow on demand unless
/// a size is set with scl_rt_sim_set_db_size().
void scl_rt_sim_set_area_size(uint8_t area, uint64_t size);
/// Set the size in bytes of a simulated data block (grown on demand if unset).
void scl_rt_sim_set_db_size(uint32_t db, uint64_t size);
/// Reset the simulation state (all areas, DBs, the virtual clock).
void scl_rt_sim_reset(void);
/// Virtual clock of the simulation in milliseconds.
uint64_t scl_rt_sim_clock_ms(void);
/// Dump all non-zero bytes of the simulated areas and DBs plus the virtual
/// clock to stdout, one `AREA[byte]=value` entry per line (sim only; a no-op
/// when a host vtable is registered). Used by the JIT driver's
/// `--dump-areas` to make memory-area writes observable in tests.
void scl_rt_sim_dump_areas(void);

/// Entry point a host runtime shared library must export when loaded with
/// `--runtime-lib=path.so`: returns the vtable the driver registers via
/// scl_rt_register().
const SclRtVTable *scl_host_runtime_vtable(void);

#ifdef __cplusplus
}
#endif

#endif // SCLANG_RUNTIME_RUNTIME_H_
