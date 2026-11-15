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

/// Timer modes for scl_rt_timer_start (the S7 AWL timer instructions).
enum SclRtTimerMode {
  SCL_RT_TIMER_PULSE = 0,     ///< SP: output while the timer runs
  SCL_RT_TIMER_EXT_PULSE = 1, ///< SE: retriggerable pulse
  SCL_RT_TIMER_ON_DELAY = 2,  ///< SS: output after the delay has elapsed
  SCL_RT_TIMER_OFF_DELAY = 3, ///< SV: output until the delay has elapsed
  SCL_RT_TIMER_IMPULSE = 4,   ///< SI: impulse
};
/// Start (or retrigger, for SE) timer `t` with the S5TIME-encoded 16-bit
/// duration `durS5` taken from ACC1 (AWL `SP/SE/SS/SV/SI T n`).
void scl_rt_timer_start(uint32_t t, uint8_t mode, uint64_t durS5);

/// Counter operations for scl_rt_counter_op (AWL `ZV`/`ZR`/`S`/`R C n`).
enum SclRtCounterOp {
  SCL_RT_COUNTER_UP = 0,   ///< ZV: increment (saturates at 999)
  SCL_RT_COUNTER_DOWN = 1, ///< ZR: decrement (floors at 0)
  SCL_RT_COUNTER_SET = 2,  ///< S: set the count to the ACC1 value
  SCL_RT_COUNTER_RESET = 3 ///< R: reset the count to 0
};
/// Apply a counter operation; `value` is only used for SCL_RT_COUNTER_SET.
void scl_rt_counter_op(uint32_t c, uint8_t op, uint64_t value);
/// Builtin timer function blocks TON/TOF/TP (mode 0/1/2) operating on an
/// instance DB buffer with the builtin layout (IN @0, PT @4 i32 BE ms,
/// Q @8, ET @12 i32 BE ms).
void scl_rt_timer_fb(uint8_t mode, uint64_t idbPtr);

/// Timer/counter semantics (simulation): reads of the `T`/`C` areas return
/// the current timer remaining time in ms / the counter value, and bit 0
/// reads return the status (timer Q output / counter != 0). The timer
/// duration is decoded from S5TIME: bits 12-15 select the time base
/// (0=10ms, 1=100ms, 2=1s, 3=10s), bits 0-11 hold the BCD value 0..999.

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
  /* Appended for timers/counters (v2). May be NULL in older host runtimes;
     the sim then handles timer/counter operations itself. */
  void (*timer_start)(uint32_t t, uint8_t mode, uint64_t durS5);
  void (*counter_op)(uint32_t c, uint8_t op, uint64_t value);
  /* Appended for builtin timer FBs TON/TOF/TP (v3). May be NULL in older
     host runtimes; the sim then handles them itself. mode: 0=TON, 1=TOF,
     2=TP. idbPtr is the instance DB buffer (builtin layout: IN @0, PT @4
     (i32 ms, big-endian), Q @8, ET @12 (i32 ms)). */
  void (*timer_fb)(uint8_t mode, uint64_t idbPtr);
  /* Appended for test oracles (v4). May be NULL in older host runtimes;
     the sim then handles asserts itself. op: 0 ==, 1 !=, 2 >, 3 <, 4 >=,
     5 <=. */
  void (*assert_cmp)(uint32_t id, uint8_t op, int64_t actual,
                     int64_t expected);
  void (*assert_cmp_r)(uint32_t id, uint8_t op, double actual,
                       double expected);
  /* Appended for the DB register / address registers / indirect access
     (v5). May be NULL in older host runtimes; the sim then handles them
     itself. */
  void (*set_current_db)(uint32_t db);
  uint32_t (*current_db)(void);
  void (*ar_set)(uint8_t ar, uint64_t ptr);
  uint64_t (*ar_add)(uint8_t ar, uint32_t offBits); // returns new ptr
  int64_t (*read_indirect)(uint64_t ptr, uint8_t widthBytes);
  void (*write_indirect)(uint64_t ptr, int64_t value, uint8_t widthBytes);
  bool (*read_bit_indirect)(uint64_t ptr);
  void (*write_bit_indirect)(uint64_t ptr, bool value);
} SclRtVTable;

/// Register a host vtable; all subsequent ABI calls forward through it.
/// Pass NULL to return to the built-in simulation.
void scl_rt_register(const SclRtVTable *vtable);

/// DB register and address registers (ABI v5), for AWL AUF/OPN, LAR1/LAR2
/// and indirect operands `B [AR1, P#…]`. A packed pointer is
/// area<<56 | db<<40 | byte<<16 | bit<<8 | widthBits. A DB operand with
/// db == 0 refers to the current DB (set by AUF/OPN DB n).
void scl_rt_set_current_db(uint32_t db);
uint32_t scl_rt_current_db(void);
void scl_rt_ar_set(uint8_t ar, uint64_t ptr);
uint64_t scl_rt_ar_get(uint8_t ar);
/// Pointer = AR<ar> + offBits (added to the byte/bit field), repacked.
uint64_t scl_rt_ar_add(uint8_t ar, uint32_t offBits);
/// Indirect access through a packed pointer.
int64_t scl_rt_read_indirect(uint64_t ptr, uint8_t widthBytes);
void scl_rt_write_indirect(uint64_t ptr, int64_t value, uint8_t widthBytes);
bool scl_rt_read_bit_indirect(uint64_t ptr);
void scl_rt_write_bit_indirect(uint64_t ptr, bool value);

/// Test oracles (ABI v4): awlsim-style __ASSERT checks lowered from the AWL
/// frontend. op: 0 ==, 1 !=, 2 >, 3 <, 4 >=, 5 <=. The sim counts failures
/// (and prints them); query with scl_rt_assert_failures().
void scl_rt_assert_cmp(uint32_t id, uint8_t op, int64_t actual,
                       int64_t expected);
void scl_rt_assert_cmp_r(uint32_t id, uint8_t op, double actual,
                         double expected);
/// Number of failed asserts so far (sim only; 0 when a host vtable without
/// assert support is registered).
uint64_t scl_rt_assert_failures(void);

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
