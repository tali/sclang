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
  SCL_RT_AREA_L = 'L',  ///< local data  (%L / AWL LW/LB/LD), per-call frame
  SCL_RT_AREA_V = 'V',  ///< parent-frame local data (S7 VL, 0x87): the caller's L frame
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

/// Timer modes for scl_rt_timer_start (the S7 AWL timer instructions,
/// awlsim-compatible edge semantics; `in` is the RLO/VKE input).
enum SclRtTimerMode {
  SCL_RT_TIMER_PULSE = 0,      ///< SP: pos edge starts, Q while running
  SCL_RT_TIMER_ON_DELAY = 1,   ///< SE: pos edge starts, Q after expiry, IN=0 stops
  SCL_RT_TIMER_LATCH_ON_DELAY = 2, ///< SS: pos edge starts, Q after expiry (latched)
  SCL_RT_TIMER_EXT_PULSE = 3,  ///< SV: pos edge starts/retriggers, Q while running
  SCL_RT_TIMER_IMPULSE = 4,    ///< SI: pos edge starts, IN=0 stops, Q while running
  SCL_RT_TIMER_OFF_DELAY = 5,  ///< SA: neg edge starts, Q=1 while IN or running
};
/// Run the timer instruction: edge-detect on `in`, start/retrigger timer `t`
/// with the S5TIME-encoded 16-bit duration `durS5` from ACC1
/// (AWL `SP/SE/SS/SV/SI/SA T n`).
void scl_rt_timer_start(uint32_t t, uint8_t mode, uint64_t durS5,
                        uint8_t in);
/// Reset timer `t` (AWL `R T n`): running = 0, Q = 0, remaining = 0.
void scl_rt_timer_reset(uint32_t t);
/// FR T n: on a positive `in` edge, reset the timer's edge memory (prevIn).
void scl_rt_timer_fr(uint32_t t, uint8_t in);

/// Counter operations for scl_rt_counter_op (AWL `ZV`/`ZR`/`S`/`R C n`).
enum SclRtCounterOp {
  SCL_RT_COUNTER_UP = 0,   ///< ZV: increment on positive `in` edge (max 999)
  SCL_RT_COUNTER_DOWN = 1, ///< ZR: decrement on positive `in` edge (min 0)
  SCL_RT_COUNTER_SET = 2,  ///< S: set count to BCD-decoded ACC1 on pos edge
  SCL_RT_COUNTER_RESET = 3, ///< R: reset the count to 0
  SCL_RT_COUNTER_FR = 4    ///< FR: on pos `in` edge, reset the edge memory
};
/// Apply a counter operation; `value` is only used for SCL_RT_COUNTER_SET,
/// `in` is the RLO/VKE input (edge-detected per counter and operation).
void scl_rt_counter_op(uint32_t c, uint8_t op, uint64_t value, uint8_t in);

/// Advance the virtual clock by `ms` (awlsim `__SLEEP n` test directive).
void scl_rt_sleep(uint32_t ms);

/// DB info pseudo operands (AWL DBLG/DBNO/DILG/DINO). kind: 0 = DBLG (DB
/// size in bytes), 1 = DBNO (DB number), 2 = DILG, 3 = DINO. db = 0 refers
/// to the current DB (set by AUF/OPN).
int64_t scl_rt_db_info(uint32_t kind, uint32_t db);
/// Builtin timer function blocks TON/TOF/TP (mode 0/1/2) operating on an
/// instance DB buffer with the builtin layout (IN @0, PT @4 i32 BE ms,
/// Q @8, ET @12 i32 BE ms).
void scl_rt_timer_fb(uint8_t mode, uint64_t idbPtr);

/// Timer/counter semantics (simulation, awlsim-compatible): reads of the
/// `T` area return the remaining time in units of the active time base
/// (binary time value, as `L T n`), bit 0 reads return the Q status; reads
/// of the `C` area return the counter value (binary), bit 0 returns
/// counter != 0. Durations are S5TIME-encoded: bits 12-15 select the time
/// base (0=10ms, 1=100ms, 2=1s, 3=10s), bits 0-11 the BCD value 0..999.

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
  void (*timer_start)(uint32_t t, uint8_t mode, uint64_t durS5,
                      uint8_t in);
  void (*counter_op)(uint32_t c, uint8_t op, uint64_t value, uint8_t in);
  void (*timer_reset)(uint32_t t);
  void (*timer_fr)(uint32_t t, uint8_t in);
  void (*sleep)(uint32_t ms);
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
  void (*sfb)(uint32_t n); ///< run SFB n on its instance area
  void (*sfc)(uint32_t n); ///< run SFC n (params staged at the marker area)
  void (*ob_start)(uint32_t obNum); ///< write the OB-temp preset header
  uint64_t (*ar_add)(uint8_t ar, uint32_t offBits); // returns new ptr
  /* Appended for LC T n (v10): timer current value in S5TIME format. May be
     NULL in older host runtimes; the sim then handles it itself. */
  int64_t (*timer_lc)(uint32_t t);
  int64_t (*read_indirect)(uint64_t ptr, uint8_t widthBytes);
  void (*write_indirect)(uint64_t ptr, int64_t value, uint8_t widthBytes);
  bool (*read_bit_indirect)(uint64_t ptr);
  void (*write_bit_indirect)(uint64_t ptr, bool value);
  /* Appended for IEC library FCs called by name (v6): LEN 21, EQ_DT 9,
     GE_DT 12, DELETE 4 (params staged at the marker area, like SFCs). */
  void (*iec)(uint32_t n);
  /* Appended for dynamic T/C access through TIMER/COUNTER-typed block
     variables (v7): the variable stores the timer/counter NUMBER; the
     frontend packs a dynamic pointer (area 'T'/'C', runtime byte = the
     number) with pack_area and the ops below decode it back. */
  uint64_t (*pack_area)(uint8_t area, uint32_t byte, uint8_t bit);
  void (*counter_op_ptr)(uint64_t ptr, uint8_t op, uint64_t value,
                         uint8_t in);
  void (*timer_start_ptr)(uint64_t ptr, uint8_t mode, uint64_t durS5,
                          uint8_t in);
  void (*timer_reset_ptr)(uint64_t ptr);
  void (*fr_ptr)(uint64_t ptr, uint8_t in);
  /* Appended for the status word (v8): bits NER=0 VKE/RLO=1 STA=2 OR=3
     OS=4 OV=5 A0/CC0=6 A1/CC1=7 BIE/BR=8 (awlsim statusword). */
  uint32_t (*stw_get)(void);
  void (*stw_set)(uint32_t word);
  void (*stw_reset)(void);
  void (*stw_set_bit)(uint8_t bit, uint8_t value);
  void (*stw_cmp_i)(uint8_t pred, int64_t lhs, int64_t rhs);
  void (*stw_cmp_f)(uint8_t pred, double lhs, double rhs);
  /* v9: arithmetic STW recording + real->int conversion (awlsim semantics) */
  void (*stw_sign_i)(int64_t value, int64_t lhs, int64_t rhs, uint8_t op, int32_t range); // sign from value; OV/OS if untruncated lhs-op-rhs out of +-range
  void (*stw_sign_f)(double value);    // setForFloatingPointDWord classification
  void (*stw_div_i)(int64_t divisor, int64_t result); // div by 0: UO+OV+OS
  void (*stw_div_f)(double divisor, double result);
  int64_t (*f2i)(double value, int64_t orig, uint8_t mode); // TRUNC/RND/..: OV/OS + keep orig if out of range
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
int64_t scl_rt_timer_lc(uint32_t n); // LC T n: S5TIME-format current value
void scl_rt_ar_set(uint8_t ar, uint64_t ptr);
uint64_t scl_rt_ar_get(uint8_t ar);
uint32_t scl_rt_ar_raw(uint8_t ar);
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

/// L-stack frames (S7 local data): every called block pushes a frame of
/// `bytes` bytes; L-area accesses are relative to the current frame base.
/// The block lowering emits scl_rt_l_push at function entry and
/// scl_rt_l_pop before every return. Sim-only (a host vtable receives raw
/// L offsets and manages its own frames).
void scl_rt_frame_set_fb(uint8_t isFb);
void scl_rt_l_push(uint32_t bytes);
/// Copy `bytes` of the FC parameter area between the marker staging area
/// (offset 8192, where CALL stores the actual parameters) and the callee's
/// own L frame at offset 8192: param_in after l_push, param_out before
/// l_pop. This keeps per-invocation parameters for recursive FCs and
/// propagates output params (RET_VAL / VAR_IN_OUT) back to the call site.
void scl_rt_param_in(uint32_t bytes);
void scl_rt_param_out(uint32_t bytes);
/// Run a system function block (SFB 0-5: CTU/CTD/CTUD counters, TON/TOF/TP
/// timers) on its instance area in the DI-register DB at the AR2 base
/// offset (multi-instance base), like awlsim's systemblocks.
void scl_rt_sfb(uint32_t n);
/// Run a system function (SFC 21 FILL / 47 WAIT / 64 TIME_TCK) on the
/// parameter staging area in the marker area at kParamBase, like awlsim's
/// systemblocks. Sim-only; hosts may override via the vtable.
void scl_rt_sfc(uint32_t n);
/// Run an IEC library FC called by name (awlsim library/iec: LEN 21,
/// EQ_DT 9, GE_DT 12, DELETE 4) on the parameter staging area in the
/// marker area at kParamBase, like the SFCs. Sim-only; hosts may override
/// via the vtable.
void scl_rt_iec(uint32_t n);
/// Pack an internal packed pointer with a runtime byte offset (dynamic T/C
/// access through TIMER/COUNTER-typed variables). Sim-only.
uint64_t scl_rt_pack_area(uint8_t area, uint32_t byte, uint8_t bit);
/// Counter/timer operations through a dynamic packed pointer (the byte
/// field holds the timer/counter number). Sim-only.
void scl_rt_counter_op_ptr(uint64_t ptr, uint8_t op, uint64_t value,
                           uint8_t in);
void scl_rt_timer_start_ptr(uint64_t ptr, uint8_t mode, uint64_t durS5,
                            uint8_t in);
void scl_rt_timer_reset_ptr(uint64_t ptr);
void scl_rt_fr_ptr(uint64_t ptr, uint8_t in);
/// Status word (STW) access, awlsim statusword bit numbering. Sim-only.
uint32_t scl_rt_stw_get(void);
void scl_rt_stw_set(uint32_t word);
void scl_rt_stw_reset(void);
void scl_rt_stw_set_bit(uint8_t bit, uint8_t value);
/// Record a comparison into A0/A1 (and set VKE). pred: 0 eq, 1 ne, 2 lt,
/// 3 le, 4 gt, 5 ge (arith::CmpIPredicate).
void scl_rt_stw_cmp_i(uint8_t pred, int64_t lhs, int64_t rhs);
void scl_rt_stw_cmp_f(uint8_t pred, double lhs, double rhs);
/// Arithmetic STW recording (awlsim statusword). Sim-only.
void scl_rt_stw_sign_i(int64_t value, int64_t lhs, int64_t rhs, uint8_t op, int32_t range);
void scl_rt_stw_sign_f(double value);
void scl_rt_stw_div_i(int64_t divisor, int64_t result);
void scl_rt_stw_div_f(double divisor, double result);
/// TRUNC/RND/RND+/RND- (mode 0..3): real -> int. Out of int32 range: keep
/// `orig` in the ACCU and set OV/OS. Sim-only.
int64_t scl_rt_f2i(double value, int64_t orig, uint8_t mode);
/// Write the S7 OB-temp preset header (EV_CLASS/SCAN/PRIORITY/OB_NUMBR/
/// cycle times/DATE_TIME) into the current L frame at OB entry
/// (awlsim obtemp.OB1TempPresets; only OB 1 has presets). Sim-only.
void scl_rt_ob_start(uint32_t obNum);
/// __FEATURE t [, v]: get/set CPU feature flags (t=0 nrAccus, t=1 OB-temp
/// presets). value < 0 means get. Returns the current value. Sim-only.
int32_t scl_rt_feature(uint32_t target, int32_t value);
void scl_rt_l_pop(void);

/// Set the DI register (AUF/OPN DI n). DINO/DILG read it. Sim-only.
void scl_rt_set_current_di(uint32_t db);
/// Read the DI register (DI-area operands DIX/DIB/DIW/DID). Sim-only.
uint32_t scl_rt_current_di(void);

/// Build an S7 32-bit pointer value (area<<24|db<<16|byte<<3|bit).
uint64_t scl_rt_pack_s7(uint32_t area, uint32_t db, uint32_t byte,
                        uint32_t bit);

/// Simulation-only configuration (NOT part of the fixed ABI): set the size
/// in bytes of a simulated area (I/Q/M/T/C). DB areas grow on demand unless
/// a size is set with scl_rt_sim_set_db_size().
void scl_rt_sim_set_area_size(uint8_t area, uint64_t size);
/// Set the size in bytes of a simulated data block (grown on demand if unset).
void scl_rt_sim_set_db_size(uint32_t db, uint64_t size);
/// Register the highest referenced byte of a DB (frontend pre-scan), used
/// for ANY-pointer (SFC 21) bounds checks on undeclared DBs. Sim-only.
void scl_rt_sim_db_ref(uint32_t db, uint64_t bytes);
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
