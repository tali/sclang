//===- SimRuntime.cpp - Simulation runtime for the SCL C ABI -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Default runtime bound by the `sclang` JIT driver (ADR 002): static per-area
// byte buffers plus lazily grown DB buffers and a virtual clock. If a host
// registers an SclRtVTable via scl_rt_register(), all calls forward to it.
//
// All multi-byte access is BIG-ENDIAN (ADR 001), matching S7 hardware.
//
//===----------------------------------------------------------------------===//

#include "sclang/Runtime/Runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>

namespace {

/// Default size of a simulated area (I/Q/M/T/C) in bytes; configurable via
/// scl_rt_sim_set_area_size() (driver CLI `--area-size`).
constexpr uint64_t kDefaultAreaSize = 65536;
/// Default size of the L (local data) area: large enough for a deep FC call
/// stack, since each frame reserves its parameter area at kParamBase.
constexpr uint64_t kDefaultLAreaSize = 1u << 20;

// Classic AWL timer, awlsim-compatible (awlsim/core/timers.py): edge-triggered
// start on the RLO input, remaining time in units of the active time base,
// Q status set/cleared at the deadline depending on the mode.
struct TimerState {
  uint8_t mode = 0;             // SclRtTimerMode
  bool prevIn = false;          // previous RLO input (edge detection)
  bool prevFR = false;          // previous RLO input of the FR instruction
  bool deadlineSetsStatus = false; // SE/SS: Q=1 at expiry, else Q=0
  bool running = false;
  bool status = false;          // Q output
  uint64_t deadlineMs = 0;      // clock at expiry
  uint64_t remainingMs = 0;     // cached remaining time (when stopped)
  uint8_t timebase = 0;         // S5TIME base of the last start (0=10ms)
  uint64_t durMs = 0;           // last preset in ms
  uint64_t tvRaw = 0;           // TV register: last written value (timebase
                                // units), returned by reads while not running
};

// Classic AWL counter, awlsim-compatible (awlsim/core/counters.py):
// ZV/ZR count on positive RLO edges, S sets the BCD-decoded ACC1 on a
// positive edge, R resets.
struct CounterState {
  uint32_t value = 0;           // count (0..999)
  bool prevS = false, prevZV = false, prevZR = false;
  bool prevFR = false;          // previous RLO input of the FR instruction
};

struct SimState {
  uint16_t stw = 0; // status word: NER=0 VKE=1 STA=2 OR=3 OS=4 OV=5 A0=6 A1=7 BIE=8
  std::vector<uint8_t> areaBytes[256];   // indexed by area code (ASCII)
  uint64_t areaSize[256] = {};           // configured size, 0 = default
  std::map<uint32_t, std::vector<uint8_t>> dbs; // DB number -> bytes
  std::map<uint32_t, uint64_t> dbSize;   // configured DB sizes, 0 = grow lazily
  std::map<uint32_t, uint64_t> dbRefHighWater; // highest referenced DB byte
  const SclRtVTable *vtable = nullptr;
  uint64_t clockMs = 0;
  uint64_t lastCycleMs = 0, minCycleMs = 0, maxCycleMs = 0; // OB1 presets
  uint32_t nrAccus = 4;      // __FEATURE 0 (awlsim tests run 4-accu CPUs)
  bool stopRequested = false; // SFC 46 (STOP) / SFC -3 (SHUTDOWN)
  uint32_t accus[4] = {0, 0, 0, 0}; // ACCU1..4 (index 0 = ACCU1), persistent across labels/blocks
  bool obTempPresets = false; // __FEATURE 1
  TimerState timers[256];                // indexed by timer number
  CounterState counters[256];            // indexed by counter number
  // builtin timer FB (TON/TOF/TP) state, keyed by instance DB buffer address
  struct TimerFbState {
    bool prevIn = false;
    bool running = false;
    bool finished = false; // TOF: PT expired while IN = FALSE (ET holds PT)
    bool qLatch = false;   // TON: Q rose once and stays TRUE until IN falls
    uint64_t startMs = 0;
  };
  std::map<uint64_t, TimerFbState> timerFbs;
  uint64_t assertFailures = 0; // failed __ASSERT oracle checks
  // ABI v5: DB register (AUF/OPN) and address registers AR1/AR2 holding
  // packed pointers (area<<56|db<<40|byte<<16|bit<<8|widthBits).
  uint32_t currentDb = 0; // 0 = none (DB 0 operands stay literal)
  uint32_t currentDi = 0; // DI register (AUF/OPN DI n); DINO/DILG read it
  uint64_t ars[2] = {0, 0};
  // L-stack (S7 local data): a stack of frame bases; L-area byte offsets
  // are relative to lBase (the base of the innermost frame).
  std::vector<uint64_t> lFrames;
  std::vector<uint64_t> lSizes; // frame size at the same stack depth
  uint64_t lBase = 0;
  uint64_t curFrameSize = 0;
  uint8_t curIsFb = 0; // awlsim adds the AR2 base to DI accesses only in FBs
  // call-saved block-call state, saved/restored with the L frame (S7 restores
  // the address registers and DB/DI registers when a block returns)
  struct FrameSave {
    uint64_t ar1, ar2;
    uint32_t db, di;
    uint8_t isFb;
  };
  std::vector<FrameSave> frameSaves;
  // SFB3-5 timer state, keyed by (DI db << 32 | AR2 base byte)
  struct SfbTimerState {
    bool prevIn = false;
    bool running = false;
    uint64_t startMs = 0;
    bool done = false;
  };
  std::map<uint64_t, SfbTimerState> sfbTimers;
  // SFB0-2 previous LD bit, keyed by (DI db << 32 | AR2 base byte)
  std::map<uint64_t, bool> sfbPrevLd;
};

/// Decode an S5TIME-encoded 16-bit value to milliseconds.
static uint64_t s5timeToMs(uint64_t s5) {
  static const uint64_t baseMs[4] = {10, 100, 1000, 10000};
  uint64_t base = (s5 >> 12) & 3;
  uint64_t bcd = s5 & 0xFFF;
  uint64_t val = ((bcd >> 8) & 0xF) * 100 + ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
  return val * baseMs[base];
}

static const uint64_t kTimeBaseMs[4] = {10, 100, 1000, 10000};

/// Lazy deadline check (awlsim Timer.__checkDeadline).
static void timerCheck(TimerState &t, uint64_t now) {
  if (!t.running)
    return;
  t.remainingMs = t.deadlineMs > now ? t.deadlineMs - now : 0;
  if (t.remainingMs == 0) {
    t.running = false;
    t.status = t.deadlineSetsStatus;
  }
}

/// (Re)start the timer with an S5TIME-encoded duration (awlsim Timer.__start).
static void timerStart(TimerState &t, uint64_t durS5, uint64_t now) {
  t.timebase = (uint8_t)((durS5 >> 12) & 3);
  t.durMs = s5timeToMs(durS5);
  t.deadlineMs = now + t.durMs;
  t.remainingMs = t.durMs;
  t.running = true;
}

/// Binary timer value in units of the active time base (awlsim
/// Timer.getTimevalBin, used by `L T n`).
static uint64_t timerValueBin(TimerState &t, uint64_t now) {
  timerCheck(t, now);
  uint64_t unit = kTimeBaseMs[t.timebase & 3];
  return (t.remainingMs + unit / 2) / unit;
}

/// Decode a BCD counter value (awlsim Counter.setValueBCD).
static uint32_t bcdDecode(uint64_t v) {
  uint32_t a = v & 0xF, b = (v >> 4) & 0xF, c = (v >> 8) & 0xF;
  uint32_t val = a + b * 10 + c * 100;
  return val > 999 ? 999 : val;
}

SimState &state() {
  static SimState s;
  return s;
}


/// Frame base for an L/V-area access: 'L' is the innermost frame, 'V' (VL,
/// S7 area 0x87) the parent frame (awlsim: VL = caller's L-stack frame).
static uint64_t frameBaseOf(uint8_t area, const SimState &s) {
  if (area == SCL_RT_AREA_V)
    return s.lFrames.empty() ? 0 : s.lFrames.back();
  return s.lBase;
}

/// Byte offset of the AR2 multi-instance base (awlsim adds the AR2 base to
/// every DI-area access, see cpu.py getArea_DI/setArea_DI).
static uint32_t ar2ByteOff() {
  if (!state().curIsFb)
    return 0; // awlsim adds the AR2 base to DI accesses only inside FBs
  uint64_t p = state().ars[1];
  if ((p >> 56) != 0)
    return (uint32_t)((p >> 16) & 0xFFFF); // internal packed pointer
  return (uint32_t)((p >> 3) & 0x1FFF);     // raw S7 pointer (area at 24-31)
}

/// Return the byte buffer for an access, growing DBs on demand and checking
/// bounds of the fixed-size areas.
// Returns the area buffer; `byteOff` is adjusted in place for frame-relative
// areas (L), so callers must index the buffer with the adjusted byteOff.
std::vector<uint8_t> &bufferFor(uint8_t area, uint32_t db, uint32_t &byteOff,
                                uint32_t width) {
  auto &s = state();
  if (area == SCL_RT_AREA_DB) {
    auto &buf = s.dbs[db];
    auto sizeIt = s.dbSize.find(db);
    if (sizeIt != s.dbSize.end()) {
      // explicitly sized DB: bounds-checked like the other areas
      if ((uint64_t)byteOff + width > sizeIt->second) {
        fprintf(stderr,
                "scl_rt: out-of-bounds access: DB %u byte %u width %u, "
                "DB size %llu\n",
                db, byteOff, width, (unsigned long long)sizeIt->second);
        abort();
      }
      if (buf.size() < sizeIt->second)
        buf.resize(sizeIt->second, 0);
      return buf;
    }
    if (buf.size() < (uint64_t)byteOff + width)
      buf.resize((uint64_t)byteOff + width, 0);
    return buf;
  }
  if (area == SCL_RT_AREA_L || area == SCL_RT_AREA_V) {
    // L/V access is frame-relative; both share the single L-stack buffer
    byteOff += (uint32_t)frameBaseOf(area, s);
    area = SCL_RT_AREA_L;
  }
  auto &buf = s.areaBytes[area];
  if (buf.empty()) {
    uint64_t size = s.areaSize[area] ? s.areaSize[area]
                     : area == SCL_RT_AREA_L ? kDefaultLAreaSize
                                              : kDefaultAreaSize;
    buf.resize(size, 0);
  }
  if ((uint64_t)byteOff + width > buf.size()) {
    fprintf(stderr,
            "scl_rt: out-of-bounds access: area '%c' byte %u width %u, "
            "area size %zu\n",
            area, byteOff, width, buf.size());
    abort();
  }
  return buf;
}

} // namespace

// LC T n: timer current value in S5TIME format — BCD-packed value in bits
// 0-11 plus the time base in bits 12-13 (awlsim Timer.getTimevalS5T).
int64_t scl_rt_timer_lc(uint32_t n) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_lc)
    return s.vtable->timer_lc(n);
  auto &tm = s.timers[n];
  timerCheck(tm, s.clockMs);
  uint64_t unit = kTimeBaseMs[tm.timebase & 3];
  uint64_t bin = unit ? (tm.remainingMs + unit / 2) / unit : 0;
  if (bin > 999)
    bin = 999;
  uint32_t bcd = (uint32_t)(bin % 10) | ((uint32_t)((bin / 10) % 10) << 4) |
                 ((uint32_t)((bin / 100) % 10) << 8);
  return (int64_t)(((uint64_t)(tm.timebase & 3) << 12) | bcd);
}

extern "C" {

// SPEC[AWL] IEC 61131-3 STL memory areas (P/Q/M/DB/DI/L/T/C, big-endian S7 storage — docs/decisions/001) -> scl_rt_read/read_bit (C ABI, docs/decisions/002)
int64_t scl_rt_read(uint8_t area, uint32_t db, uint32_t byteOff,
                    uint8_t widthBytes) {
  if (state().vtable)
    return state().vtable->read(area, db, byteOff, widthBytes);
  if (widthBytes == 0 || widthBytes > 8) {
    fprintf(stderr, "scl_rt_read: invalid width %u (area '%c' db %u byte %u)\n", widthBytes, area, db, byteOff);
    abort();
  }
  if (area == SCL_RT_AREA_DB && db == 0xFFFF) {
    db = state().currentDi;  // DI-area pointer: DB from the DI register
    byteOff += ar2ByteOff(); // multi-instance base offset from AR2
  }
  if (area == SCL_RT_AREA_DB && db == 0 && state().currentDb != 0)
    db = state().currentDb; // DB 0 operand = current DB (AUF/OPN)
  // Timers/counters: the value is computed from the sim state (remaining
  // time in ms / counter value), not stored in the area buffer.
  if (area == SCL_RT_AREA_T && widthBytes > 1) {
    auto &tm = state().timers[byteOff];
    timerCheck(tm, state().clockMs);
    // while running (or stopped mid-run): remaining time in timebase units
    // (awlsim getTimevalBin); a T T n / %TWn := write loads the TV register
    // and is returned verbatim while not running
    uint64_t unit = kTimeBaseMs[tm.timebase & 3];
    uint64_t v = tm.running ? (tm.remainingMs + unit / 2) / unit
                            : (tm.tvRaw != 0 ? tm.tvRaw
                                             : (tm.remainingMs + unit / 2) / unit);
    if (widthBytes < 8) {
      uint64_t signBit = 1ull << (widthBytes * 8 - 1);
      if (v & signBit)
        v |= ~0ull << (widthBytes * 8);
    }
    return (int64_t)v;
  }
  if (area == SCL_RT_AREA_C && widthBytes > 1)
    return (int64_t)state().counters[byteOff].value;
  auto &buf = bufferFor(area, db, byteOff, widthBytes);
  // big-endian: byte 0 is the most significant byte
  uint64_t v = 0;
  for (uint32_t i = 0; i < widthBytes; ++i)
    v = (v << 8) | buf[byteOff + i];
  // sign-extend from widthBytes*8 bits
  if (widthBytes < 8) {
    uint64_t signBit = 1ull << (widthBytes * 8 - 1);
    if (v & signBit)
      v |= ~0ull << (widthBytes * 8);
  }
  return (int64_t)v;
}

// SPEC[AWL] IEC 61131-3 STL memory areas, stores -> scl_rt_write/write_bit (C ABI)
void scl_rt_write(uint8_t area, uint32_t db, uint32_t byteOff,
                  uint8_t widthBytes, uint64_t value) {
  if (state().vtable) {
    state().vtable->write(area, db, byteOff, widthBytes, value);
    return;
  }
  if (widthBytes == 0 || widthBytes > 8) {
    fprintf(stderr, "scl_rt_write: invalid width %u\n", widthBytes);
    abort();
  }
  if (area == SCL_RT_AREA_DB && db == 0xFFFF) {
    db = state().currentDi;  // DI-area pointer: DB from the DI register
    byteOff += ar2ByteOff(); // multi-instance base offset from AR2
  }
  if (area == SCL_RT_AREA_DB && db == 0 && state().currentDb != 0)
    db = state().currentDb; // DB 0 operand = current DB (AUF/OPN)
  // Writing a counter value (e.g. SCL `%Cn := v`) sets the count; writing a
  // timer value (e.g. `%TWn := v`) sets the preset (TV).
  if (area == SCL_RT_AREA_C && widthBytes > 1) {
    state().counters[byteOff].value = (uint32_t)(value > 999 ? 999 : value);
    return;
  }
  if (area == SCL_RT_AREA_T && widthBytes > 1) {
    // T T n: load the TV register; restarts a running timer (awlsim
    // Timer.setTimevalS5T)
    auto &tm = state().timers[byteOff];
    tm.tvRaw = value;
    if (tm.running)
      timerStart(tm, value, state().clockMs);
    else {
      tm.timebase = (uint8_t)((value >> 12) & 3);
      tm.durMs = s5timeToMs(value);
    }
    return;
  }
  auto &buf = bufferFor(area, db, byteOff, widthBytes);
  // big-endian: the low byte of the value goes to the highest offset
  for (uint32_t i = 0; i < widthBytes; ++i)
    buf[byteOff + widthBytes - 1 - i] = (uint8_t)(value >> (8 * i));
}

bool scl_rt_read_bit(uint8_t area, uint32_t db, uint32_t byteOff,
                     uint8_t bit) {
  if (state().vtable)
    return state().vtable->read_bit(area, db, byteOff, bit);
  if (bit > 7) {
    fprintf(stderr, "scl_rt_read_bit: invalid bit %u\n", bit);
    abort();
  }
  if (area == SCL_RT_AREA_DB && db == 0xFFFF) {
    db = state().currentDi;  // DI-area pointer: DB from the DI register
    byteOff += ar2ByteOff(); // multi-instance base offset from AR2
  }
  if (area == SCL_RT_AREA_DB && db == 0 && state().currentDb != 0)
    db = state().currentDb; // DB 0 operand = current DB (AUF/OPN)
  // Timer/counter status bit (AWL `U T n` / `U Z n`).
  if (bit == 0 && area == SCL_RT_AREA_T) {
    timerCheck(state().timers[byteOff], state().clockMs);
    return state().timers[byteOff].status;
  }
  if (bit == 0 && area == SCL_RT_AREA_C)
    return state().counters[byteOff].value > 0;
  auto &buf = bufferFor(area, db, byteOff, 1);
  return (buf[byteOff] >> bit) & 1;
}

void scl_rt_write_bit(uint8_t area, uint32_t db, uint32_t byteOff,
                      uint8_t bit, bool value) {
  if (state().vtable) {
    state().vtable->write_bit(area, db, byteOff, bit, value);
    return;
  }
  if (bit > 7) {
    fprintf(stderr, "scl_rt_write_bit: invalid bit %u\n", bit);
    abort();
  }
  if (area == SCL_RT_AREA_DB && db == 0xFFFF) {
    db = state().currentDi;  // DI-area pointer: DB from the DI register
    byteOff += ar2ByteOff(); // multi-instance base offset from AR2
  }
  if (area == SCL_RT_AREA_DB && db == 0 && state().currentDb != 0)
    db = state().currentDb; // DB 0 operand = current DB (AUF/OPN)
  auto &buf = bufferFor(area, db, byteOff, 1);
  if (value)
    buf[byteOff] |= (uint8_t)(1u << bit);
  else
    buf[byteOff] &= (uint8_t)~(1u << bit);
}

void scl_rt_update_inputs(void) {
  if (state().vtable)
    state().vtable->update_inputs();
  // In the simulation the process image is simply the area buffers; there is
  // nothing to copy. Hosts use this hook to push their input image.
}

void scl_rt_update_outputs(void) {
  if (state().vtable)
    state().vtable->update_outputs();
  // See scl_rt_update_inputs().
}

void scl_rt_cycle(uint64_t elapsedMs) {
  if (state().vtable) {
    state().vtable->cycle(elapsedMs);
    return;
  }
  state().clockMs += elapsedMs;
}

void scl_rt_timer_start(uint32_t t, uint8_t mode, uint64_t durS5,
                        uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_start) {
    s.vtable->timer_start(t, mode, durS5, in);
    return;
  }
  if (t > 255) {
    fprintf(stderr, "scl_rt_timer_start: invalid timer %u\n", t);
    abort();
  }
  TimerState &tm = s.timers[t];
  bool posEdge = in && !tm.prevIn;
  bool negEdge = !in && tm.prevIn;
  uint64_t now = s.clockMs;
  switch (mode) {
  case SCL_RT_TIMER_ON_DELAY: // SE
    tm.deadlineSetsStatus = true;
    if (in) {
      if (posEdge)
        timerStart(tm, durS5, now);
    } else {
      timerCheck(tm, now);
      tm.running = false;
      tm.status = false;
    }
    break;
  case SCL_RT_TIMER_LATCH_ON_DELAY: // SS
    tm.deadlineSetsStatus = true;
    if (posEdge)
      timerStart(tm, durS5, now);
    break;
  case SCL_RT_TIMER_EXT_PULSE: // SV
    tm.deadlineSetsStatus = false;
    if (posEdge) {
      tm.status = true;
      timerStart(tm, durS5, now);
    }
    break;
  case SCL_RT_TIMER_IMPULSE: // SI
    tm.deadlineSetsStatus = false;
    if (in) {
      if (posEdge) {
        tm.status = true;
        timerStart(tm, durS5, now);
      }
    } else {
      timerCheck(tm, now);
      tm.running = false;
      tm.status = false;
    }
    break;
  case SCL_RT_TIMER_OFF_DELAY: // SA
    tm.deadlineSetsStatus = false;
    if (posEdge) {
      timerCheck(tm, now);
      tm.status = true;
      tm.running = false;
    }
    if (negEdge) {
      tm.status = true;
      timerStart(tm, durS5, now);
    }
    break;
  default: // SP: pulse — pos edge starts, Q while running
    tm.deadlineSetsStatus = false;
    if (posEdge) {
      tm.status = true;
      timerStart(tm, durS5, now);
    }
    break;
  }
  tm.prevIn = in != 0;
}

void scl_rt_timer_reset(uint32_t t) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_reset) {
    s.vtable->timer_reset(t);
    return;
  }
  if (t > 255) {
    fprintf(stderr, "scl_rt_timer_reset: invalid timer %u\n", t);
    abort();
  }
  TimerState &tm = s.timers[t];
  tm.running = false;
  tm.status = false;
  tm.remainingMs = 0;
}

void scl_rt_timer_fr(uint32_t t, uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_fr) {
    s.vtable->timer_fr(t, in);
    return;
  }
  if (t > 255) {
    fprintf(stderr, "scl_rt_timer_fr: invalid timer %u\n", t);
    abort();
  }
  TimerState &tm = s.timers[t];
  if (in && !tm.prevFR)
    tm.prevIn = false; // re-enable: next timer instruction sees a pos edge
  tm.prevFR = in != 0;
}

int64_t scl_rt_db_info(uint32_t kind, uint32_t db) {
  auto &s = state();
  switch (kind) {
  case 0: { // DBLG: size of the DB in the DB register
    uint32_t d = db == 0 ? s.currentDb : db;
    auto sizeIt = s.dbSize.find(d);
    if (sizeIt != s.dbSize.end())
      return (int64_t)sizeIt->second;
    auto it = s.dbs.find(d);
    return it != s.dbs.end() ? (int64_t)it->second.size() : 0;
  }
  case 1: // DBNO: DB number in the DB register
    return (int64_t)(db == 0 ? s.currentDb : db);
  case 2: { // DILG: size of the DB in the DI register
    uint32_t d = db == 0 ? s.currentDi : db;
    auto sizeIt = s.dbSize.find(d);
    if (sizeIt != s.dbSize.end())
      return (int64_t)sizeIt->second;
    auto it = s.dbs.find(d);
    return it != s.dbs.end() ? (int64_t)it->second.size() : 0;
  }
  default: // DINO: DB number in the DI register
    return (int64_t)(db == 0 ? s.currentDi : db);
  }
}

uint64_t scl_rt_pack_s7(uint32_t area, uint32_t db, uint32_t byte,
                        uint32_t bit) {
  return ((uint64_t)(area & 0xFF) << 24) | ((uint64_t)(db & 0xFF) << 16) |
         ((uint64_t)(byte & 0x1FFF) << 3) | (bit & 7);
}

// Set whether the current L-stack frame belongs to an FB (awlsim
// __fetchDI/__storeDI add the AR2 multi-instance base only for FB blocks).
void scl_rt_frame_set_fb(uint8_t isFb) { state().curIsFb = isFb; }

// SPEC[AWL] Siemens AWL manual local data stack (L/VL frames per block call, incl. FC param area and 256-depth guard) -> scl_rt_l_push/l_pop/param_in/param_out
void scl_rt_l_push(uint32_t bytes) {
  auto &s = state();
  // awlsim raises when callStackDepth >= specs.callStackSize (default 256);
  // without a guard the JIT'd recursion overflows the native stack (SIGSEGV)
  if (s.lFrames.size() >= 256) {
    fprintf(stderr, "scl_rt: call stack overflow (depth 256, awlsim "
            "callStackSize) — infinite recursion\n");
    exit(2);
  }
  s.lFrames.push_back(s.lBase);
  s.lSizes.push_back(s.curFrameSize);
  s.frameSaves.push_back({s.ars[0], s.ars[1], s.currentDb, s.currentDi,
                          s.curIsFb});
  // Frames tile the L-stack without overlap: the new frame starts after the
  // CURRENT (caller's) frame. Frames are even-aligned and always advance by
  // at least 2 bytes, so blocks without TEMP vars still get a separate L
  // frame (a callee must not clobber the caller's LB/LW/LD local data,
  // including the VL parameter/literal area past kParamBase).
  uint64_t adv = (s.curFrameSize + 1) & ~uint64_t(1);
  s.lBase += adv < 2 ? 2 : adv;
  s.curFrameSize = bytes;
}

void scl_rt_l_pop(void) {
  auto &s = state();
  if (!s.lFrames.empty()) {
    s.lBase = s.lFrames.back();
    s.lFrames.pop_back();
    s.curFrameSize = s.lSizes.back();
    s.lSizes.pop_back();
    if (!s.frameSaves.empty()) {
      auto &f = s.frameSaves.back();
      s.ars[0] = f.ar1;
      s.ars[1] = f.ar2;
      s.currentDb = f.db;
      s.currentDi = f.di;
      s.curIsFb = f.isFb;
      s.frameSaves.pop_back();
    }
  }
}

// Base of the FC parameter area within an L frame; must match kLocalBase
// in lib/AwlGen/AwlGen.cpp.
static const uint32_t kParamBase = 8192;

// Copy the FC parameter area between the absolute marker staging area
// (where the call site stores the actual parameters) and the CALLER's L
// frame at kParamBase (awlsim: the FC parameter area lives in the caller's
// L-stack and is reached from the callee via the VL area, 0x87).
// param_in runs after the callee's scl_rt_l_push, param_out before
// scl_rt_l_pop, so the parent frame (lFrames.back()) is the caller's in both.
static void paramCopy(uint32_t bytes, bool toCallee) {
  auto &s = state();
  auto &lbuf = s.areaBytes[SCL_RT_AREA_L];
  if (lbuf.empty())
    lbuf.resize(kDefaultLAreaSize, 0);
  auto &mbuf = s.areaBytes[SCL_RT_AREA_M];
  if (mbuf.empty())
    mbuf.resize(kDefaultAreaSize, 0);
  // FC parameter area lives in the CALLER's frame (awlsim: VL from the
  // callee); param_in runs after the callee's l_push, so the parent frame
  // is the caller's
  uint64_t lOff = frameBaseOf(SCL_RT_AREA_V, s) + kParamBase;
  uint64_t mOff = kParamBase;             // marker staging area
  uint64_t dst = toCallee ? lOff : mOff;
  uint64_t src = toCallee ? mOff : lOff;
  std::vector<uint8_t> &dstBuf = toCallee ? lbuf : mbuf;
  std::vector<uint8_t> &srcBuf = toCallee ? mbuf : lbuf;
  if (dst + bytes > dstBuf.size() || src + bytes > srcBuf.size()) {
    fprintf(stderr, "scl_rt: FC parameter area out of bounds "
                    "(need %llu bytes)\n",
            (unsigned long long)(dst + bytes));
    abort();
  }
  memcpy(dstBuf.data() + dst, srcBuf.data() + src, bytes);
}

void scl_rt_param_in(uint32_t bytes) { paramCopy(bytes, true); }

void scl_rt_param_out(uint32_t bytes) { paramCopy(bytes, false); }

// MARK: system function blocks (SFB 0-5)
//
// The instance area is in the DI-register DB at the AR2 base offset
// (awlsim systemblocks; layout: counters CU/CD/R/LD @0.0-0.3, PV @2,
// Q @4.0, CV @6, CUO/COD @8.0/8.1; timers IN @0.0, PT @2, Q @4.0, ET @6).
void scl_rt_sfb(uint32_t n) {
  auto &s = state();
  if (s.vtable && s.vtable->sfb) {
    s.vtable->sfb(n);
    return;
  }
  // awlsim system_sfb_*: s.BIE = 1 on success, 0 on error (e.g. PT < 0)
  auto bie = [&](bool ok) {
    if (ok)
      s.stw |= (uint16_t)(1u << 8);
    else
      s.stw &= (uint16_t)~(1u << 8);
  };
  uint32_t db = s.currentDi;
  uint32_t base = ar2ByteOff();
  auto rd = [&](uint32_t byte, uint8_t w) {
    return scl_rt_read(SCL_RT_AREA_DB, db, base + byte, w);
  };
  auto wr = [&](uint32_t byte, uint8_t w, uint64_t v) {
    scl_rt_write(SCL_RT_AREA_DB, db, base + byte, w, v);
  };
  auto rdb = [&](uint32_t byte, uint8_t bit) {
    return scl_rt_read_bit(SCL_RT_AREA_DB, db, base + byte, bit);
  };
  auto wrb = [&](uint32_t byte, uint8_t bit, bool v) {
    scl_rt_write_bit(SCL_RT_AREA_DB, db, base + byte, bit, v);
  };
  uint64_t key = ((uint64_t)db << 32) | base;
  if (n > 5) {
    bie(true); // unknown SFB number (e.g. CALL SFB -1): no-op, BIE := 1
    return;
  }
  if (n <= 2) { // CTU / CTD / CTUD counters
    int cv = (int16_t)(uint16_t)(rd(6, 2) & 0xFFFF);
    int pv = (int16_t)(uint16_t)(rd(2, 2) & 0xFFFF);
    if (n == 0) { // CTU: CU@0.0 R@0.1, Q = CV >= PV
      bool cu = rdb(0, 0), r = rdb(0, 1), cuo = rdb(8, 0);
      bool edge = cu && !cuo;
      wrb(8, 0, cu);
      if (r)
        cv = 0;
      else if (edge && cv < 32767)
        cv++;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv >= pv);
    } else if (n == 1) { // CTD: CD@0.0 LOAD@0.1, Q = CV <= 0
      bool cd = rdb(0, 0), load = rdb(0, 1), cdo = rdb(8, 0);
      bool cdEdge = cd && !cdo;
      wrb(8, 0, cd);
      if (load)
        cv = pv; // LOAD is level-triggered (awlsim sfb_1): CV := PV
      else if (cdEdge && cv > -32768)
        cv--;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv <= 0);
    } else { // CTUD: CU@0.0 CD@0.1 R@0.2 LOAD@0.3, QU = CV>=PV, QD = CV<=0
      bool cu = rdb(0, 0), cd = rdb(0, 1), r = rdb(0, 2), load = rdb(0, 3);
      bool cuo = rdb(8, 0), cdo = rdb(8, 1);
      bool eu = cu && !cuo, ed = cd && !cdo;
      wrb(8, 0, cu);
      wrb(8, 1, cd);
      if (r)
        cv = 0;
      else if (load)
        cv = pv; // LOAD is level-triggered
      else if (eu && !ed && cv < 32767)
        cv++;
      else if (ed && !eu && cv > -32768)
        cv--;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv >= pv);
      wrb(4, 1, cv <= 0);
    }
    bie(true); // counters always succeed
    return;
  }
  // SFB3 TP / SFB4 TON / SFB5 TOF (awlsim system_sfb_3/4/5). Instance layout:
  // IN@0.0, PT@2 (32-bit TIME, ms), Q@6.0, ET@8 (32-bit), STATE@12 (byte),
  // STIME@14 (32-bit), ATIME@18 (32-bit). The STATE/STIME/ATIME live in the
  // instance DB itself.
  uint64_t now = s.clockMs;
  int64_t pt = (int32_t)(uint32_t)rd(2, 4); // PT: 32-bit TIME in ms
  bie(pt >= 0); // negative PT is an error (awlsim system_sfb_3/4/5)
  bool in = rdb(0, 0);
  uint8_t state = (uint8_t)(rd(12, 1) & 0xFF);
  const uint8_t ST_RUNNING = 1, ST_FINISHED = 2;
  if (pt <= 0) { // invalid/zero PT: reset (awlsim: PT==0 also clears IN)
    if (pt == 0)
      wrb(0, 0, false);
    wrb(6, 0, false);
    wr(8, 4, 0);
    wr(12, 1, 0);
    return;
  }
  if (n == 3) { // TP (awlsim system_sfb_3): Q=1 while running, Q=0 when elapsed
    if (in && !(state & (ST_RUNNING | ST_FINISHED))) {
      wr(14, 4, now); // STIME
      state |= ST_RUNNING;
      wr(12, 1, state);
    }
    if (state & ST_RUNNING) {
      wr(18, 4, now); // ATIME
      uint64_t stime = (uint32_t)rd(14, 4);
      uint64_t et = (now - stime) & 0x7FFFFFFF;
      if ((int64_t)et >= pt) {
        et = (uint64_t)pt;
        state &= ~ST_RUNNING;
        state |= ST_FINISHED;
        wr(12, 1, state);
        wrb(6, 0, false); // Q = 0 when elapsed (awlsim sfb_3)
      } else {
        wrb(6, 0, true); // Q = 1 while running
      }
      wr(8, 4, et);
    }
    if (!in && (state & ST_FINISHED)) {
      state &= ~ST_FINISHED;
      wr(12, 1, state);
      wr(8, 4, 0);
    }
  } else if (n == 4) { // TON (awlsim system_sfb_4): IN starts, !IN shuts down, Q=1 when elapsed
    if (in) {
      if (!(state & (ST_RUNNING | ST_FINISHED))) {
        wr(14, 4, now);
        state |= ST_RUNNING;
        wr(12, 1, state);
      }
    } else {
      state &= ~(ST_FINISHED | ST_RUNNING);
      wr(12, 1, state);
      wr(8, 4, 0);
      wrb(6, 0, false);
    }
    if (state & ST_RUNNING) {
      wr(18, 4, now); // ATIME
      uint64_t stime = (uint32_t)rd(14, 4);
      uint64_t et = (now - stime) & 0x7FFFFFFF;
      if ((int64_t)et >= pt) {
        et = (uint64_t)pt;
        state &= ~ST_RUNNING;
        state |= ST_FINISHED;
        wr(12, 1, state);
        wrb(6, 0, true); // Q = 1 when elapsed
      } else {
        wrb(6, 0, false);
      }
      wr(8, 4, et);
    }
  } else { // TOF (awlsim system_sfb_5): STATE_PREV_IN = 1, STATE_DELAYING = 2
    if (in) {
      state &= ~ST_FINISHED;
      state |= ST_RUNNING; // PREV_IN
      wr(12, 1, state);
      wr(8, 4, 0);
      wrb(6, 0, true); // Q = 1 while IN
    } else if (state & ST_RUNNING) { // negative edge: start the delay
      wr(14, 4, now);
      state |= ST_FINISHED;  // DELAYING
      state &= ~ST_RUNNING;   // ~PREV_IN
      wr(12, 1, state);
    }
    if (state & ST_FINISHED) { // delaying
      wr(18, 4, now); // ATIME
      uint64_t stime = (uint32_t)rd(14, 4);
      uint64_t et = (now - stime) & 0x7FFFFFFF;
      if ((int64_t)et >= pt) {
        et = (uint64_t)pt;
        state &= ~ST_FINISHED;
        wr(12, 1, state);
        wrb(6, 0, false); // Q = 0 when elapsed
      }
      wr(8, 4, et);
    }
  }
}

void scl_rt_sleep(uint32_t ms) {
  auto &s = state();
  if (s.vtable && s.vtable->sleep) {
    s.vtable->sleep(ms);
    return;
  }
  s.clockMs += ms;
}

void scl_rt_counter_op(uint32_t c, uint8_t op, uint64_t value, uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->counter_op) {
    s.vtable->counter_op(c, op, value, in);
    return;
  }
  if (c > 255) {
    fprintf(stderr, "scl_rt_counter_op: invalid counter %u\n", c);
    abort();
  }
  CounterState &cs = s.counters[c];
  switch (op) {
  case SCL_RT_COUNTER_UP: // ZV
    if (in && !cs.prevZV && cs.value < 999)
      ++cs.value;
    cs.prevZV = in != 0;
    break;
  case SCL_RT_COUNTER_DOWN: // ZR
    if (in && !cs.prevZR && cs.value > 0)
      --cs.value;
    cs.prevZR = in != 0;
    break;
  case SCL_RT_COUNTER_SET: // S: BCD-decoded ACC1 on positive edge
    if (in && !cs.prevS)
      cs.value = bcdDecode(value);
    cs.prevS = in != 0;
    break;
  case SCL_RT_COUNTER_RESET: // R
    cs.value = 0;
    break;
  case SCL_RT_COUNTER_FR: // FR: reset edge memory on positive edge
    if (in && !cs.prevFR) {
      cs.prevS = cs.prevZV = cs.prevZR = false;
    }
    cs.prevFR = in != 0;
    break;
    break;
  default:
    fprintf(stderr, "scl_rt_counter_op: invalid op %u\n", op);
    abort();
  }
}

// MARK: builtin timer FBs (TON/TOF/TP)
//
// Builtin instance-DB layout (big-endian, ADR 001): IN @ byte 0 (i8 bool),
// PT @ bytes 4..7 (i32 preset ms), Q @ byte 8 (i8), ET @ bytes 12..15 (i32 ms).
static uint8_t fbReadByte(uint8_t *p, uint64_t off) { return p[off]; }
static void fbWriteByte(uint8_t *p, uint64_t off, uint8_t v) { p[off] = v; }
static int16_t fbReadI16(uint8_t *p, uint64_t off) {
  return (int16_t)((uint16_t)p[off] << 8 | (uint16_t)p[off + 1]);
}
static void fbWriteI16(uint8_t *p, uint64_t off, int16_t v) {
  p[off] = (uint8_t)((uint16_t)v >> 8);
  p[off + 1] = (uint8_t)(uint16_t)v;
}
static int32_t fbReadI32(uint8_t *p, uint64_t off) {
  return (int32_t)((uint32_t)p[off] << 24 | (uint32_t)p[off + 1] << 16 |
                   (uint32_t)p[off + 2] << 8 | (uint32_t)p[off + 3]);
}
static void fbWriteI32(uint8_t *p, uint64_t off, int32_t v) {
  p[off] = (uint8_t)((uint32_t)v >> 24);
  p[off + 1] = (uint8_t)((uint32_t)v >> 16);
  p[off + 2] = (uint8_t)((uint32_t)v >> 8);
  p[off + 3] = (uint8_t)v;
}

void scl_rt_timer_fb(uint8_t mode, uint64_t idbPtr) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_fb) {
    s.vtable->timer_fb(mode, idbPtr);
    return;
  }
  auto *p = reinterpret_cast<uint8_t *>(idbPtr);
  auto &st = s.timerFbs[idbPtr];
  bool in = fbReadByte(p, 0) != 0;
  int32_t pt = fbReadI32(p, 4);
  // negative PT is an error: BIE := 0 (awlsim system_sfb_3/4/5)
  if (pt >= 0)
    s.stw |= (uint16_t)(1u << 8);
  else
    s.stw &= (uint16_t)~(1u << 8);
  if (pt < 0)
    pt = 0;
  uint64_t now = s.clockMs;
  // The end-of-time test (elapsed >= PT) is only applied while the timer was
  // already running at the start of this evaluation, not on the cycle of the
  // edge that starts it (IronPLC behavior recorded in the implementation-
  // defined traces: with PT = 0 the outputs switch one cycle after the edge).
  bool wasRunning = st.running;
  bool q = false;
  int32_t et = 0;
  switch (mode) {
  case 0: { // TON: on-delay. The live PT is used for the comparison and for
    // clamping ET; once Q is TRUE it stays TRUE until IN falls (latched).
    if (in && !st.prevIn) {
      st.running = true;
      st.startMs = now;
    }
    if (!in) {
      st.running = false;
      st.qLatch = false;
    }
    uint64_t elapsed = st.running ? now - st.startMs : 0;
    if (st.running && wasRunning && elapsed >= (uint64_t)pt)
      st.qLatch = true;
    q = st.running && st.qLatch;
    et = st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt) : 0;
    break;
  }
  case 1: // TOF: off-delay (IEC ed.3 fig. 17 c): Q follows IN up at once;
    // on the falling edge of IN timing starts; Q stays TRUE until elapsed
    // reaches PT, then falls and ET holds PT while IN stays FALSE.
    if (in) {
      st.running = false;
      st.finished = false;
      q = true;
      et = 0;
    } else {
      if (st.prevIn) { // falling edge
        st.running = true;
        st.startMs = now;
        st.finished = false;
      }
      uint64_t elapsed = st.running ? now - st.startMs : 0;
      if (st.running && wasRunning && elapsed >= (uint64_t)pt) {
        st.running = false;
        st.finished = true;
      }
      // Q = FALSE at power-up (standard default initial value; the
      // implementation-defined timer_pt_zero trace records IronPLC's
      // deviating TRUE here — see run_st_traces.py's impl-defined handling)
      q = st.running;
      et = st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt)
                      : (st.finished ? pt : 0);
    }
    break;
  default: { // TP: pulse (IEC ed.3 fig. 17 a): on the rising edge of IN a
    // pulse of length PT starts (Q = TRUE, ET counts up regardless of IN);
    // after PT, Q = FALSE and ET holds PT while IN stays TRUE, else 0.
    // A rising edge of IN during a running pulse is ignored (IEC fig. 17 a).
    if (in && !st.prevIn && !st.running) {
      st.running = true;
      st.startMs = now;
      st.finished = false;
    }
    uint64_t elapsed = st.running ? now - st.startMs : 0;
    if (st.running && wasRunning && elapsed >= (uint64_t)pt) {
      st.running = false;
      st.finished = true;
    }
    q = st.running;
    et = st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt)
                    : (st.finished && in ? pt : 0);
    break;
  }
  }
  st.prevIn = in;
  fbWriteByte(p, 8, q ? 1 : 0);
  fbWriteI32(p, 12, et);
}

// Builtin counter FBs CTU/CTD/CTUD (awlsim system_sfb_0/1/2 semantics, but
// with the SCL builtin layout: bool members occupy a full byte, ints are
// big-endian i16). See Runtime.h for the layouts.
void scl_rt_counter_fb(uint8_t mode, uint64_t idbPtr) {
  auto &s = state();
  if (s.vtable && s.vtable->counter_fb) {
    s.vtable->counter_fb(mode, idbPtr);
    return;
  }
  auto *p = reinterpret_cast<uint8_t *>(idbPtr);
  if (mode == 0) { // CTU: CU @0, R @1, PV @2, Q @4, CV @6, CU_OLD @8
    bool cu = fbReadByte(p, 0) != 0, r = fbReadByte(p, 1) != 0;
    bool cuo = fbReadByte(p, 8) != 0;
    fbWriteByte(p, 8, cu ? 1 : 0);
    int cv = fbReadI16(p, 6);
    int pv = fbReadI16(p, 2);
    if (r)
      cv = 0;
    else if (cu && !cuo && cv < 32767)
      cv++;
    fbWriteI16(p, 6, (int16_t)cv);
    fbWriteByte(p, 4, cv >= pv ? 1 : 0);
  } else if (mode == 1) { // CTD: CD @0, LD @1, PV @2, Q @4, CV @6, CD_OLD @8
    bool cd = fbReadByte(p, 0) != 0, ld = fbReadByte(p, 1) != 0;
    bool cdo = fbReadByte(p, 8) != 0;
    fbWriteByte(p, 8, cd ? 1 : 0);
    int cv = fbReadI16(p, 6);
    int pv = fbReadI16(p, 2);
    if (ld)
      cv = pv; // LOAD is level-triggered (awlsim sfb_1)
    else if (cd && !cdo && cv > -32768)
      cv--;
    fbWriteI16(p, 6, (int16_t)cv);
    fbWriteByte(p, 4, cv <= 0 ? 1 : 0);
  } else { // CTUD: CU @0, CD @1, R @2, LD @3, PV @4, QU @6, QD @7, CV @8
    bool cu = fbReadByte(p, 0) != 0, cd = fbReadByte(p, 1) != 0;
    bool r = fbReadByte(p, 2) != 0, ld = fbReadByte(p, 3) != 0;
    bool cuo = fbReadByte(p, 10) != 0, cdo = fbReadByte(p, 11) != 0;
    bool eu = cu && !cuo, ed = cd && !cdo;
    fbWriteByte(p, 10, cu ? 1 : 0);
    fbWriteByte(p, 11, cd ? 1 : 0);
    int cv = fbReadI16(p, 8);
    int pv = fbReadI16(p, 4);
    if (r)
      cv = 0;
    else if (ld)
      cv = pv;
    else if (eu && !ed && cv < 32767)
      cv++;
    else if (ed && !eu && cv > -32768)
      cv--;
    fbWriteI16(p, 8, (int16_t)cv);
    fbWriteByte(p, 6, cv >= pv ? 1 : 0);
    fbWriteByte(p, 7, cv <= 0 ? 1 : 0);
  }
}

// Builtin edge/memory FBs R_TRIG/F_TRIG/SR/RS (IEC 61131-3 ed.3 table 30).
void scl_rt_edge_fb(uint8_t mode, uint64_t idbPtr) {
  auto &s = state();
  if (s.vtable && s.vtable->edge_fb) {
    s.vtable->edge_fb(mode, idbPtr);
    return;
  }
  auto *p = reinterpret_cast<uint8_t *>(idbPtr);
  if (mode <= 1) { // R_TRIG / F_TRIG: CLK @0, Q @1, CLK_OLD @2
    bool clk = fbReadByte(p, 0) != 0;
    bool old = fbReadByte(p, 2) != 0;
    bool q = mode == 0 ? (clk && !old) : (!clk && old);
    fbWriteByte(p, 2, clk ? 1 : 0);
    fbWriteByte(p, 1, q ? 1 : 0);
  } else if (mode == 2) { // SR: S @0, R @1, Q @2 (S dominates)
    bool sv = fbReadByte(p, 0) != 0, rv = fbReadByte(p, 1) != 0;
    bool q = fbReadByte(p, 2) != 0;
    if (sv)
      q = true;
    else if (rv)
      q = false;
    fbWriteByte(p, 2, q ? 1 : 0);
  } else { // RS: R @0, S @1, Q @2 (R dominates)
    bool rv = fbReadByte(p, 0) != 0, sv = fbReadByte(p, 1) != 0;
    bool q = fbReadByte(p, 2) != 0;
    if (rv)
      q = false;
    else if (sv)
      q = true;
    fbWriteByte(p, 2, q ? 1 : 0);
  }
}

void scl_rt_register(const SclRtVTable *vtable) { state().vtable = vtable; }

void scl_rt_sim_set_area_size(uint8_t area, uint64_t size) {
  auto &s = state();
  s.areaSize[area] = size;
  auto &buf = s.areaBytes[area];
  if (!buf.empty())
    buf.resize(size, 0);
}

void scl_rt_sim_set_db_size(uint32_t db, uint64_t size) {
  auto &s = state();
  s.dbSize[db] = size;
  s.dbs[db].resize(size, 0);
}

// Register the highest referenced byte of a DB (frontend pre-scan); used
// for ANY-pointer bounds checks on DBs without a DATA_BLOCK declaration.
void scl_rt_sim_db_ref(uint32_t db, uint64_t bytes) {
  auto &s = state();
  auto &hw = s.dbRefHighWater[db];
  if (bytes > hw)
    hw = bytes;
}

void scl_rt_sim_reset(void) {
  auto &s = state();
  for (auto &b : s.areaBytes)
    b.clear();
  for (auto &sz : s.areaSize)
    sz = 0;
  s.dbs.clear();
  s.dbSize.clear();
  s.dbRefHighWater.clear();
  s.currentDb = 0;
  s.ars[0] = s.ars[1] = 0;
  s.clockMs = 0;
  for (auto &t : s.timers)
    t = TimerState{};
  for (auto &c : s.counters)
    c = CounterState{};
}

uint64_t scl_rt_sim_clock_ms(void) { return state().clockMs; }

// MARK: DB register / address registers / indirect access (ABI v5)

static uint8_t ptrArea(uint64_t p) { return (uint8_t)(p >> 56); }
static uint32_t ptrDb(uint64_t p) { return (uint32_t)((p >> 40) & 0xFFFF); }
static uint32_t ptrByte(uint64_t p) { return (uint32_t)((p >> 16) & 0xFFFF); }
static uint8_t ptrBit(uint64_t p) { return (uint8_t)((p >> 8) & 0xFF); }
static uint64_t packPtr(uint8_t area, uint32_t db, uint32_t byte, uint8_t bit,
                        uint8_t widthBits) {
  return ((uint64_t)area << 56) | ((uint64_t)db << 40) |
         ((uint64_t)byte << 16) | ((uint64_t)bit << 8) | widthBits;
}

void scl_rt_set_current_di(uint32_t db) {
  auto &s = state();
  if (s.vtable && s.vtable->set_current_db) {
    // the vtable has no separate DI entry; hosts that care about DI
    // registers implement set_current_db for both (documented in ADR)
  }
  s.currentDi = db;
}

void scl_rt_set_current_db(uint32_t db) {
  auto &s = state();
  if (s.vtable && s.vtable->set_current_db) {
    s.vtable->set_current_db(db);
    return;
  }
  s.currentDb = db;
}

uint32_t scl_rt_current_di(void) {
  auto &s = state();
  return s.currentDi;
}

uint32_t scl_rt_current_db(void) {
  auto &s = state();
  if (s.vtable && s.vtable->current_db)
    return s.vtable->current_db();
  return s.currentDb;
}

void scl_rt_ar_set(uint8_t ar, uint64_t ptr) {
  auto &s = state();
  if (s.vtable && s.vtable->ar_set) {
    s.vtable->ar_set(ar, ptr);
    return;
  }
  if (ar >= 1 && ar <= 2)
    s.ars[ar - 1] = ptr;
}

// SPEC[AWL] Siemens AWL manual address registers AR1/AR2 -> scl_rt_ar_get/ar_set/ar_add/ar_raw (packed internal form)
uint64_t scl_rt_ar_get(uint8_t ar) {
  auto &s = state();
  if (ar >= 1 && ar <= 2)
    return s.ars[ar - 1];
  return 0;
}

// TARn <memory>: the AR is stored as a raw 32-bit S7 pointer value
// (area byte << 24 | db << 16 | byte << 3 | bit). A raw pointer value
// (area byte 0 in the packed format's top byte) is returned as-is.
uint32_t scl_rt_ar_raw(uint8_t ar) {
  uint64_t p = scl_rt_ar_get(ar);
  if (ptrArea(p) == 0)
    return (uint32_t)(p & 0xFFFFFFFFull);
  uint8_t a = ptrArea(p);
  uint8_t s7 = a == 'P' ? 0x80 : a == 'I' ? 0x81 : a == 'Q' ? 0x82
             : a == 'M' ? 0x83 : a == 'D' ? 0x84 : a == 'L' ? 0x86
             : a == 'V' ? 0x87 : 0;
  uint32_t db = ptrDb(p) & 0xFF;
  uint32_t byte = ptrByte(p) & (s7 == 0x87 ? 0x1FFFFF : 0x1FFF);
  return ((uint32_t)s7 << 24) | (db << 16) | (byte << 3) | ptrBit(p);
}

// [ARn, P#x.y] addressing: returns ARn + offset WITHOUT modifying ARn
// (the offset form is a temporary, per awlsim's operand evaluation).
uint64_t scl_rt_ar_add(uint8_t ar, uint32_t offBits) {
  auto &s = state();
  if (s.vtable && s.vtable->ar_add)
    return s.vtable->ar_add(ar, offBits);
  uint64_t p = scl_rt_ar_get(ar);
  // raw P#x.y pointer (area byte 0, value = byte*8+bit): plain add
  if (ptrArea(p) == 0)
    return p + offBits;
  uint32_t bit = ptrBit(p) + offBits;
  uint32_t byte = ptrByte(p) + bit / 8;
  bit %= 8;
  return packPtr(ptrArea(p), ptrDb(p), byte, (uint8_t)bit, (uint8_t)(p & 0xFF));
}

// A raw pointer (area byte 0) addresses the marker area: value = byte*8+bit.
// An S7-format 32-bit pointer (area code 0x80-0x86 at bits 24-31, e.g. from
// a P#DBX 0.0 literal) is decoded to the internal packed format.
static uint64_t rawToPacked(uint64_t p) {
  if (ptrArea(p) != 0)
    return p;
  uint8_t s7 = (uint8_t)((p >> 24) & 0xFF);
  if (p < 0x100000000ull && s7 >= 0x80 && s7 <= 0x87) {
    static const uint8_t s7area[8] = {'P', 'I', 'Q', 'M', 'D', 'D', 'L', 'V'}; // 0x87 = VL (parent frame)
    uint8_t area = s7area[s7 - 0x80];
    // 0x85 = DI area: the DB number comes from the DI register (db=0xFFFF)
    uint32_t db = s7 == 0x85 ? 0xFFFF : (uint32_t)((p >> 16) & 0xFF);
    // VL (0x87) has no DB field, so its byte offset may use bits 3..23
    // (the FC parameter area starts at kLocalBase = 8192 > 8191)
    uint32_t byte = s7 == 0x87 ? (uint32_t)((p >> 3) & 0x1FFFFF)
                               : (uint32_t)((p >> 3) & 0x1FFF);
    uint8_t bit = (uint8_t)(p & 7);
    return packPtr(area, db, byte, bit, 0);
  }
  return packPtr(SCL_RT_AREA_M, 0, (uint32_t)(p / 8), (uint8_t)(p % 8), 0);
}

// SPEC[AWL] Siemens AWL manual indirect addressing -> scl_rt_read_indirect/write_indirect (raw S7 <-> packed decode)
int64_t scl_rt_read_indirect(uint64_t ptr, uint8_t widthBytes) {
  auto &s = state();
  if (s.vtable && s.vtable->read_indirect)
    return s.vtable->read_indirect(ptr, widthBytes);
  uint64_t p = rawToPacked(ptr);
  return scl_rt_read(ptrArea(p), ptrDb(p), ptrByte(p), widthBytes);
}

void scl_rt_write_indirect(uint64_t ptr, int64_t value, uint8_t widthBytes) {
  auto &s = state();
  if (s.vtable && s.vtable->write_indirect) {
    s.vtable->write_indirect(ptr, value, widthBytes);
    return;
  }
  uint64_t p = rawToPacked(ptr);
  scl_rt_write(ptrArea(p), ptrDb(p), ptrByte(p), widthBytes, (uint64_t)value);
}

bool scl_rt_read_bit_indirect(uint64_t ptr) {
  auto &s = state();
  if (s.vtable && s.vtable->read_bit_indirect)
    return s.vtable->read_bit_indirect(ptr);
  uint64_t p = rawToPacked(ptr);
  return scl_rt_read_bit(ptrArea(p), ptrDb(p), ptrByte(p), ptrBit(p));
}

void scl_rt_write_bit_indirect(uint64_t ptr, bool value) {
  auto &s = state();
  if (s.vtable && s.vtable->write_bit_indirect) {
    s.vtable->write_bit_indirect(ptr, value);
    return;
  }
  uint64_t p = rawToPacked(ptr);
  scl_rt_write_bit(ptrArea(p), ptrDb(p), ptrByte(p), ptrBit(p), value);
}

// MARK: system functions (SFCs). The frontend stages the parameters in the
// marker area at kParamBase (SFC 21: BVAL/BLK as i64 S7 pointers @+0/+8,
// RET_VAL i16 @+16; SFC 47: WT i16 @+0; SFC 64: RET_VAL i32 @+0).
// SPEC[AWL] Siemens AWL manual system functions SFC/SFB -> scl_rt_sfc/scl_rt_sfb (runtime-staged params in the marker area)
void scl_rt_sfc(uint32_t n) {
  auto &s = state();
  if (s.vtable && s.vtable->sfc) {
    s.vtable->sfc(n);
    return;
  }
  // awlsim system_sfc_*: s.BIE = 1 on success, 0 on error paths
  auto bie = [&](bool ok) {
    if (ok)
      s.stw |= (uint16_t)(1u << 8);
    else
      s.stw &= (uint16_t)~(1u << 8);
  };
  if (n == 64) { // TIME_TCK: 31-bit millisecond timestamp
    scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + 0, 4,
                 s.clockMs & 0x7FFFFFFF);
    bie(true);
    return;
  }
  if (n == 47) { // WAIT: WT in microseconds, max 32767 (awlsim sfc_47)
    int wt = (int16_t)(uint16_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + 0, 2);
    if (wt > 0)
      s.clockMs += ((uint64_t)std::min(wt, 32767) + 999) / 1000;
    bie(true);
    return;
  }
  if (n == 46 || n == -3) {
    // awlsim system_sfc_46 / system_sfc_m3: STOP/SHUTDOWN abort the scan
    s.stopRequested = true;
    bie(true);
    return;
  }
  if (n != 21) {
    bie(true); // other SFCs: host-provided stub, BIE := 1
    return;
  }
  // SFC 21 FILL (awlsim system_sfc_21): fill the BLK ANY block with BVAL.
  // The frontend stages the full 10-byte ANY headers inline in the marker
  // area: BVAL @ kParamBase+0, BLK @ kParamBase+10, RET_VAL (i16) @ +20.
  constexpr uint32_t BVAL = 0, BLK = 10, RETV = 20;
  auto fail = [&](uint16_t code) {
    scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + RETV, 2, code);
    bie(false); // error path: BIE := 0 (awlsim system_sfc_21)
  };
  auto anyByte = [&](uint32_t base, uint32_t i) -> uint8_t {
    return (uint8_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + base + i, 1);
  };
  auto anyWord = [&](uint32_t base, uint32_t i) -> uint32_t {
    return ((uint32_t)anyByte(base, i) << 8) | anyByte(base, i + 1);
  };
  if (anyByte(BVAL, 0) != 0x10)
    return fail(0x8124); // E_RAREA p1: invalid BVAL magic
  if (anyByte(BLK, 0) != 0x10)
    return fail(0x8325); // E_WAREA p3: invalid BLK magic
  // ANY type code -> width in bits (awlsim ANYPointerConst + typeWidths)
  auto typeWidth = [](uint32_t code) -> int {
    switch (code) {
    case 0x01: return 1;                                   // BOOL
    case 0x02: case 0x03: return 8;                        // BYTE CHAR
    case 0x04: case 0x05: case 0x09: case 0x0C: return 16; // WORD INT DATE S5T
    case 0x06: case 0x07: case 0x08: case 0x0A:
    case 0x0B: return 32; // DWORD DINT REAL TOD TIME
    case 0x0E: return 64;                                  // DT
    case 0x17: case 0x18: case 0x19: case 0x1A:
    case 0x1C: case 0x1D: return 16; // BLOCK_* COUNTER TIMER
    default: return -1;
    }
  };
  int bw = typeWidth(anyByte(BVAL, 1));
  if (bw <= 0)
    return fail(0x8124);
  int lw = typeWidth(anyByte(BLK, 1));
  if (lw <= 0)
    return fail(0x8325);
  uint64_t blen = (uint64_t)bw * anyWord(BVAL, 2);
  uint64_t llen = (uint64_t)lw * anyWord(BLK, 2);
  if (blen % 8)
    return fail(0x8122); // E_RLEN p1
  if (llen % 8)
    return fail(0x8323); // E_WLEN p3
  blen /= 8;
  llen /= 8;
  // decode the ANY pointer dwords (S7 format: area<<24|db<<16|byte<<3|bit)
  auto decode = [&](uint32_t base, uint32_t &area, uint32_t &byte,
                    uint32_t &bit) {
    uint64_t ptr = ((uint64_t)anyByte(base, 6) << 24) |
                   ((uint64_t)anyByte(base, 7) << 16) |
                   ((uint64_t)anyByte(base, 8) << 8) | anyByte(base, 9);
    area = (uint32_t)(ptr >> 24) & 0xFF;
    byte = (uint32_t)(ptr >> 3) & 0x1FFF;
    bit = (uint32_t)ptr & 7;
  };
  uint32_t bArea, bByte, bBit, lArea, lByte, lBit;
  decode(BVAL, bArea, bByte, bBit);
  decode(BLK, lArea, lByte, lBit);
  // resolve the effective runtime area; DB/DI areas take the DB number from
  // the ANY header (word @4); DB 0 / DB >= 0x100 are never loaded (awlsim
  // dbsAlloc = 0x100)
  auto resolve = [&](uint32_t area, uint32_t base, uint8_t &outArea,
                     uint32_t &outDb) -> int {
    if (area == 0x84 || area == 0x85) {
      uint32_t dbNr = anyWord(base, 4);
      // awlsim system_sfc_21: E_DBNOTEXIST if DB 0 or the DB is not loaded
      // (DB numbers are not limited to 0xFF; tests use e.g. DB 900)
      if (dbNr == 0 || !s.dbs.count(dbNr))
        return 1;
      outArea = SCL_RT_AREA_DB;
      outDb = dbNr;
      return 0;
    }
    static const uint8_t s7area[8] = {'P', 'I', 'Q', 'M', 'D', 'D', 'L', 'V'}; // 0x87 = VL (parent frame)
    if (area >= 0x80 && area <= 0x86) {
      outArea = s7area[area - 0x80];
      outDb = 0;
      return 0;
    }
    return 2; // invalid area
  };
  uint8_t bRta = 0, lRta = 0;
  uint32_t bRtd = 0, lRtd = 0;
  int br = resolve(bArea, BVAL, bRta, bRtd);
  if (br == 1)
    return fail(0x813A); // E_DBNOTEXIST p1
  int lr = resolve(lArea, BLK, lRta, lRtd);
  if (lr == 1)
    return fail(0x833A); // E_DBNOTEXIST p3
  if (br != 0 || bBit != 0)
    return fail(0x8128); // E_RALIGN p1
  if (lr != 0 || lBit != 0)
    return fail(0x8329); // E_WALIGN p3
  // bounds check for the copy loop (OOB -> fetch/store error, awlsim raises)
  auto areaBound = [&](uint8_t area, uint32_t db, uint32_t byte,
                       uint32_t w) -> bool {
    uint32_t off = byte;
    if (area == SCL_RT_AREA_DB) {
      auto it = s.dbSize.find(db);
      uint64_t size = it != s.dbSize.end() ? it->second : 0;
      if (s.dbs.count(db) && s.dbs[db].size() > size)
        size = s.dbs[db].size();
      auto hw = s.dbRefHighWater.find(db);
      if (hw != s.dbRefHighWater.end() && hw->second > size)
        size = hw->second; // referenced extent of an undeclared DB
      return (uint64_t)off + w <= size;
    }
    if (area == SCL_RT_AREA_L || area == SCL_RT_AREA_V) {
      off += (uint32_t)frameBaseOf(area, s);
      area = SCL_RT_AREA_L; // V shares the L-stack buffer
    }
    auto &buf = s.areaBytes[area];
    uint64_t size = buf.empty() ? (s.areaSize[area] ? s.areaSize[area]
                                                   : kDefaultAreaSize)
                                : (uint64_t)buf.size();
    return (uint64_t)off + w <= size;
  };
  // copy loop: DWORD/WORD/BYTE chunks; BVAL wraps around to its begin when
  // exhausted (awlsim system_sfc_21), so the fill pattern repeats
  uint64_t bOff = 0;
  for (uint64_t off = 0; off < llen;) {
    uint32_t w = 1;
    if (llen - off >= 4 && blen - bOff >= 4)
      w = 4;
    else if (llen - off >= 2 && blen - bOff >= 2)
      w = 2;
    if (!areaBound(bRta, bRtd, bByte + bOff, w))
      return fail(0x8124); // fetch error -> E_RAREA p1
    if (!areaBound(lRta, lRtd, lByte + off, w))
      return fail(0x8325); // store error -> E_WAREA p3
    uint64_t v = scl_rt_read(bRta, bRtd, bByte + bOff, (uint8_t)w);
    scl_rt_write(lRta, lRtd, lByte + off, (uint8_t)w, v);
    bOff += w;
    if (bOff >= blen)
      bOff = 0;
    off += w;
  }
  scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + RETV, 2, 0); // RET_VAL = 0
  bie(true); // success: BIE := 1
}

// MARK: IEC library FCs called by name (awlsim library/iec). The frontend
// stages the parameters in the marker area at kParamBase (AwlGen
// emitRtCall): LEN 21: S ANY @+0, RET_VAL i16 @+20; EQ_DT 9 / GE_DT 12:
// DT1 ANY @+0, DT2 ANY @+10, RET_VAL bool @+20.0; DELETE 4: IN ANY @+0,
// L i16 @+10, P i16 @+12, RET_VAL (OUT STRING) ANY @+20.
void scl_rt_iec(uint32_t n) {
  auto &s = state();
  if (s.vtable && s.vtable->iec) {
    s.vtable->iec(n);
    return;
  }
  // the IEC library FCs are AWL functions that end with SAVE (BIE := RLO):
  // success sets BIE=1, an error path sets BIE=0 (awlsim library/iec/fc*)
  auto bie = [&](bool ok) {
    if (ok)
      s.stw |= (uint16_t)(1u << 8);
    else
      s.stw &= (uint16_t)~(1u << 8);
  };
  auto anyByte = [&](uint32_t base, uint32_t i) -> uint8_t {
    return (uint8_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + base + i, 1);
  };
  auto anyWord = [&](uint32_t base, uint32_t i) -> uint32_t {
    return ((uint32_t)anyByte(base, i) << 8) | anyByte(base, i + 1);
  };
  // decode a staged 10-byte ANY header into a runtime (area, db, byte, bit)
  auto anyResolve = [&](uint32_t base, uint8_t &area, uint32_t &db,
                        uint32_t &byte, uint8_t &bit) -> bool {
    if (anyByte(base, 0) != 0x10)
      return false;
    uint64_t ptr = ((uint64_t)anyByte(base, 6) << 24) |
                   ((uint64_t)anyByte(base, 7) << 16) |
                   ((uint64_t)anyByte(base, 8) << 8) | anyByte(base, 9);
    uint32_t s7 = (uint32_t)(ptr >> 24) & 0xFF;
    byte = (uint32_t)(ptr >> 3) & 0x1FFF;
    bit = (uint8_t)(ptr & 7);
    static const uint8_t s7area[8] = {'P', 'I', 'Q', 'M', 'D', 'D', 'L', 'V'}; // 0x87 = VL (parent frame)
    if (s7 == 0x84 || s7 == 0x85) { // DB / DI: DB number from the header
      uint32_t dbNr = anyWord(base, 4);
      if (dbNr == 0 || dbNr >= 0x100)
        return false;
      area = SCL_RT_AREA_DB;
      db = dbNr;
      return true;
    }
    if (s7 >= 0x80 && s7 <= 0x87) {
      area = s7area[s7 - 0x80];
      db = 0;
      return true;
    }
    return false;
  };
  auto readBytes = [&](uint8_t area, uint32_t db, uint32_t byte, uint32_t len,
                       uint8_t *out) {
    for (uint32_t i = 0; i < len; ++i)
      out[i] = (uint8_t)scl_rt_read(area, db, byte + i, 1);
  };
  if (n == 21) { // LEN: RET_VAL := actual length of S (string byte 1)
    uint8_t a, bit;
    uint32_t d, b;
    if (!anyResolve(0, a, d, b, bit)) {
      bie(false);
      return;
    }
    int len = (int)(uint8_t)scl_rt_read(a, d, b + 1, 1);
    scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + 20, 2,
                 (uint64_t)(uint16_t)len);
    bie(true);
    return;
  }
  if (n == 9 || n == 12) { // EQ_DT / GE_DT: compare two DATE_AND_TIMEs
    uint8_t a1, a2, bit1, bit2;
    uint32_t d1, d2, b1, b2;
    if (!anyResolve(0, a1, d1, b1, bit1) ||
        !anyResolve(10, a2, d2, b2, bit2)) {
      bie(false);
      return;
    }
    uint8_t t1[8], t2[8];
    readBytes(a1, d1, b1, 8, t1);
    readBytes(a2, d2, b2, 8, t2);
    bool res;
    if (n == 9) {
      res = memcmp(t1, t2, 8) == 0;
    } else {
      // awlsim fc12: invalid year BCD -> FAIL (RET_VAL untouched)
      auto bcdOk = [](uint8_t b) {
        return ((b >> 4) & 0xF) <= 9 && (b & 0xF) <= 9;
      };
      if (!bcdOk(t1[0]) || !bcdOk(t2[0])) {
        bie(false); // BCD failure: BIE := 0, RET_VAL untouched
        return;
      }
      // year correction (1990-1999 -> +1900, 2000-2089 -> +2000), then
      // compare the remaining 7 bytes as BCD fields
      auto yearKey = [](uint8_t bcd) {
        int y = ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
        return y >= 90 ? 1900 + y : 2000 + y;
      };
      int y1 = yearKey(t1[0]), y2 = yearKey(t2[0]);
      res = y1 > y2 || (y1 == y2 && memcmp(t1 + 1, t2 + 1, 7) >= 0);
    }
    scl_rt_write_bit(SCL_RT_AREA_M, 0, kParamBase + 20, 0, res);
    bie(true);
    return;
  }
  if (n == 4) { // DELETE: RET_VAL := IN with L chars deleted from position P
    uint8_t ai, ar, biti, bitr;
    uint32_t di, dr, bi, br;
    if (!anyResolve(0, ai, di, bi, biti) ||
        !anyResolve(20, ar, dr, br, bitr)) {
      bie(false);
      return;
    }
    int L = (int16_t)(uint16_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + 10, 2);
    int P = (int16_t)(uint16_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + 12, 2);
    int inLen = (int)(uint8_t)scl_rt_read(ai, di, bi + 1, 1);
    int outMax = (int)(uint8_t)scl_rt_read(ar, dr, br, 1);
    if (L < 0 || P < 0) { // awlsim ERR: RET_VAL actual length := 0, BIE := 0
      scl_rt_write(ar, dr, br + 1, 1, 0);
      bie(false);
      return;
    }
    int maxCopy = inLen < outMax ? inLen : outMax;
    // byte offsets within the IN string (chars start at byte 2)
    int delStart, delEnd, inEnd = inLen + 2;
    if (L == 0 || P == 0 || P > maxCopy) {
      delStart = delEnd = inEnd; // full copy, nothing deleted
    } else {
      delStart = P + 1; // P is 1-based
      delEnd = delStart + L;
      if (delEnd > inEnd)
        delEnd = inEnd;
    }
    int limit = maxCopy + 2;
    if (delStart > limit)
      delStart = delEnd = limit;
    int limit2 = limit + (delEnd - delStart);
    if (inEnd > limit2)
      inEnd = limit2;
    // copy [2, delStart) and [delEnd, inEnd) to RET_VAL from byte 2 on
    uint32_t w = 2;
    for (int off = 2; off < delStart; ++off, ++w)
      scl_rt_write(ar, dr, br + w, 1,
                   (uint64_t)(uint8_t)scl_rt_read(ai, di, bi + off, 1));
    for (int off = delEnd; off < inEnd; ++off, ++w)
      scl_rt_write(ar, dr, br + w, 1,
                   (uint64_t)(uint8_t)scl_rt_read(ai, di, bi + off, 1));
    scl_rt_write(ar, dr, br + 1, 1, w - 2); // RET_VAL actual length
    bie(true);
    return;
  }
}

// MARK: dynamic T/C access (TIMER/COUNTER-typed block variables store the
// timer/counter NUMBER; the frontend packs area 'T'/'C' with the runtime
// number as the byte field via scl_rt_pack_area)
uint64_t scl_rt_pack_area(uint8_t area, uint32_t byte, uint8_t bit) {
  auto &s = state();
  if (s.vtable && s.vtable->pack_area)
    return s.vtable->pack_area(area, byte, bit);
  return packPtr(area, 0, byte, bit, 0);
}

void scl_rt_counter_op_ptr(uint64_t ptr, uint8_t op, uint64_t value,
                           uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->counter_op_ptr) {
    s.vtable->counter_op_ptr(ptr, op, value, in);
    return;
  }
  scl_rt_counter_op(ptrByte(rawToPacked(ptr)), op, value, in);
}

void scl_rt_timer_start_ptr(uint64_t ptr, uint8_t mode, uint64_t durS5,
                            uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_start_ptr) {
    s.vtable->timer_start_ptr(ptr, mode, durS5, in);
    return;
  }
  scl_rt_timer_start(ptrByte(rawToPacked(ptr)), mode, durS5, in);
}

void scl_rt_timer_reset_ptr(uint64_t ptr) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_reset_ptr) {
    s.vtable->timer_reset_ptr(ptr);
    return;
  }
  scl_rt_timer_reset(ptrByte(rawToPacked(ptr)));
}

void scl_rt_fr_ptr(uint64_t ptr, uint8_t in) {
  auto &s = state();
  if (s.vtable && s.vtable->fr_ptr) {
    s.vtable->fr_ptr(ptr, in);
    return;
  }
  uint64_t p = rawToPacked(ptr);
  uint32_t n = ptrByte(p);
  if (ptrArea(p) == SCL_RT_AREA_T)
    scl_rt_timer_fr(n, in);
  else
    scl_rt_counter_op(n, SCL_RT_COUNTER_FR, 0, in);
}

// MARK: status word (awlsim statusword; bits NER=0 VKE/RLO=1 STA=2 OR=3
// OS=4 OV=5 A0/CC0=6 A1/CC1=7 BIE/BR=8)
// ACCU registers (awlsim cpu.accu1..4): persistent runtime accumulator
// state, so the frontend can reload the parse-time acc stack at labels.
uint32_t scl_rt_accu_get(uint8_t index) {
  auto &s = state();
  if (s.vtable && s.vtable->accu_get)
    return s.vtable->accu_get(index);
  if (index < 1 || index > 4)
    return 0;
  return s.accus[index - 1];
}
void scl_rt_accu_set(uint8_t index, uint32_t value) {
  auto &s = state();
  if (s.vtable && s.vtable->accu_set) {
    s.vtable->accu_set(index, value);
    return;
  }
  if (index < 1 || index > 4)
    return;
  s.accus[index - 1] = value;
}
uint32_t scl_rt_stw_get(void) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_get)
    return s.vtable->stw_get();
  return s.stw;
}
void scl_rt_stw_set(uint32_t word) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_set) {
    s.vtable->stw_set(word);
    return;
  }
  s.stw = (uint16_t)word;
}
void scl_rt_stw_reset(void) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_reset) {
    s.vtable->stw_reset();
    return;
  }
  s.stw = 0;
}
void scl_rt_stw_set_bit(uint8_t bit, uint8_t value) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_set_bit) {
    s.vtable->stw_set_bit(bit, value);
    return;
  }
  if (value)
    s.stw |= (uint16_t)(1u << bit);
  else
    s.stw &= (uint16_t)~(1u << bit);
}
static void stw_cmp_set(SimState &s, uint8_t pred, bool gt, bool lt, bool eq) {
  // awlsim insn_eq_i etc.: A1=1 when lhs>rhs, A0=1 when lhs<rhs, and
  // VKE = whether the comparison predicate holds (VKE mirrors the RLO)
  bool res;
  switch (pred) {
  case 0: res = eq; break;   // ==
  case 1: res = !eq; break;  // <>
  case 2: res = lt; break;   // <
  case 3: res = lt || eq; break; // <=
  case 4: res = gt; break;   // >
  default: res = gt || eq; break; // >=
  }
  s.stw = (uint16_t)((s.stw & ~((1u << 6) | (1u << 7) | (1u << 1))) |
                     (gt ? (1u << 7) : 0u) | (lt ? (1u << 6) : 0u) |
                     (res ? (1u << 1) : 0u));
}
void scl_rt_stw_cmp_i(uint8_t pred, int64_t lhs, int64_t rhs) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_cmp_i) {
    s.vtable->stw_cmp_i(pred, lhs, rhs);
    return;
  }
  stw_cmp_set(s, pred, lhs > rhs, lhs < rhs, lhs == rhs);
}
void scl_rt_stw_cmp_f(uint8_t pred, double lhs, double rhs) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_cmp_f) {
    s.vtable->stw_cmp_f(pred, lhs, rhs);
    return;
  }
  bool unord = std::isnan(lhs) || std::isnan(rhs);
  stw_cmp_set(s, pred, lhs > rhs, lhs < rhs, lhs == rhs);
  if (unord) {
    // awlsim: unordered (NaN) comparison -> A0=A1=OV=OS=1
    s.stw |= (uint16_t)((1u << 6) | (1u << 7) | (1u << 5) | (1u << 4));
  }
}
// v9: arithmetic STW recording (awlsim statusword.setForInteger /
// setForFloatingPointDWord). A0/A1 = result sign, OV cleared unless the
// value is inf/nan/denorm; OS only ever latches (until stw_reset).
static void stw_set_a0a1(SimState &s, bool a1, bool a0) {
  s.stw = (uint16_t)((s.stw & ~((1u << 6) | (1u << 7))) |
                     (a0 ? (1u << 6) : 0u) | (a1 ? (1u << 7) : 0u));
}
void scl_rt_stw_sign_i(int64_t value, int64_t lhs, int64_t rhs, uint8_t op,
                       int32_t range) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_sign_i) {
    s.vtable->stw_sign_i(value, lhs, rhs, op, range);
    return;
  }
  if (value == 0)
    stw_set_a0a1(s, false, false);
  else if (value < 0)
    stw_set_a0a1(s, false, true);
  else
    stw_set_a0a1(s, true, false);
  // overflow: the untruncated result left the accumulator range
  int64_t wide = op == 1 ? lhs - rhs : op == 2 ? lhs * rhs : lhs + rhs;
  if (wide > (int64_t)range || wide < -(int64_t)range - 1)
    s.stw |= (1u << 4) | (1u << 5); // OV sets, OS latches
  else
    s.stw &= (uint16_t)~(1u << 5); // OV=0
}
void scl_rt_stw_sign_f(double value) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_sign_f) {
    s.vtable->stw_sign_f(value);
    return;
  }
  // awlsim setForFloatingPoint first checks isDenormalPyFloat on the DOUBLE
  // result (0 < |v| < 2^-126, i.e. denormal as float32): A0=A1=0, OV=OS=1
  double av = std::fabs(value);
  if (av > 0.0 && av < 1.17549435e-38) {
    stw_set_a0a1(s, false, false);
    s.stw |= (1u << 4) | (1u << 5);
    return;
  }
  // classify the float32 bit pattern (awlsim setForFloatingPointDWord)
  float f = (float)value;
  uint32_t dword;
  std::memcpy(&dword, &f, 4);
  uint32_t noSign = dword & 0x7FFFFFFFu;
  if (noSign == 0) { // zero
    stw_set_a0a1(s, false, false);
    s.stw &= (uint16_t)~(1u << 5);
  } else if (noSign < 0x00800000u) { // denorm
    stw_set_a0a1(s, false, false);
    s.stw |= (1u << 4) | (1u << 5);
  } else if (noSign >= 0x7F800000u) {
    if (noSign == 0x7F800000u) { // inf
      stw_set_a0a1(s, !(dword & 0x80000000u), (dword & 0x80000000u) != 0);
    } else { // nan: unordered
      stw_set_a0a1(s, true, true);
    }
    s.stw |= (1u << 4) | (1u << 5);
  } else if (dword & 0x80000000u) { // norm neg
    stw_set_a0a1(s, false, true);
    s.stw &= (uint16_t)~(1u << 5);
  } else { // norm pos
    stw_set_a0a1(s, true, false);
    s.stw &= (uint16_t)~(1u << 5);
  }
}
// awlsim insn_pl_r/mi_r/mu_r/di_r: real arith is computed in double and
// rounded to float32 on store (setPyFloat). *R special-cases inf operands
// (inf * x = x with inf's sign XORed in; inf * 0 = NaN) and sets OV=OS=1;
// /R maps inf divisor to 0.0 and zero divisor to +-inf by the dividend's
// sign. The STW is set here (setForFloatingPoint + the *R OV/OS override).
// op: 0=+R 1=-R 2=*R 3=/R; a = ACCU2, b = ACCU1 (top).
double scl_rt_arith_f(uint8_t op, double a, double b) {
  auto &s = state();
  if (s.vtable && s.vtable->arith_f)
    return s.vtable->arith_f(op, a, b);
  float fa = (float)a, fb = (float)b;
  uint32_t ab, bb;
  std::memcpy(&ab, &fa, 4);
  std::memcpy(&bb, &fb, 4);
  auto isInf = [](uint32_t d) { return (d & 0x7FFFFFFFu) == 0x7F800000u; };
  auto isZero = [](uint32_t d) { return (d & 0x7FFFFFFFu) == 0; };
  double res;
  bool mulInf = false;
  // awlsim (Apple Silicon/ARM) propagates a negative all-ones NaN from real
  // arith: every NaN result observed in the awlsim testsuite is 0xFFFFFFFF
  auto negNan = []() {
    uint64_t bits = 0xFFFFFFFFFFFFFFFFULL;
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
  };
  switch (op) {
  case 0: res = a + b; break; // +R
  case 1: res = a - b; break; // -R
  case 2: // *R
    if (isInf(ab) || isInf(bb)) {
      mulInf = true;
      if (isZero(ab) || isZero(bb)) {
        res = negNan(); // inf * 0 = NaN (awlsim nNaNDWord = 0xFFFFFFFF)
      } else {
        uint32_t r = ab ^ (bb & 0x80000000u); // accu2 bits ^ sign(accu1)
        float f;
        std::memcpy(&f, &r, 4);
        res = f;
      }
    } else {
      res = a * b;
    }
    break;
  case 3: // /R
    if (isInf(bb))
      res = 0.0; // inf divisor
    else if (b != 0.0)
      res = a / b;
    else // zero divisor: +-inf by the dividend's sign
      res = a >= 0.0 ? __builtin_huge_val() : -__builtin_huge_val();
    break;
  default: res = 0.0; break;
  }
  if (std::isnan(res))
    res = negNan();
  scl_rt_stw_sign_f(res); // classifies the f32-rounded result
  if (mulInf)
    s.stw |= (1u << 4) | (1u << 5); // *R with an inf operand: OV=OS=1
  return res;
}
// awlsim unary real math (insn_sin/cos/tan/asin/acos/atan/sqrt/exp/ln/sqr):
// computed in double, rounded to f32 on store, STW classified from the
// result. NaN input -> ACCU := pNaNDWord (0x7FFFFFFF); inf input -> unchanged;
// domain errors -> pNaNDWord; sin/cos/tan snap results ~= -1/0/1 exactly;
// tan(+-pi/2) -> +-inf; exp(x>88) -> +inf. op: 0=sin 1=cos 2=tan 3=asin
// 4=acos 5=atan 6=sqrt 7=exp 8=ln 9=sqr
double scl_rt_realmath_f(uint8_t op, double a) {
  auto &s = state();
  if (s.vtable && s.vtable->realmath_f)
    return s.vtable->realmath_f(op, a);
  float fa = (float)a;
  uint32_t d;
  std::memcpy(&d, &fa, 4);
  bool isNan = (d & 0x7FFFFFFFu) > 0x7F800000u;
  bool isInf = (d & 0x7FFFFFFFu) == 0x7F800000u;
  constexpr double kEps = 0.0000001; // awlsim floatConst.epsilonFloat
  auto eq = [&](double x, double y) { return std::fabs(x - y) < kEps; };
  auto pNaN = []() {
    uint32_t b = 0x7FFFFFFFu; // awlsim pNaNDWord
    float f;
    std::memcpy(&f, &b, 4);
    return (double)f;
  };
  auto snap = [&](double x) {
    if (eq(x, -1.0)) return -1.0;
    if (eq(x, 0.0)) return 0.0;
    if (eq(x, 1.0)) return 1.0;
    return x;
  };
  double res = a;
  switch (op) {
  case 0: // sin (negZero passes through unchanged, awlsim insn_sin)
    if (isNan) res = pNaN();
    else if (isInf || d == 0x80000000u) res = a;
    else res = snap(std::sin(a));
    break;
  case 1: // cos (negZero is NOT special-cased, awlsim insn_cos)
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else res = snap(std::cos(a));
    break;
  case 2: // tan
    if (isNan) res = pNaN();
    else if (isInf || d == 0x80000000u) res = a;
    else if (eq(a, 3.14159265358979 / 2)) res = __builtin_huge_val();
    else if (eq(a, -3.14159265358979 / 2)) res = -__builtin_huge_val();
    else res = snap(std::tan(a));
    break;
  case 3: // asin
  case 4: // acos
    if (isInf) res = a;
    else if (isNan || a > 1.0 || a < -1.0) res = pNaN();
    else res = op == 3 ? std::asin(a) : std::acos(a);
    break;
  case 5: // atan
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else res = std::atan(a);
    break;
  case 6: // sqrt
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else if (a < 0.0) res = pNaN();
    else res = std::sqrt(a);
    break;
  case 7: // exp
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else if ((d & 0x80000000u) == 0 && d > 0x42B00000u) res = __builtin_huge_val();
    else res = std::exp(a);
    break;
  case 8: // ln
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else if (a <= 0.0) res = pNaN();
    else res = std::log(a);
    break;
  case 9: // sqr
    if (isNan) res = pNaN();
    else if (isInf) res = a;
    else res = a * a;
    break;
  default: res = a; break;
  }
  scl_rt_stw_sign_f(res); // setForFloatingPoint on the (unrounded) result
  return res;
}
void scl_rt_stw_div_i(int64_t divisor, int64_t result, int32_t range) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_div_i) {
    s.vtable->stw_div_i(divisor, result, range);
    return;
  }
  if (divisor == 0) { // division by zero: unordered + overflow
    stw_set_a0a1(s, true, true);
    s.stw |= (1u << 4) | (1u << 5);
    return;
  }
  // awlsim insn_di_i/insn_di_d: A0/A1/OV from the quotient, OV=OS=1 when the
  // quotient leaves +-range (0x7FFF for /I, 0x7FFFFFFF for /D)
  scl_rt_stw_sign_i(result, result, 0, 0, range);
}
void scl_rt_stw_div_f(double divisor, double result) {
  auto &s = state();
  if (s.vtable && s.vtable->stw_div_f) {
    s.vtable->stw_div_f(divisor, result);
    return;
  }
  (void)divisor;
  scl_rt_stw_sign_f(result); // 1.0/0.0 = inf etc. is classified by the result
}
int64_t scl_rt_f2i(double value, int64_t orig, uint8_t mode) {
  auto &s = state();
  if (s.vtable && s.vtable->f2i)
    return s.vtable->f2i(value, orig, mode);
  // awlsim insn_trunc/insn_rnd: out of int32 range -> OV=OS=1, ACCU unchanged
  if (!(value >= -2147483648.0 && value <= 2147483647.0)) {
    s.stw |= (1u << 4) | (1u << 5);
    return orig;
  }
  switch (mode) {
  case 1: return (int64_t)std::nearbyint(value); // RND: Python round() = half-to-even
  case 2: return (int64_t)std::ceil(value);    // RND+
  case 3: return (int64_t)std::floor(value);   // RND-
  default: return (int64_t)value;              // TRUNC
  }
}

// MARK: OB-temp presets (awlsim obtemp.OB1TempPresets; only OB 1 has them)
void scl_rt_ob_start(uint32_t obNum) {
  auto &s = state();
  if (s.vtable && s.vtable->ob_start) {
    s.vtable->ob_start(obNum);
    return;
  }
  // SPEC[AWL] Siemens AWL manual status word (STW) cleared at OB entry (awlsim statusWord.reset) -> scl_rt_ob_start
  s.stw = 0; // STW is cleared at OB entry (awlsim statusWord.reset)
  if (obNum != 1)
    return; // OBTempPresets_table only has OB 1
  // write the 20-byte header into the current L frame (bytes 0..19)
  auto put = [&](uint32_t byte, uint8_t w, uint64_t v) {
    scl_rt_write(SCL_RT_AREA_L, 0, byte, w, v);
  };
  auto bcd = [](uint32_t n) { return (uint8_t)(((n / 10) << 4) | (n % 10)); };
  put(0, 1, 0x11);      // EV_CLASS: cyclic OB
  put(1, 1, 0x03);      // SCAN_1
  put(2, 1, 0x01);      // PRIORITY
  put(3, 1, obNum);     // OB_NUMBR
  put(4, 1, 0);         // RESERVED_1
  put(5, 1, 0);         // RESERVED_2
  uint64_t ms = s.clockMs;
  uint64_t sec = ms / 1000;
  put(6, 2, s.lastCycleMs & 0x7FFF);  // PREV_CYCLE
  put(8, 2, s.minCycleMs & 0x7FFF);   // MIN_CYCLE
  put(10, 2, s.maxCycleMs & 0x7FFF);  // MAX_CYCLE
  // DATE_TIME (BCD), date fixed at 2024-01-01 plus the sim clock time-of-day
  put(12, 1, bcd(24));                       // year
  put(13, 1, bcd(1));                        // month
  put(14, 1, bcd(1));                        // day
  put(15, 1, bcd((sec / 3600) % 24));        // hour
  put(16, 1, bcd((sec / 60) % 60));          // minute
  put(17, 1, bcd(sec % 60));                 // second
  put(18, 1, bcd((ms % 1000) / 10));         // ms high
  put(19, 1, (uint8_t)(bcd(ms % 10) * 16 + 1)); // ms low + day of week
}

// __FEATURE t [, v] (awlsim insn_feature): t=0 number of accumulator
// registers (default 4), t=1 OB-temp presets enabled. value < 0: get.
int scl_rt_stop_requested(void) {
  auto &s = state();
  if (s.vtable && s.vtable->stop_requested)
    return s.vtable->stop_requested();
  return s.stopRequested ? 1 : 0;
}
// SPEC[AWL] awlsim __FEATURE t [, v] -> scl_rt_feature (nrAccus, OB-temp presets)
int32_t scl_rt_feature(uint32_t target, int32_t value) {
  auto &s = state();
  if (target == 0) {
    if (value >= 0)
      s.nrAccus = (uint32_t)value;
    return (int32_t)s.nrAccus;
  }
  if (target == 1) {
    if (value >= 0)
      s.obTempPresets = value != 0;
    return s.obTempPresets ? 1 : 0;
  }
  return 0;
}

// MARK: test oracles (ABI v4)
// awlsim semantics: operands are compared as scalars masked to their width
// (asScalar), i.e. the 32-bit ACCU and literals compare as unsigned 32-bit
// bit patterns. 64-bit-exact equality is also accepted (pointer/DT values).
static bool assertCheck(uint8_t op, int64_t actual, int64_t expected) {
  uint32_t a = (uint32_t)actual, b = (uint32_t)expected;
  switch (op) {
  case 0: return actual == expected || a == b;
  case 1: return actual != expected && a != b;
  case 2: return a > b;
  case 3: return a < b;
  case 4: return a >= b;
  default: return a <= b;
  }
}

static bool assertCheckR(uint8_t op, double actual, double expected) {
  switch (op) {
  case 0: return actual == expected;
  case 1: return actual != expected;
  case 2: return actual > expected;
  case 3: return actual < expected;
  case 4: return actual >= expected;
  default: return actual <= expected;
  }
}

void scl_rt_assert_cmp(uint32_t id, uint8_t op, int64_t actual,
                       int64_t expected) {
  auto &s = state();
  if (s.vtable && s.vtable->assert_cmp) {
    s.vtable->assert_cmp(id, op, actual, expected);
    return;
  }
  if (!assertCheck(op, actual, expected)) {
    ++s.assertFailures;
    fprintf(stderr, "ASSERT FAILED line %u: actual=%lld expected=%lld op=%u\n",
            id, (long long)actual, (long long)expected, op);
  }
}

void scl_rt_assert_cmp_r(uint32_t id, uint8_t op, double actual,
                         double expected) {
  auto &s = state();
  if (s.vtable && s.vtable->assert_cmp_r) {
    s.vtable->assert_cmp_r(id, op, actual, expected);
    return;
  }
  if (!assertCheckR(op, actual, expected)) {
    ++s.assertFailures;
    fprintf(stderr, "ASSERT FAILED line %u: actual=%f expected=%f op=%u\n",
            id, actual, expected, op);
  }
}

uint64_t scl_rt_assert_failures(void) { return state().assertFailures; }

void scl_rt_sim_dump_areas(void) {
  auto &s = state();
  if (s.vtable)
    return; // a host runtime owns the memory; nothing to dump
  auto dumpBuf = [](char area, uint32_t db, const std::vector<uint8_t> &buf) {
    bool any = false;
    for (size_t i = 0; i < buf.size(); ++i) {
      if (buf[i] == 0)
        continue;
      if (!any) {
        if (area == SCL_RT_AREA_DB)
          printf("DB%u:\n", db);
        else
          printf("%c:\n", area);
        any = true;
      }
      if (area == SCL_RT_AREA_DB)
        printf("  DB%u[%zu]=%u\n", db, i, buf[i]);
      else
        printf("  %c[%zu]=%u\n", area, i, buf[i]);
    }
  };
  for (char area : {'I', 'Q', 'M', 'L'})
    dumpBuf(area, 0, s.areaBytes[(uint8_t)area]);
  for (auto &kv : s.dbs)
    dumpBuf(SCL_RT_AREA_DB, kv.first, kv.second);
  for (uint32_t t = 0; t < 256; ++t) {
    if (!s.timers[t].running && s.timers[t].durMs == 0)
      continue;
    timerCheck(s.timers[t], s.clockMs);
    printf("T%u=%llums%s\n", t,
           (unsigned long long)s.timers[t].remainingMs,
           s.timers[t].status ? " on" : " off");
  }
  for (uint32_t c = 0; c < 256; ++c)
    if (s.counters[c].value != 0)
      printf("C%u=%u on\n", c, s.counters[c].value);
  printf("clock=%llums\n", (unsigned long long)s.clockMs);
}

} // extern "C"
