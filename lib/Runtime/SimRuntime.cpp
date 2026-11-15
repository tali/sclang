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
  std::vector<uint8_t> areaBytes[256];   // indexed by area code (ASCII)
  uint64_t areaSize[256] = {};           // configured size, 0 = default
  std::map<uint32_t, std::vector<uint8_t>> dbs; // DB number -> bytes
  std::map<uint32_t, uint64_t> dbSize;   // configured DB sizes, 0 = grow lazily
  std::map<uint32_t, uint64_t> dbRefHighWater; // highest referenced DB byte
  const SclRtVTable *vtable = nullptr;
  uint64_t clockMs = 0;
  TimerState timers[256];                // indexed by timer number
  CounterState counters[256];            // indexed by counter number
  // builtin timer FB (TON/TOF/TP) state, keyed by instance DB buffer address
  struct TimerFbState {
    bool prevIn = false;
    bool running = false;
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
  uint64_t lBase = 0;
  // call-saved block-call state, saved/restored with the L frame (S7 restores
  // the address registers and DB/DI registers when a block returns)
  struct FrameSave {
    uint64_t ar1, ar2;
    uint32_t db, di;
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

/// Byte offset of the AR2 multi-instance base (awlsim adds the AR2 base to
/// every DI-area access, see cpu.py getArea_DI/setArea_DI).
static uint32_t ar2ByteOff() {
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
  if (area == SCL_RT_AREA_L)
    byteOff += (uint32_t)s.lBase; // L-area access is frame-relative
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

extern "C" {

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

void scl_rt_l_push(uint32_t bytes) {
  auto &s = state();
  s.lFrames.push_back(s.lBase);
  s.frameSaves.push_back({s.ars[0], s.ars[1], s.currentDb, s.currentDi});
  // frames are even-aligned and always advance by at least 2 bytes, so
  // blocks without TEMP vars still get a separate L frame (a callee must
  // not clobber the caller's LB/LW/LD local data)
  uint64_t adv = (bytes + 1) & ~uint64_t(1);
  s.lBase += adv < 2 ? 2 : adv;
}

void scl_rt_l_pop(void) {
  auto &s = state();
  if (!s.lFrames.empty()) {
    s.lBase = s.lFrames.back();
    s.lFrames.pop_back();
    if (!s.frameSaves.empty()) {
      auto &f = s.frameSaves.back();
      s.ars[0] = f.ar1;
      s.ars[1] = f.ar2;
      s.currentDb = f.db;
      s.currentDi = f.di;
      s.frameSaves.pop_back();
    }
  }
}

// Base of the FC parameter area within an L frame; must match kLocalBase
// in lib/AwlGen/AwlGen.cpp.
static const uint32_t kParamBase = 8192;

// Copy the FC parameter area between the absolute marker staging area
// (where the call site stores the actual parameters) and the callee's own L
// frame at kParamBase. param_in runs after the callee's scl_rt_l_push,
// param_out before scl_rt_l_pop, so the callee's frame is current in both.
static void paramCopy(uint32_t bytes, bool toCallee) {
  auto &s = state();
  auto &lbuf = s.areaBytes[SCL_RT_AREA_L];
  if (lbuf.empty())
    lbuf.resize(kDefaultLAreaSize, 0);
  auto &mbuf = s.areaBytes[SCL_RT_AREA_M];
  if (mbuf.empty())
    mbuf.resize(kDefaultAreaSize, 0);
  uint64_t lOff = s.lBase + kParamBase;   // callee frame parameter area
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
  if (n > 5)
    return; // unknown SFB number (e.g. CALL SFB -1): no-op, BIE stays 1
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
    return;
  }
  // SFB3 TP / SFB4 TON / SFB5 TOF (awlsim system_sfb_3/4/5). Instance layout:
  // IN@0.0, PT@2 (32-bit TIME, ms), Q@6.0, ET@8 (32-bit), STATE@12 (byte),
  // STIME@14 (32-bit), ATIME@18 (32-bit). The STATE/STIME/ATIME live in the
  // instance DB itself.
  uint64_t now = s.clockMs;
  int64_t pt = (int32_t)(uint32_t)rd(2, 4); // PT: 32-bit TIME in ms
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
  if (pt < 0)
    pt = 0;
  uint64_t now = s.clockMs;
  uint64_t elapsed = st.running ? now - st.startMs : 0;
  bool q = false;
  int32_t et = 0;
  switch (mode) {
  case 0: // TON: on-delay
    if (in && !st.prevIn) {
      st.running = true;
      st.startMs = now;
      elapsed = 0;
    }
    if (!in)
      st.running = false;
    q = st.running && elapsed >= (uint64_t)pt;
    et = st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt) : 0;
    break;
  case 1: // TOF: off-delay
    if (!in && st.prevIn) {
      st.running = true;
      st.startMs = now;
      elapsed = 0;
    }
    if (in)
      st.running = false;
    q = st.running && elapsed < (uint64_t)pt;
    et = st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt) : 0;
    break;
  default: // TP: pulse
    if (in && !st.prevIn) {
      st.running = true;
      st.startMs = now;
      elapsed = 0;
    }
    if (st.running && elapsed >= (uint64_t)pt)
      st.running = false;
    q = st.running && elapsed < (uint64_t)pt;
    et = in && st.running ? (int32_t)std::min<uint64_t>(elapsed, (uint64_t)pt) : 0;
    break;
  }
  st.prevIn = in;
  fbWriteByte(p, 8, q ? 1 : 0);
  fbWriteI32(p, 12, et);
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

uint64_t scl_rt_ar_get(uint8_t ar) {
  auto &s = state();
  if (ar >= 1 && ar <= 2)
    return s.ars[ar - 1];
  return 0;
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
  if (p < 0x100000000ull && s7 >= 0x80 && s7 <= 0x86) {
    static const uint8_t s7area[7] = {'P', 'I', 'Q', 'M', 'D', 'D', 'L'};
    uint8_t area = s7area[s7 - 0x80];
    // 0x85 = DI area: the DB number comes from the DI register (db=0xFFFF)
    uint32_t db = s7 == 0x85 ? 0xFFFF : (uint32_t)((p >> 16) & 0xFF);
    uint32_t byte = (uint32_t)((p >> 3) & 0x1FFF);
    uint8_t bit = (uint8_t)(p & 7);
    return packPtr(area, db, byte, bit, 0);
  }
  return packPtr(SCL_RT_AREA_M, 0, (uint32_t)(p / 8), (uint8_t)(p % 8), 0);
}

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
void scl_rt_sfc(uint32_t n) {
  auto &s = state();
  if (s.vtable && s.vtable->sfc) {
    s.vtable->sfc(n);
    return;
  }
  if (n == 64) { // TIME_TCK: 31-bit millisecond timestamp
    scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + 0, 4,
                 s.clockMs & 0x7FFFFFFF);
    return;
  }
  if (n == 47) { // WAIT: WT in microseconds, max 32767 (awlsim sfc_47)
    int wt = (int16_t)(uint16_t)scl_rt_read(SCL_RT_AREA_M, 0, kParamBase + 0, 2);
    if (wt > 0)
      s.clockMs += ((uint64_t)std::min(wt, 32767) + 999) / 1000;
    return;
  }
  if (n != 21)
    return; // other SFCs: host-provided stub
  // SFC 21 FILL (awlsim system_sfc_21): fill the BLK ANY block with BVAL.
  // The frontend stages the full 10-byte ANY headers inline in the marker
  // area: BVAL @ kParamBase+0, BLK @ kParamBase+10, RET_VAL (i16) @ +20.
  constexpr uint32_t BVAL = 0, BLK = 10, RETV = 20;
  auto fail = [&](uint16_t code) {
    scl_rt_write(SCL_RT_AREA_M, 0, kParamBase + RETV, 2, code);
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
      if (dbNr == 0 || dbNr >= 0x100)
        return 1;
      outArea = SCL_RT_AREA_DB;
      outDb = dbNr;
      return 0;
    }
    static const uint8_t s7area[7] = {'P', 'I', 'Q', 'M', 'D', 'D', 'L'};
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
    if (area == SCL_RT_AREA_L)
      off += (uint32_t)s.lBase;
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
  for (char area : {'I', 'Q', 'M'})
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
