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
  if (n <= 2) { // CTU / CTD / CTUD counters
    int cv = (int16_t)(uint16_t)(rd(6, 2) & 0xFFFF);
    int pv = (int16_t)(uint16_t)(rd(2, 2) & 0xFFFF);
    bool r = rdb(0, 2);
    bool ld = rdb(0, 3);
    bool ldEdge = ld && !s.sfbPrevLd[key];
    s.sfbPrevLd[key] = ld;
    if (ldEdge)
      cv = pv; // LD: load PV into CV
    if (n == 0) { // CTU
      bool cu = rdb(0, 0), cuo = rdb(8, 0);
      bool edge = cu && !cuo;
      wrb(8, 0, cu);
      if (r)
        cv = 0;
      else if (edge && cv < 32767)
        cv++;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv >= pv);
    } else if (n == 1) { // CTD
      bool cd = rdb(0, 1), cod = rdb(8, 1);
      bool edge = cd && !cod;
      wrb(8, 1, cd);
      if (r)
        cv = 0;
      else if (edge && cv > -32768)
        cv--;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv <= pv);
    } else { // CTUD
      bool cu = rdb(0, 0), cuo = rdb(8, 0);
      bool cd = rdb(0, 1), cod = rdb(8, 1);
      bool eu = cu && !cuo, ed = cd && !cod;
      wrb(8, 0, cu);
      wrb(8, 1, cd);
      if (r)
        cv = 0;
      else if (eu && cv < 32767)
        cv++;
      else if (ed && cv > -32768)
        cv--;
      wr(6, 2, (uint16_t)cv);
      wrb(4, 0, cv >= pv);
    }
    return;
  }
  // SFB3 TON / SFB4 TOF / SFB5 TP
  uint64_t pt = s5timeToMs(rd(2, 2) & 0xFFFF);
  bool in = rdb(0, 0);
  auto &st = s.sfbTimers[key];
  uint64_t now = s.clockMs;
  if (n == 3) { // TON
    if (in && !st.prevIn) {
      st.running = true;
      st.startMs = now;
      st.done = false;
    }
    if (st.running && !st.done && now - st.startMs >= pt) {
      st.running = false;
      st.done = true;
    }
    wrb(4, 0, st.done);
    wr(6, 2, st.done ? pt : (now - st.startMs));
  } else if (n == 4) { // TOF
    if (!in && st.prevIn) {
      st.running = true;
      st.startMs = now;
    }
    if (st.running && now - st.startMs >= pt)
      st.running = false;
    wrb(4, 0, in || st.running);
    wr(6, 2, in ? 0 : (now - st.startMs > pt ? pt : now - st.startMs));
  } else { // TP
    if (in && !st.prevIn) {
      st.running = true;
      st.startMs = now;
    }
    if (st.running && now - st.startMs >= pt)
      st.running = false;
    wrb(4, 0, st.running);
    wr(6, 2, st.running ? (now - st.startMs > pt ? pt : now - st.startMs) : 0);
  }
  st.prevIn = in;
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

void scl_rt_sim_reset(void) {
  auto &s = state();
  for (auto &b : s.areaBytes)
    b.clear();
  for (auto &sz : s.areaSize)
    sz = 0;
  s.dbs.clear();
  s.dbSize.clear();
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
