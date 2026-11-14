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
#include <algorithm>
#include <map>
#include <vector>

namespace {

/// Default size of a simulated area (I/Q/M/T/C) in bytes; configurable via
/// scl_rt_sim_set_area_size() (driver CLI `--area-size`).
constexpr uint64_t kDefaultAreaSize = 65536;

struct TimerState {
  uint8_t mode = 0;      // SclRtTimerMode
  bool running = false;
  uint64_t startMs = 0;  // clock at (re)start
  uint64_t durMs = 0;    // decoded duration in ms
};

struct SimState {
  std::vector<uint8_t> areaBytes[256];   // indexed by area code (ASCII)
  uint64_t areaSize[256] = {};           // configured size, 0 = default
  std::map<uint32_t, std::vector<uint8_t>> dbs; // DB number -> bytes
  std::map<uint32_t, uint64_t> dbSize;   // configured DB sizes, 0 = grow lazily
  const SclRtVTable *vtable = nullptr;
  uint64_t clockMs = 0;
  TimerState timers[256];                // indexed by timer number
  uint32_t counters[256] = {};           // counter values (0..999)
  // builtin timer FB (TON/TOF/TP) state, keyed by instance DB buffer address
  struct TimerFbState {
    bool prevIn = false;
    bool running = false;
    uint64_t startMs = 0;
  };
  std::map<uint64_t, TimerFbState> timerFbs;
};

/// Decode an S5TIME-encoded 16-bit value to milliseconds.
static uint64_t s5timeToMs(uint64_t s5) {
  static const uint64_t baseMs[4] = {10, 100, 1000, 10000};
  uint64_t base = (s5 >> 12) & 3;
  uint64_t bcd = s5 & 0xFFF;
  uint64_t val = ((bcd >> 8) & 0xF) * 100 + ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
  return val * baseMs[base];
}

/// Current timer value in ms: while running, the remaining time (0 when
/// expired); when stopped, the preset (writing %T sets the preset, as on S7).
static uint64_t timerRemainingMs(const TimerState &t, uint64_t clockMs) {
  if (!t.running)
    return t.durMs;
  uint64_t elapsed = clockMs - t.startMs;
  return elapsed >= t.durMs ? 0 : t.durMs - elapsed;
}

/// Timer status bit (Q output): on-delay timers switch on after the delay,
/// the others while running.
static bool timerStatus(const TimerState &t, uint64_t clockMs) {
  if (!t.running)
    return false;
  bool expired = (clockMs - t.startMs) >= t.durMs;
  if (t.mode == SCL_RT_TIMER_ON_DELAY)
    return expired;
  return !expired;
}

SimState &state() {
  static SimState s;
  return s;
}

/// Return the byte buffer for an access, growing DBs on demand and checking
/// bounds of the fixed-size areas.
std::vector<uint8_t> &bufferFor(uint8_t area, uint32_t db, uint32_t byteOff,
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
  auto &buf = s.areaBytes[area];
  if (buf.empty()) {
    uint64_t size = s.areaSize[area] ? s.areaSize[area] : kDefaultAreaSize;
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
    fprintf(stderr, "scl_rt_read: invalid width %u\n", widthBytes);
    abort();
  }
  // Timers/counters: the value is computed from the sim state (remaining
  // time in ms / counter value), not stored in the area buffer.
  if (area == SCL_RT_AREA_T && widthBytes > 1) {
    uint64_t v = timerRemainingMs(state().timers[byteOff], state().clockMs);
    if (widthBytes < 8) {
      uint64_t signBit = 1ull << (widthBytes * 8 - 1);
      if (v & signBit)
        v |= ~0ull << (widthBytes * 8);
    }
    return (int64_t)v;
  }
  if (area == SCL_RT_AREA_C && widthBytes > 1)
    return (int64_t)state().counters[byteOff];
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
  // Writing a counter value (e.g. SCL `%Cn := v`) sets the count; writing a
  // timer value (e.g. `%TWn := v`) sets the preset (TV).
  if (area == SCL_RT_AREA_C && widthBytes > 1) {
    state().counters[byteOff] = (uint32_t)(value > 999 ? 999 : value);
    return;
  }
  if (area == SCL_RT_AREA_T && widthBytes > 1) {
    state().timers[byteOff].durMs = value;
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
  // Timer/counter status bit (AWL `U T n` / `U Z n`).
  if (bit == 0 && area == SCL_RT_AREA_T)
    return timerStatus(state().timers[byteOff], state().clockMs);
  if (bit == 0 && area == SCL_RT_AREA_C)
    return state().counters[byteOff] > 0;
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

void scl_rt_timer_start(uint32_t t, uint8_t mode, uint64_t durS5) {
  auto &s = state();
  if (s.vtable && s.vtable->timer_start) {
    s.vtable->timer_start(t, mode, durS5);
    return;
  }
  if (t > 255) {
    fprintf(stderr, "scl_rt_timer_start: invalid timer %u\n", t);
    abort();
  }
  uint64_t durMs = s5timeToMs(durS5);
  TimerState &tm = s.timers[t];
  if (mode == SCL_RT_TIMER_EXT_PULSE && tm.running) {
    // SE retrigger extends the running pulse
    tm.durMs = timerRemainingMs(tm, s.clockMs) + durMs;
    tm.startMs = s.clockMs;
  } else {
    tm.mode = mode;
    tm.running = true;
    tm.startMs = s.clockMs;
    tm.durMs = durMs;
  }
}

void scl_rt_counter_op(uint32_t c, uint8_t op, uint64_t value) {
  auto &s = state();
  if (s.vtable && s.vtable->counter_op) {
    s.vtable->counter_op(c, op, value);
    return;
  }
  if (c > 255) {
    fprintf(stderr, "scl_rt_counter_op: invalid counter %u\n", c);
    abort();
  }
  switch (op) {
  case SCL_RT_COUNTER_UP:
    s.counters[c] = s.counters[c] >= 999 ? 999 : s.counters[c] + 1;
    break;
  case SCL_RT_COUNTER_DOWN:
    s.counters[c] = s.counters[c] == 0 ? 0 : s.counters[c] - 1;
    break;
  case SCL_RT_COUNTER_SET:
    s.counters[c] = (uint32_t)(value > 999 ? 999 : value);
    break;
  case SCL_RT_COUNTER_RESET:
    s.counters[c] = 0;
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
  s.clockMs = 0;
  for (auto &t : s.timers)
    t = TimerState{};
  for (auto &c : s.counters)
    c = 0;
}

uint64_t scl_rt_sim_clock_ms(void) { return state().clockMs; }

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
    printf("T%u=%llums%s\n", t,
           (unsigned long long)timerRemainingMs(s.timers[t], s.clockMs),
           timerStatus(s.timers[t], s.clockMs) ? " on" : " off");
  }
  for (uint32_t c = 0; c < 256; ++c)
    if (s.counters[c] != 0)
      printf("C%u=%u%s\n", c, s.counters[c], s.counters[c] > 0 ? " on" : "");
  printf("clock=%llums\n", (unsigned long long)s.clockMs);
}

} // extern "C"
