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
#include <map>
#include <vector>

namespace {

/// Default size of a simulated area (I/Q/M/T/C) in bytes; configurable via
/// scl_rt_sim_set_area_size() (driver CLI `--area-size`).
constexpr uint64_t kDefaultAreaSize = 65536;

struct SimState {
  std::vector<uint8_t> areaBytes[256];   // indexed by area code (ASCII)
  uint64_t areaSize[256] = {};           // configured size, 0 = default
  std::map<uint32_t, std::vector<uint8_t>> dbs; // DB number -> bytes
  std::map<uint32_t, uint64_t> dbSize;   // configured DB sizes, 0 = grow lazily
  const SclRtVTable *vtable = nullptr;
  uint64_t clockMs = 0;
};

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
  for (char area : {'I', 'Q', 'M', 'T', 'C'})
    dumpBuf(area, 0, s.areaBytes[(uint8_t)area]);
  for (auto &kv : s.dbs)
    dumpBuf(SCL_RT_AREA_DB, kv.first, kv.second);
  printf("clock=%llums\n", (unsigned long long)s.clockMs);
}

} // extern "C"
