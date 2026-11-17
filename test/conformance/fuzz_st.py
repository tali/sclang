#!/usr/bin/env python3
# Differential fuzzer for the ST (IEC 61131-3) trace corpus: for each
# s-celles .st PROGRAM, generate random per-scan inputs (seeded, reproducible)
# and check that the SCL->JIT and SCL->AWL->JIT backends produce identical
# scan outputs. Complements run_st_traces.py --cross (fixed trace inputs)
# with inputs the traces don't cover.
#
# Usage:
#   python3 test/conformance/fuzz_st.py --sclang build.release/bin/sclang \
#       [--corpora <dir>] [--scans 3] [--seed 1] [--only <substr>]
import os
import random
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_st_traces as R

TIMEOUT_S = 25


def rand_value(rng, kind):
    if kind == 'bit':
        return rng.choice([0, 1])
    if kind in ('int',):
        return rng.choice([0, 1, -1, 7, -13, 255, -255, 32767, -32768,
                           rng.randint(-1000, 1000)])
    if kind in ('uint',):
        return rng.choice([0, 1, 255, 65535, rng.randint(0, 60000)])
    if kind == 'float':
        return rng.choice([0.0, 1.5, -2.75, 123456.75,
                           rng.uniform(-1e4, 1e4)])
    if kind == 'lreal':
        return rng.choice([0.0, 1.5e300, -2.75e-10, rng.uniform(-1e6, 1e6)])
    if kind == 'time':
        return rng.randint(0, 86400000)
    if kind == 'str':
        n = rng.randint(0, 8)
        return ''.join(rng.choice('abcXYZ019 ') for _ in range(n))
    return 0


def gen_fuzz_inputs(rng, nscans, in_slots, in_kinds):
    lines = []
    for idx in range(nscans):
        parts = ['%d' % idx]
        for s in in_slots:
            kind = in_kinds.get(s.name)
            if kind is None:
                continue
            v = rand_value(rng, kind)
            if kind == 'float':
                v = struct.unpack('<I', struct.pack('<f', v))[0]
            elif kind == 'lreal':
                v = struct.unpack('<Q', struct.pack('<d', v))[0]
            elif kind == 'str':
                parts.append("%s='%s'" % (s.name, v.replace("'", "$27")))
                continue
            parts.append('%s=%d' % (s.name, v))
        lines.append(' '.join(parts))
    return '\n'.join(lines) + '\n'


def close(a, b, kind):
    if kind in ('float', 'lreal'):
        if a != a and b != b:
            return True  # NaN
        scale = max(1.0, abs(a))
        return abs(a - b) <= 1e-5 * scale
    return a == b


def compare_obs(obs1, obs2, out_slots):
    kind_of = {s.name: s.kind for s in out_slots}
    errs = []
    for idx in sorted(set(obs1) | set(obs2)):
        o1, o2 = obs1.get(idx), obs2.get(idx)
        if o1 is None or o2 is None:
            errs.append('scan %d missing in one backend' % idx)
            continue
        for name, v1 in o1.items():
            if name not in o2:
                continue
            v2 = o2[name]
            kind = kind_of.get(name)
            if kind in ('float', 'lreal') and isinstance(v1, int) \
                    and isinstance(v2, int):
                w = 32 if kind == 'float' else 64
                f = 'f' if kind == 'float' else 'd'
                a = struct.unpack('<' + f, struct.pack('<Q' if w == 64 else '<I',
                                                       v1 & ((1 << w) - 1)))[0]
                b = struct.unpack('<' + f, struct.pack('<Q' if w == 64 else '<I',
                                                       v2 & ((1 << w) - 1)))[0]
                if not close(a, b, kind):
                    errs.append('scan %d %s: %r vs %r' % (idx, name, a, b))
            elif not close(v1, v2, kind):
                errs.append('scan %d %s: %r vs %r' % (idx, name, v1, v2))
    return errs


def fuzz_one(args, st_path, tmpdir, rng, nscans):
    trace = R.parse_trace(st_path[:-3] + '.trace')
    parsed = R.parse_program(st_path)
    if parsed is None:
        return None  # CONFIGURATION: not fuzzed
    if trace.get('status') == 'implementation-defined':
        return None  # undefined behavior may differ between backends
    prog, decls, raw, strlens = parsed
    in_decls = decls.get('VAR_INPUT', [])
    out_decls = decls.get('VAR_OUTPUT', [])
    in_kinds = dict(in_decls)
    out_kinds = dict(out_decls)
    for n in [n for n, _ in in_decls] + [n for n, _ in out_decls]:
        k = in_kinds.get(n) or out_kinds.get(n)
        if k not in R.TYPE_MAP:
            return None
    in_names = trace['inputs'] or [n for n, _ in in_decls]
    out_names = trace['outputs'] or [n for n, _ in out_decls]
    in_slots, nxt = R.alloc_slots(in_names, in_kinds, R.M_BASE)
    out_slots, _ = R.alloc_slots(out_names, out_kinds, nxt or R.M_BASE + 64,
                                 strlens)
    if in_slots is None or out_slots is None:
        return None
    wrapper = R.gen_wrapper(raw, prog, in_slots, out_slots)
    wpath = os.path.join(tmpdir, os.path.basename(st_path) + '.fuzz.scl')
    with open(wpath, 'w') as f:
        f.write(wrapper)
    infile = os.path.join(tmpdir, os.path.basename(st_path) + '.fuzz.in')
    with open(infile, 'w') as f:
        f.write(gen_fuzz_inputs(rng, nscans, in_slots, in_kinds))
    io_args = [s.scan_io_arg('in') for s in in_slots]
    io_args += [s.scan_io_arg('out') for s in out_slots]
    base = ['-x=st', '--emit=jit', '--jit-cycles', str(nscans),
            '--cycle-ms', str(trace['period_ms']), '--area-size', 'M=8192']
    base += io_args + ['--scan-inputs', infile, wpath]
    rc1, out1 = R.run_sclang(args.sclang, base, timeout=TIMEOUT_S)
    if rc1 not in (0, 2):
        return 'skip', 'scl jit rc=%d' % rc1
    obs1 = R.parse_scan_lines(out1)
    # cross: SCL->AWL->JIT via in-memory awl-dialect lowering (--emit=jit-awl)
    if R.has_var_initializers(raw):
        return 'skip', 'cross-skip'
    cross = R.awl_compat(wrapper, prog)
    cwpath = wpath + '.cross.scl'
    with open(cwpath, 'w') as f:
        f.write(cross)
    # pre-check: only the SCL->awl-dialect lowering; failure = not
    # representable in AWL -> skip (same condition as the trace harness)
    rcg, outg = R.run_sclang(args.sclang,
                             ['-x=st', '--emit=awl-mlir', cwpath],
                             timeout=TIMEOUT_S)
    if rcg != 0 or any(m in outg for m in R.CROSS_SKIP_MARKERS):
        return 'skip', 'cross-skip'
    base_awl = ['-x=st', '--emit=jit-awl', '--jit-cycles', str(nscans),
                '--cycle-ms', str(trace['period_ms']),
                '--area-size', 'M=65536', '--area-size', 'DB1=8192']
    base_awl += io_args + ['--scan-inputs', infile, cwpath]
    rc2, out2 = R.run_sclang(args.sclang, base_awl, timeout=TIMEOUT_S)
    if rc2 not in (0, 2):
        return 'fail', 'awl jit rc=%d %s' % (rc2, out2[-150:])
    obs2 = R.parse_scan_lines(out2)
    errs = compare_obs(obs1, obs2, out_slots)
    if errs:
        return 'fail', '; '.join(errs[:3])
    return 'pass', 'ok'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    ap.add_argument('--sclang-gen', default=None)
    ap.add_argument('--corpora', default='/Users/martin/scl-external')
    ap.add_argument('--scans', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    if args.sclang_gen is None:
        args.sclang_gen = os.path.join(os.path.dirname(args.sclang),
                                       'sclang-gen')
    corpora = os.path.join(args.corpora,
                           's-celles-iec61131-conformance', 'conformance')
    tests = []
    for root, _, files in os.walk(corpora):
        for fn in sorted(files):
            if fn.endswith('.st'):
                tests.append(os.path.join(root, fn))
    if args.only:
        tests = [t for t in tests if args.only in t]
    rng = random.Random(args.seed)
    npass = nfail = nskip = 0
    tmpdir = tempfile.mkdtemp(prefix='sclang-fuzz-')
    for st in sorted(tests):
        res = fuzz_one(args, st, tmpdir, rng, args.scans)
        if res is None:
            nskip += 1
            continue
        status, detail = res
        if status == 'pass':
            npass += 1
        elif status == 'skip':
            nskip += 1
        else:
            nfail += 1
            print('FAIL %s :: %s' % (os.path.relpath(st, corpora), detail))
    print('fuzz: %d pass, %d fail, %d skip (seed %d, %d scans each)' %
          (npass, nfail, nskip, args.seed, args.scans))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
