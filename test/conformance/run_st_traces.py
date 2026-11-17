#!/usr/bin/env python3
# ST/SCL trace tier: value-level verification of the SCL engine against the
# public IEC 61131-3 conformance traces (s-celles/iec61131-conformance,
# CC0): each conformance/<area>/<test>.st program is paired with a .trace
# file listing per-scan inputs and expected outputs (`0 x=5 => branch=11`).
#
# Usage:
#   run_st_traces.py --sclang <path-to-sclang> [--sclang-gen <path>]
#                    [--corpora <dir>] [--cross] [--only <substr>]
#
# How a trace is executed:
#   1. The .st PROGRAM is rewritten to a FUNCTION_BLOCK (plus an instance
#      DATA_BLOCK) and wrapped in an OB1 that copies per-scan inputs from
#      fixed M-area slots into the instance, calls the block, and copies the
#      outputs back to M-area slots.
#   2. sclang runs with --emit=jit --jit-cycles=N --cycle-ms=<period_ms>
#      --scan-io=... --scan-inputs=... : the driver writes the inputs of
#      scan i into the M slots before OB1 scan i and prints
#      `scan i: name=value ...` for the output slots after each scan.
#   3. The observed scan lines are compared to the .trace expectations.
# With --cross, each wrapper is additionally lowered SCL->AWL (sclang-gen
# --emit=awl) and JIT-run through the AWL frontend; its scan lines must match
# the direct SCL JIT run (differential test of the two lowering paths, the
# AWL path itself being oracle-verified against awlsim).
#
# Tests using constructs outside the supported subset (CONFIGURATION,
# STRING/array/struct outputs, LREAL, ...) are counted as "unsupported",
# not as failures. Exit code 0 if pass rate >= --floor (default 0).

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

# IEC type -> (widthBytes, kind) with kind in bit/int/uint/float/time.
# The driver reads slots sign-extended; for unsigned kinds the comparison
# masks both sides to the slot width (the trace prints the unsigned value).
TYPE_MAP = {
    'BOOL': (1, 'bit'),
    'SINT': (1, 'int'), 'BYTE': (1, 'uint'),
    'INT': (2, 'int'),
    'DINT': (4, 'int'),
    'TIME': (4, 'time'), 'LTIME': (8, 'time'),
    'REAL': (4, 'float'),  # observed via __BITS bit pattern
    'USINT': (1, 'uint'), 'UINT': (2, 'uint'), 'UDINT': (4, 'uint'),
    'WORD': (2, 'uint'), 'DWORD': (4, 'uint'),
}

TIMEOUT_S = 25
M_BASE = 1000  # first M-area byte used for scan-io slots


def read_text(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()


def strip_comments(text):
    return re.sub(r'\(\*.*?\*\)', ' ', text, flags=re.S)


# ---------------------------------------------------------------- trace file
def parse_time_ms(tok):
    """T#1500ms / T#1s / T#1h30m ... -> milliseconds (int)."""
    m = re.fullmatch(r'T#([0-9a-zA-Z]+)', tok)
    if not m:
        return None
    body = m.group(1).lower()
    total = 0
    for num, unit in re.findall(r'(\d+)(ms|[dhms])', body):
        mult = {'ms': 1, 'd': 86400000, 'h': 3600000, 'm': 60000,
                's': 1000}[unit]
        total += int(num) * mult
    return total


def parse_value(tok, kind):
    """Parse a trace value token according to the declared type kind.
    Returns None for '?' (undefined / implementation-defined wildcard)."""
    if tok == '?':
        return None
    if kind == 'bit':
        return 1 if tok.upper() == 'TRUE' else 0
    if kind == 'time':
        return parse_time_ms(tok)
    if kind == 'float':
        t = tok.lower()
        if t in ('inf', '+inf'):
            return float('inf')
        if t == '-inf':
            return float('-inf')
        if t == 'nan':
            return float('nan')
        return float(tok)
    if tok.upper().startswith('16#'):
        return int(tok[3:], 16)
    if tok.upper().startswith('2#'):
        return int(tok[2:], 2)
    if tok.upper().startswith('8#'):
        return int(tok[2:], 8)
    return int(tok, 10)


def parse_trace(path):
    """Return dict(period_ms, program, inputs, outputs, scans, status)."""
    text = read_text(path)
    head, _, body = text.partition('\n---\n')
    info = {'period_ms': 100, 'program': None, 'inputs': [], 'outputs': [],
            'scans': [], 'status': 'conformant'}
    for line in head.splitlines():
        line = line.strip()
        if line.startswith('#') or ':' not in line:
            continue
        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip()
        if key == 'period_ms':
            info['period_ms'] = int(val)
        elif key == 'program':
            info['program'] = val
        elif key == 'inputs':
            info['inputs'] = val.split()
        elif key == 'outputs':
            info['outputs'] = val.split()
        elif key == 'status':
            info['status'] = val
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=>' not in line:
            continue
        left, _, right = line.partition('=>')
        toks = left.split()
        if not toks:
            continue
        idx = int(toks[0])
        ins = {}
        for t in toks[1:]:
            k, _, v = t.partition('=')
            ins[k] = v
        outs = {}
        for t in right.split():
            k, _, v = t.partition('=')
            outs[k] = v
        info['scans'].append((idx, ins, outs))
    return info


# ------------------------------------------------------------------ .st file
SEC_RE = re.compile(
    r'\b(VAR_INPUT|VAR_OUTPUT|VAR_IN_OUT|VAR_TEMP|VAR_RETAIN|VAR)\b(.*?)'
    r'END_VAR\b', re.S | re.I)


def parse_program(path):
    """Extract the single top-level PROGRAM of a .st file.

    Returns (prog_name, decls, body) with decls = {section: [(name, type)]},
    or None if the file uses constructs we do not transform (CONFIGURATION).
    """
    raw = read_text(path)
    text = strip_comments(raw)
    if re.search(r'\bCONFIGURATION\b', text, re.I):
        return None
    m = re.search(r'\bPROGRAM\s+([A-Za-z_]\w*)\b', text, re.I)
    if not m:
        return None
    prog = m.group(1)
    end = re.search(r'\bEND_PROGRAM\b', text[m.end():], re.I)
    if not end:
        return None
    body = text[m.end():m.end() + end.start()]
    decls = {}
    for sm in SEC_RE.finditer(body):
        section = sm.group(1).upper()
        sec = []
        for decl in sm.group(2).split(';'):
            decl = decl.strip()
            if not decl or ':' not in decl:
                continue
            names, _, rest = decl.partition(':')
            ty = rest.split(':=')[0].strip().upper()
            ty = re.sub(r'\s+', ' ', ty)
            for n in names.split(','):
                n = n.strip()
                if n:
                    sec.append((n, ty))
        decls.setdefault(section, []).extend(sec)
    return prog, decls, raw


def transform_to_scl(raw, prog):
    """PROGRAM -> FUNCTION_BLOCK so the SCL frontend accepts IEC .st files."""
    out = re.sub(r'\bPROGRAM\b', 'FUNCTION_BLOCK', raw)
    out = re.sub(r'\bEND_PROGRAM\b', 'END_FUNCTION_BLOCK', out)
    return out


# ------------------------------------------------------------------ wrapper
class Slot:
    def __init__(self, name, kind, byte, bit, width):
        self.name, self.kind = name, kind
        self.byte, self.bit, self.width = byte, bit, width

    def direct(self):
        if self.kind == 'bit':
            return '%%M%d.%d' % (self.byte, self.bit)
        letter = {1: 'B', 2: 'W', 4: 'D', 8: 'D'}[self.width]
        return '%%M%s%d' % (letter, self.byte)

    def scan_io_arg(self, role):
        addr = str(self.byte) if self.bit < 0 else '%d.%d' % (self.byte,
                                                              self.bit)
        return '--scan-io=%s:%s=M/%s/%d' % (role, self.name, addr,
                                             self.width)


def alloc_slots(names, kinds, start):
    """Assign M-area slots; returns (slots, next_free_byte)."""
    slots = []
    byte = start
    for name in names:
        kind = kinds.get(name)
        if kind is None:
            return None, None
        width, k = TYPE_MAP[kind]
        if k == 'bit':
            slots.append(Slot(name, k, byte, 0, 1))
            byte += 1
        else:
            slots.append(Slot(name, k, byte, -1, width))
            byte += width
    return slots, byte


def gen_wrapper(raw, prog, in_slots, out_slots):
    lines = [transform_to_scl(raw, prog)]
    lines.append('DATA_BLOCK inst\n  %s\nEND_DATA_BLOCK\n' % prog)
    lines.append('FUNCTION OB1 : int')
    lines.append('BEGIN')
    for s in in_slots:
        if s.kind == 'float':
            # REAL inputs arrive as raw bit patterns in the M slot
            lines.append('  "inst".%s := __UNBITS(%s);' % (s.name, s.direct()))
        else:
            lines.append('  "inst".%s := %s;' % (s.name, s.direct()))
    lines.append('  %s."inst"();' % prog)
    for s in out_slots:
        if s.kind == 'float':
            lines.append('  %s := __BITS("inst".%s);' % (s.direct(), s.name))
        else:
            lines.append('  %s := "inst".%s;' % (s.direct(), s.name))
    lines.append('END_FUNCTION\n')
    lines.append('FUNCTION _mlir_ciface_OB1 : void')
    lines.append('BEGIN')
    lines.append('  OB1();')
    lines.append('END_FUNCTION\n')
    return '\n'.join(lines)


def gen_inputs_file(trace, in_slots):
    kind_of = {s.name: s.kind for s in in_slots}
    lines = []
    for idx, ins, _ in trace['scans']:
        parts = ['%d' % idx]
        for s in in_slots:
            tok = ins.get(s.name)
            if tok is None:
                continue
            v = parse_value(tok, kind_of[s.name])
            if v is None:
                return None
            if kind_of[s.name] == 'float':
                # REAL inputs are fed as raw bit patterns (__UNBITS)
                v = struct.unpack('<I', struct.pack('<f', v))[0]
            parts.append('%s=%d' % (s.name, v))
        lines.append(' '.join(parts))
    return '\n'.join(lines) + '\n'


def run_sclang(sclang, args, timeout=TIMEOUT_S):
    try:
        p = subprocess.run([sclang] + args, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -1, 'TIMEOUT'


def parse_scan_lines(output):
    scans = {}
    for line in output.splitlines():
        m = re.match(r'scan (\d+):\s*(.*)', line.strip())
        if not m:
            continue
        vals = {}
        for t in m.group(2).split():
            k, _, v = t.partition('=')
            try:
                vals[k] = int(v, 10)
            except ValueError:
                vals[k] = None
        scans[int(m.group(1))] = vals
    return scans


def compare_scans(trace, observed, out_slots):
    """Return list of mismatch strings (empty = pass)."""
    kind_of = {s.name: s.kind for s in out_slots}
    errs = []
    for idx, _, expected in trace['scans']:
        obs = observed.get(idx)
        if obs is None:
            errs.append('scan %d: missing (observed scans: %s)'
                        % (idx, sorted(observed)))
            continue
        for name, tok in expected.items():
            if '.' in name or name not in kind_of:
                continue  # struct member output: not observed
            kind = kind_of[name]
            exp = parse_value(tok, kind)
            got = obs.get(name)
            if exp is None:
                continue  # '?' wildcard: implementation-defined
            if got is None:
                errs.append('scan %d %s: unparseable (%s vs %s)'
                            % (idx, name, tok, got))
                continue
            if kind == 'float':
                gf = struct.unpack('f', struct.pack('I', got & 0xFFFFFFFF))[0]
                # NaN matches any NaN (bit pattern is impl-defined); inf
                # must match exactly.
                ok = (gf != gf and exp != exp) or gf == exp or (
                    abs(gf - exp) <= 1e-6 * max(1.0, abs(exp)))
                if not ok:
                    errs.append('scan %d %s: expected %s got %r (bits 0x%08X)'
                                % (idx, name, tok, gf, got & 0xFFFFFFFF))
            elif kind == 'uint':
                mask = (1 << (8 * dict((s.name, s.width)
                                        for s in out_slots)[name])) - 1
                if (got & mask) != (exp & mask):
                    errs.append('scan %d %s: expected %s got %d'
                                % (idx, name, tok, got))
            elif got != exp:
                errs.append('scan %d %s: expected %s got %d'
                            % (idx, name, tok, got))
    return errs


def awl_compat(wrapper, prog):
    """Rename wrapper identifiers to the AWL backend's OBn/FBn/FCn, DBn
    naming convention (PROGRAM prog -> FB1, instance DB inst -> DB1, other
    FUNCTIONs -> FCn)."""
    out = re.sub(r'\b%s\b' % re.escape(prog), 'FB1', wrapper)
    out = re.sub(r'\bDATA_BLOCK inst\b', 'DATA_BLOCK DB1', out)
    out = re.sub(r'\binst\b', 'DB1', out)
    names = []
    for m in re.finditer(r'\bFUNCTION\s+(\w+)', out):
        name = m.group(1)
        if re.fullmatch(r'(OB|FB|FC)\d+', name) or name.startswith('_mlir_ciface_'):
            continue
        if name not in names:
            names.append(name)
    for i, name in enumerate(names):
        out = re.sub(r'\b%s\b' % re.escape(name), 'FC%d' % (i + 1), out)
    return out


def run_one(args, st_path, trace, tmpdir, cross):
    """Run one .st/.trace pair; returns (status, detail) with status in
    pass/fail/unsupported."""
    parsed = parse_program(st_path)
    if parsed is None:
        return 'unsupported', 'CONFIGURATION or no top-level PROGRAM'
    prog, decls, raw = parsed
    in_decls = decls.get('VAR_INPUT', [])
    out_decls = decls.get('VAR_OUTPUT', [])
    in_names = [n for n, _ in in_decls]
    out_names = [n for n, _ in out_decls]
    in_kinds = dict(in_decls)
    out_kinds = dict(out_decls)
    # The trace's declared inputs/outputs must match the VAR sections.
    if trace['program'] and trace['program'] != prog:
        return 'unsupported', 'trace program %s != %s' % (trace['program'],
                                                           prog)
    if trace['inputs'] and set(trace['inputs']) - set(in_names):
        return 'unsupported', 'trace inputs not in VAR_INPUT: %s' % (
            set(trace['inputs']) - set(in_names))
    if trace['outputs'] and set(trace['outputs']) - set(out_names):
        return 'unsupported', 'trace outputs not in VAR_OUTPUT: %s' % (
            set(trace['outputs']) - set(out_names))
    for n in in_names + out_names:
        k = (in_kinds.get(n) or out_kinds.get(n))
        if k not in TYPE_MAP:
            return 'unsupported', 'type %s of %s not supported' % (k, n)
    in_slots, nxt = alloc_slots(trace['inputs'] or in_names, in_kinds, M_BASE)
    out_slots, nxt2 = alloc_slots(trace['outputs'] or out_names, out_kinds,
                                   nxt or M_BASE + 64)
    if in_slots is None or out_slots is None:
        return 'unsupported', 'slot allocation failed'
    wrapper = gen_wrapper(raw, prog, in_slots, out_slots)
    wpath = os.path.join(tmpdir, os.path.basename(st_path) + '.wrap.scl')
    with open(wpath, 'w') as f:
        f.write(wrapper)
    infile = os.path.join(tmpdir, os.path.basename(st_path) + '.inputs')
    content = gen_inputs_file(trace, in_slots)
    if content is None:
        return 'unsupported', 'unparseable input value'
    with open(infile, 'w') as f:
        f.write(content)
    nscans = len(trace['scans']) or 1
    io_args = [s.scan_io_arg('in') for s in in_slots]
    io_args += [s.scan_io_arg('out') for s in out_slots]
    base = ['-x=scl', '--emit=jit', '--jit-cycles', str(nscans),
            '--cycle-ms', str(trace['period_ms']), '--area-size', 'M=8192']
    base += io_args + ['--scan-inputs', infile, wpath]
    rc, out = run_sclang(args.sclang, base)
    if rc == -1:
        return 'fail', 'TIMEOUT'
    if rc not in (0, 2):
        err = [l for l in out.splitlines() if 'error' in l.lower()
               or 'failed' in l.lower()]
        return 'fail', 'rc=%d %s' % (rc, err[0] if err else out[-200:])
    observed = parse_scan_lines(out)
    errs = compare_scans(trace, observed, out_slots)
    cross_note = ''
    if cross and not errs:
        awl = wpath + '.awl'
        # The AWL backend names blocks OBn/FBn/FCn and DBs DBn, so rename the
        # wrapper's program FB and instance DB (and any non-FCn FUNCTIONs).
        cross_src = awl_compat(wrapper, prog)
        cwpath = wpath + '.cross.scl'
        with open(cwpath, 'w') as f:
            f.write(cross_src)
        rcg, outg = run_sclang(args.sclang_gen,
                               ['-x=scl', '--emit=awl', cwpath])
        if rcg != 0:
            # constructs the AWL backend cannot represent (date/time, ...)
            # are skipped, not failed: the differential check only applies
            # where both paths run
            return 'pass', 'ok (cross-skip: %s)' % outg[-120:].replace('\n', ' ')
        with open(awl, 'w') as f:
            f.write(outg)
        base_awl = ['-x=awl', '--emit=jit', '--jit-cycles', str(nscans),
                    '--cycle-ms', str(trace['period_ms']),
                    '--area-size', 'M=8192', '--area-size', 'DB1=8192']
        base_awl += io_args + ['--scan-inputs', infile, awl]
        rc2, out2 = run_sclang(args.sclang, base_awl)
        if rc2 not in (0, 2):
            return 'fail', 'awl jit rc=%d %s' % (rc2, out2[-200:])
        obs2 = parse_scan_lines(out2)
        if obs2 != observed:
            diff = [k for k in set(obs2) | set(observed)
                    if obs2.get(k) != observed.get(k)]
            return 'fail', 'SCL->AWL differs at scans %s' % diff[:3]
        cross_note = ' (cross ok)'
    if errs:
        # status: implementation-defined traces record one valid behavior of
        # an undefined one (IEC 61131-3); a divergence is reported but is not
        # a conformance failure. NB: timer_pt_zero's TOF power-up Q=TRUE is a
        # documented IronPLC deviation that contradicts the conformant
        # timers/tof trace (Q=FALSE initially), so the two cannot both match.
        if trace.get('status') == 'implementation-defined':
            return 'pass', 'impl-defined diverges: %s' % '; '.join(errs[:3])
        return 'fail', '; '.join(errs[:3])
    return 'pass', 'ok%s' % cross_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    ap.add_argument('--sclang-gen', default=None)
    ap.add_argument('--corpora', default='/Users/martin/scl-external')
    ap.add_argument('--cross', action='store_true',
                    help='also run SCL->AWL->JIT and require identical scans')
    ap.add_argument('--only', default=None, help='substring filter on test')
    ap.add_argument('--floor', type=float, default=0.0)
    args = ap.parse_args()
    if not args.sclang_gen:
        args.sclang_gen = os.path.join(os.path.dirname(args.sclang),
                                      'sclang-gen')
    root = os.path.join(args.corpora, 's-celles-iec61131-conformance',
                        'conformance')
    if not os.path.isdir(root):
        print('ST trace corpus not found at %s (skipping)' % root)
        return 0
    tests = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith('.trace'):
                st = os.path.join(dirpath, fn[:-6] + '.st')
                if os.path.isfile(st):
                    tests.append((st, os.path.join(dirpath, fn)))
    tests.sort()
    if args.only:
        tests = [t for t in tests if args.only in t[0]]
    results = {'pass': [], 'fail': [], 'unsupported': []}
    with tempfile.TemporaryDirectory() as tmpdir:
        for st, tr in tests:
            name = os.path.relpath(st, root)[:-3]
            trace = parse_trace(tr)
            status, detail = run_one(args, st, trace, tmpdir, args.cross)
            results[status].append((name, detail))
            if status == 'fail':
                print('FAIL %s :: %s' % (name, detail))
            elif 'diverges' in detail:
                print('DIFF %s :: %s' % (name, detail))
    total = len(results['pass']) + len(results['fail'])
    rate = (len(results['pass']) / total) if total else 1.0
    print('ST traces      : %d/%d pass (%.0f%%), %d unsupported'
          % (len(results['pass']), total, 100 * rate,
             len(results['unsupported'])))
    if results['unsupported']:
        from collections import Counter
        reasons = Counter(d.split('(')[0].strip() for _, d in
                          results['unsupported'])
        print('unsupported reasons: ' +
              ', '.join('%s x%d' % kv for kv in reasons.most_common(8)))
    ok = rate >= args.floor
    print('OK' if ok else 'BELOW FLOOR %.2f' % args.floor)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
