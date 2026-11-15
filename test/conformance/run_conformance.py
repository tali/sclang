#!/usr/bin/env python3
# Conformance runner: compiles publicly available SCL/AWL sources (external
# corpora, see docs/conformance.md) with sclang and reports pass rates.
#
# Usage:
#   run_conformance.py --sclang <path-to-sclang> [--corpora <dir>] [--dump <file>]
#
# Corpora root defaults to $SCLANG_CONFORMANCE_CORPORA or
# /Users/martin/scl-external. If the root does not exist the script prints
# SKIP and exits 0 (the lit test then passes without testing anything).
# Otherwise it exits 1 when a pass rate drops below the floor constants
# below (a regression), so the lit test catches regressions when the
# corpora are present.
#
# Two tiers:
#  - compile tier: --emit=mlir (frontend + MLIR verifier) on every file;
#  - JIT tier: --emit=jit --jit-cycles=2 (full lowering + ExecutionEngine
#    smoke run) on files defining OB 1 — checks the code also lowers and
#    executes without crashing/missing symbols, not computed values.

import argparse
import os
import re
import subprocess
import sys

# Minimum pass rates (fractions) when corpora are present. Current state
# (LLVM 17 build): compile tier AWL 109/116 = 0.94, SCL single 13/22 = 0.59,
# SCL per-project 11/17 = 0.65; JIT tier (OB1 files) 69/77 = 0.90.
# Remaining failures are documented gaps (awl shift/rotate + real math ops;
# TIA library types in SCL; a few awlsim files with exotic syntax), see
# docs/conformance.md.
FLOOR_AWL = 0.90
FLOOR_SCL_SINGLE = 0.55
FLOOR_SCL_PROJECT = 0.60
FLOOR_JIT = 0.85

TIMEOUT_S = 20

# awlsim project files use quoted block names / ORGANIZATION_BLOCK etc.;
# SCL exports use FUNCTION / unquoted DATA_BLOCK / VAR_INPUT etc.
AWL_RE = re.compile(
    r'ORGANIZATION_BLOCK|^[ \t]*DATA_BLOCK|^[ \t]*(OB|FB|FC)[ \t]+[0-9"]'
    r'|^[ \t]*TYPE[ \t]+"', re.M)
SCL_RE = re.compile(
    r'^[ \t]*(FUNCTION|FUNCTION_BLOCK)[ \t]|^[ \t]*VAR_(INPUT|OUTPUT|IN_OUT)[ \t]'
    r'|^[ \t]*DATA_BLOCK[ \t]+[^"]|^[ \t]*TYPE[ \t]+[^"]', re.M)
# files defining OB 1 can be JIT-run (the driver invokes _mlir_ciface_OB1)
OB1_RE = re.compile(r'^[ \t]*(ORGANIZATION_BLOCK[ \t]+)?OB[ \t]+1([ \t]|$)', re.M)


def sniff(text):
    if AWL_RE.search(text):
        return '-x=awl'
    if SCL_RE.search(text):
        return '-x=scl'
    return '-x=awl'


def run_one(sclang, path, xflag, emit='mlir', extra=()):
    try:
        p = subprocess.run([sclang, xflag, '--emit=' + emit, *extra, path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=TIMEOUT_S)
        return p.returncode == 0, p.stderr.decode('utf-8', 'replace').splitlines()
    except subprocess.TimeoutExpired:
        return False, ['TIMEOUT (possible intra-block jump loop)']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    ap.add_argument('--corpora',
                    default=os.environ.get('SCLANG_CONFORMANCE_CORPORA',
                                           '/Users/martin/scl-external'))
    ap.add_argument('--dump', default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.corpora):
        print('SKIP: corpora root %s not found '
              '(set SCLANG_CONFORMANCE_CORPORA)' % args.corpora)
        return 0

    scl_files = []
    awl_files = []
    for root, dirs, files in os.walk(args.corpora):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.endswith('.scl') and f != 'makefile.scl':
                scl_files.append(p)
            elif f.endswith('.awl'):
                awl_files.append(p)

    lines = []

    def record(ok, path, xflag, err):
        if not ok:
            reason = err[0] if err else '?'
            lines.append('FAIL(%s) %s :: %s' % (xflag, path, reason[:150]))
        return ok

    # --- compile tier, AWL (and awlsim files that are actually SCL) ---
    awl_ok = 0
    for p in awl_files:
        with open(p, 'r', errors='replace') as fh:
            xflag = sniff(fh.read())
        ok, err = run_one(args.sclang, p, xflag)
        awl_ok += record(ok, p, xflag, err)
    awl_total = len(awl_files)

    # --- compile tier, SCL single files ---
    scl_ok = 0
    for p in scl_files:
        ok, err = run_one(args.sclang, p, '-x=scl')
        scl_ok += record(ok, p, '-x=scl', err)
    scl_total = len(scl_files)

    # --- JIT tier: full pipeline (lowering + ExecutionEngine) on files
    # that define OB 1; smoke test (no crash, no missing symbol), not an
    # oracle check of computed values ---
    jit_ok = 0
    jit_total = 0
    for p in awl_files + scl_files:
        with open(p, 'r', errors='replace') as fh:
            text = fh.read()
        if not OB1_RE.search(text):
            continue
        xflag = sniff(text)
        jit_total += 1
        ok, err = run_one(args.sclang, p, xflag, emit='jit',
                          extra=('--jit-cycles=2',))
        jit_ok += record(ok, p, xflag + ' jit', err)

    # --- compile tier, SCL per-project: concatenate all .scl files of a dir ---
    dirs = sorted(set(os.path.dirname(p) for p in scl_files))
    proj_ok = 0
    proj_total = 0
    for d in dirs:
        files = sorted(p for p in scl_files if os.path.dirname(p) == d)
        if not files:
            continue
        proj_total += 1
        tmp = os.path.join('/tmp', 'sclang-conformance-proj.scl')
        with open(tmp, 'w') as out:
            for i, p in enumerate(files):
                if i:
                    out.write('\n')
                with open(p, 'r', errors='replace') as fh:
                    out.write(fh.read())
        ok, err = run_one(args.sclang, tmp, '-x=scl')
        proj_ok += record(ok, d + ' (project)', '-x=scl', err)

    def rate(ok, total):
        return ok / total if total else 1.0

    print('AWL            : %d/%d pass (%.0f%%)' %
          (awl_ok, awl_total, 100 * rate(awl_ok, awl_total)))
    print('SCL single     : %d/%d pass (%.0f%%)' %
          (scl_ok, scl_total, 100 * rate(scl_ok, scl_total)))
    print('SCL projects   : %d/%d pass (%.0f%%)' %
          (proj_ok, proj_total, 100 * rate(proj_ok, proj_total)))
    print('JIT (OB1 files): %d/%d pass (%.0f%%)' %
          (jit_ok, jit_total, 100 * rate(jit_ok, jit_total)))
    if lines:
        print('--- failures ---')
        for l in lines:
            print(l)
    if args.dump:
        with open(args.dump, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')

    bad = (rate(awl_ok, awl_total) < FLOOR_AWL or
           rate(scl_ok, scl_total) < FLOOR_SCL_SINGLE or
           rate(proj_ok, proj_total) < FLOOR_SCL_PROJECT or
           rate(jit_ok, jit_total) < FLOOR_JIT)
    if bad:
        print('REGRESSION: pass rate below floor '
              '(awl>=%.2f scl>=%.2f proj>=%.2f jit>=%.2f)' %
              (FLOOR_AWL, FLOOR_SCL_SINGLE, FLOOR_SCL_PROJECT, FLOOR_JIT))
        return 1
    print('OK (floors: awl>=%.2f scl>=%.2f proj>=%.2f jit>=%.2f)' %
          (FLOOR_AWL, FLOOR_SCL_SINGLE, FLOOR_SCL_PROJECT, FLOOR_JIT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
