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
# Three tiers:
#  - compile tier: --emit=mlir (frontend + MLIR verifier) on every file;
#  - JIT smoke tier: --emit=jit --jit-cycles=2 (full lowering +
#    ExecutionEngine run) on files defining OB 1 — checks the code also
#    lowers and executes without crashing/missing symbols, not computed
#    values. Exit code 2 (awlsim __ASSERT oracles failed) still counts as
#    smoke-pass: it proves lowering + execution worked;
#  - oracle tier: --emit=jit --jit-cycles=1 --cycle-ms=0 on files defining OB 1 that
#    contain awlsim __ASSERT directives — pass iff exit code is 0, i.e.
#    every executed __ASSERT oracle held. This checks computed values.

import argparse
import os
import re
import subprocess
import sys

# Minimum pass rates (fractions) when corpora are present. Current state
# (LLVM 17 build): compile tier AWL 116/116 = 1.00, SCL single 15/15 = 1.00,
# SCL per-project 17/17 = 1.00; JIT smoke tier (OB1 files) 102/105 = 0.97;
# oracle tier (__ASSERT + OB1/bare-instruction files) 54/71 = 0.76. Known-open
# oracle failures: startup_multiple (2-accu CPU), insn_ARITH_R, insn_SE/SS
# (timer oracles need real time), insn_CALL, sfb0-5, tc200 array/pointer/udt,
# shutdown. Remaining spec deviations are the
# approx rows of the SPEC matrix, see docs/conformance.md and
# docs/spec-conformance.md.
FLOOR_AWL = 0.90
FLOOR_SCL_SINGLE = 0.55
FLOOR_SCL_PROJECT = 0.60
FLOOR_JIT = 0.85
FLOOR_ORACLE = 0.15

TIMEOUT_S = 20

# awlsim project files use quoted block names / ORGANIZATION_BLOCK etc.;
# SCL exports use FUNCTION / unquoted DATA_BLOCK / VAR_INPUT etc.
AWL_RE = re.compile(
    r'ORGANIZATION_BLOCK|^[ \t]*DATA_BLOCK|^[ \t]*(OB|FB|FC)[ \t]+[0-9"]'
    r'|^[ \t]*TYPE[ \t]+"', re.M)
SCL_RE = re.compile(
    r'^[ \t]*(FUNCTION|FUNCTION_BLOCK)[ \t]|^[ \t]*VAR_(INPUT|OUTPUT|IN_OUT)[ \t]'
    r'|^[ \t]*DATA_BLOCK[ \t]+[^"]|^[ \t]*TYPE[ \t]+[^"]', re.M)
# files defining OB 1 can be JIT-run (the driver invokes _mlir_ciface_OB1).
# Bare instruction lists (no block header at all) are wrapped in an implicit
# OB 1 by the AWL frontend, exactly like awlsim's own test runner does, so
# they are JIT-runnable too.
OB1_RE = re.compile(r'^[ \t]*(ORGANIZATION_BLOCK[ \t]+)?OB[ \t]+1([ \t]|$)', re.M)
BLOCK_HDR_RE = re.compile(
    r'^[ \t]*(ORGANIZATION_BLOCK|DATA_BLOCK|TYPE|VAR[ \t]|BEGIN|END_VAR'
    r'|OB|FB|FC|FUNCTION_BLOCK|FUNCTION)[ \t]', re.M)


def has_ob1(text):
    if OB1_RE.search(text):
        return True
    # bare instruction list (awlsim wraps these in OB 1 implicitly)
    return not BLOCK_HDR_RE.search(text)


def read_text(p):
    with open(p, 'r', errors='replace') as fh:
        return fh.read().lstrip('\ufeff')


def sniff(text):
    if AWL_RE.search(text):
        return '-x=awl'
    if SCL_RE.search(text):
        return '-x=scl'
    return '-x=awl'


# awlsim project symbol table (.awlpro "Symbol table" CDATA):
#   126,Data                    DB 42       DB 42 ...
# -> --db-sym Data=42 so P#"Data".VAR resolves without the project file.
AWLSYM_RE = re.compile(r'^\s*\d+,([A-Za-z_][\w]*)\s+(?:DB|DI)\s+(\d+)\s', re.M)


def awlpro_db_syms(path):
    import base64
    pro = path[:-4] + '.awlpro' if path.endswith('.awl') else None
    if not pro or not os.path.isfile(pro):
        return ()
    text = read_text(pro)
    chunks = [text]
    # legacy fileformat-v0: INI with base64-encoded symbol tables
    # (sym_tab_0=<base64 of "126,Data   DB 42 ...">)
    for m in re.finditer(r'^sym_tab_\d+=(.*)$', text, re.M):
        try:
            chunks.append(base64.b64decode(m.group(1).strip()).decode(
                'utf-8', 'replace'))
        except Exception:
            pass
    syms = []
    for chunk in chunks:
        for m in AWLSYM_RE.finditer(chunk):
            syms.append('--db-sym=%s=%s' % (m.group(1), m.group(2)))
    return tuple(syms)


# Returns (exit_code, stderr_lines). 0 = success, 2 = executed but __ASSERT
# oracles failed (oracle tier), anything else = compile/lower/run failure.
def run_one(sclang, path, xflag, emit='mlir', extra=()):
    extra = tuple(extra) + awlpro_db_syms(path)
    try:
        p = subprocess.run([sclang, xflag, '--emit=' + emit, *extra, path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=TIMEOUT_S)
        return p.returncode, p.stderr.decode('utf-8', 'replace').splitlines()
    except subprocess.TimeoutExpired:
        return -1, ['TIMEOUT (possible intra-block jump loop)']


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

    # Files of a multi-file project (more than one .scl in the directory)
    # reference symbols from their sibling files, so they cannot compile
    # standalone in any compiler (TIA neither); they are covered by the
    # per-project tier below, so skip them in the single-file tier (but keep
    # them for the project tier).
    all_scl_files = scl_files
    scl_dir_counts = {}
    for p in scl_files:
        d = os.path.dirname(p)
        scl_dir_counts[d] = scl_dir_counts.get(d, 0) + 1
    scl_files = [p for p in scl_files if scl_dir_counts[os.path.dirname(p)] == 1]

    lines = []

    def record(ok, path, xflag, err):
        if not ok:
            reason = err[0] if err else '?'
            lines.append('FAIL(%s) %s :: %s' % (xflag, path, reason[:150]))
        return ok

    # --- compile tier, AWL (and awlsim files that are actually SCL) ---
    awl_ok = 0
    for p in awl_files:
        xflag = sniff(read_text(p))
        rc, err = run_one(args.sclang, p, xflag)
        awl_ok += record(rc == 0, p, xflag, err)
    awl_total = len(awl_files)

    # --- compile tier, SCL single files ---
    scl_ok = 0
    for p in scl_files:
        rc, err = run_one(args.sclang, p, '-x=scl')
        scl_ok += record(rc == 0, p, '-x=scl', err)
    scl_total = len(scl_files)

    # --- JIT smoke tier: full pipeline (lowering + ExecutionEngine) on
    # files that define OB 1; smoke test (no crash, no missing symbol),
    # not an oracle check of computed values. rc 2 means the code ran but
    # __ASSERT oracles failed — still a smoke pass ---
    jit_ok = 0
    jit_total = 0
    oracle_ok = 0
    oracle_total = 0
    for p in awl_files + scl_files:
        text = read_text(p)
        if not has_ob1(text):
            continue
        # .scl files are always SCL (sniff() may misread unquoted DATA_BLOCK
        # as AWL); .awl files are sniffed per content
        xflag = '-x=scl' if p.endswith('.scl') else sniff(text)
        # only JIT-run files that compiled standalone (cross-file references
        # are covered by the project tier)
        rc0, _ = run_one(args.sclang, p, xflag)
        if rc0 != 0:
            continue
        jit_total += 1
        rc, err = run_one(args.sclang, p, xflag, emit='jit',
                          extra=('--jit-cycles=2',))
        jit_ok += record(rc in (0, 2), p, xflag + ' jit', err)
        # --- oracle tier: awlsim __ASSERT directives as value-level
        # checks; pass iff every executed oracle held (rc == 0) ---
        if '__ASSERT' in text:
            oracle_total += 1
            rc, err = run_one(args.sclang, p, xflag, emit='jit',
                              extra=('--jit-cycles=1', '--cycle-ms=0'))
            oracle_ok += record(rc == 0, p, xflag + ' oracle', err)

    # --- compile tier, SCL per-project: concatenate all .scl files of a dir ---
    dirs = sorted(set(os.path.dirname(p) for p in all_scl_files))
    proj_ok = 0
    proj_total = 0
    for d in dirs:
        files = sorted(p for p in all_scl_files if os.path.dirname(p) == d)
        if not files:
            continue
        proj_total += 1
        tmp = os.path.join('/tmp', 'sclang-conformance-proj.scl')
        with open(tmp, 'w') as out:
            for i, p in enumerate(files):
                if i:
                    out.write('\n')
                out.write(read_text(p))
        rc, err = run_one(args.sclang, tmp, '-x=scl')
        proj_ok += record(rc == 0, d + ' (project)', '-x=scl', err)

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
    print('Oracle (__ASSERT): %d/%d pass (%.0f%%)' %
          (oracle_ok, oracle_total, 100 * rate(oracle_ok, oracle_total)))
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
           rate(jit_ok, jit_total) < FLOOR_JIT or
           rate(oracle_ok, oracle_total) < FLOOR_ORACLE)
    if bad:
        print('REGRESSION: pass rate below floor '
              '(awl>=%.2f scl>=%.2f proj>=%.2f jit>=%.2f oracle>=%.2f)' %
              (FLOOR_AWL, FLOOR_SCL_SINGLE, FLOOR_SCL_PROJECT, FLOOR_JIT,
               FLOOR_ORACLE))
        return 1
    print('OK (floors: awl>=%.2f scl>=%.2f proj>=%.2f jit>=%.2f oracle>=%.2f)' %
          (FLOOR_AWL, FLOOR_SCL_SINGLE, FLOOR_SCL_PROJECT, FLOOR_JIT,
           FLOOR_ORACLE))
    return 0


if __name__ == '__main__':
    sys.exit(main())
