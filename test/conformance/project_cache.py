#!/usr/bin/env python3
# Test for the per-file build cache (--cache-dir, sclang/ProjectCache.cpp;
# docs/design/multi-file-incremental.md phase 2). Builds a synthetic 3-file
# project (FC + FB + DB in separate files, cross-file calls, a forward
# declaration) and checks:
#   1. first cached build: all misses, output identical to the uncached build
#   2. second cached build: all hits, output identical
#   3. editing one file: exactly one miss, JIT scan values reflect the edit,
#      output identical to uncached
#   4. a corrupted cache entry is treated as a miss and rebuilt correctly
#   5. a duplicate program unit across files is still a project error
#
# Usage: python3 test/conformance/project_cache.py --sclang build.23/bin/sclang
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

FC = """function double_it : int
var_input
  x : int;
end_var
begin
  double_it := x + 10;
end_function
"""

# forward declaration of add_one (empty body); real definition below
FWD = """function add_one : int
var_input
  x : int;
end_var
begin
end_function
"""

FB = """function_block counter
var_input
  inc : int;
end_var
var_output
  count : int;
end_var
begin
  count := count + inc;
end_function_block

data_block "GlobalDB"
var
  g : int;
end_var
begin
end_data_block

function add_one : int
var_input
  x : int;
end_var
begin
  add_one := x + 1;
end_function
"""

OB1 = """function OB1 : int
var
  c : counter;
end_var
begin
  %MW500 := double_it(%MW500 + 5);
  c(inc := add_one(%MW500));
  %MW502 := c.count;
end_function
function _mlir_ciface_OB1 : void
begin
  OB1();
end_function
"""


def run(sclang, args, timeout=60):
    p = subprocess.run([sclang] + args, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    return p.returncode, p.stdout.decode('utf-8', 'replace'), \
        p.stderr.decode('utf-8', 'replace')


def cache_stats(stderr):
    m = re.search(r'sclang cache: (\d+) hits, (\d+) misses \((\d+) files\)',
                  stderr)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='sclang-cache-test')
    try:
        fc = os.path.join(tmp, 'fc.scl')
        fb = os.path.join(tmp, 'fb.scl')
        ob = os.path.join(tmp, 'ob.scl')
        fwd = os.path.join(tmp, 'fwd.scl')
        for path, text in [(fc, FC), (fb, FB), (ob, OB1), (fwd, FWD)]:
            with open(path, 'w') as f:
                f.write(text)
        files = [fc, fb, ob, fwd]
        cache = os.path.join(tmp, 'cache')
        jit = ['--emit=jit', '--jit-cycles=3', '--scan-io=out:mw=M/500/2',
               '--scan-io=out:cnt=M/502/2']

        # uncached reference
        rc0, out0, _ = run(args.sclang, ['--emit=mlir'] + files)
        assert rc0 == 0, 'uncached build failed'
        rc0j, out0j, _ = run(args.sclang, jit + files)
        assert rc0j == 0, 'uncached jit failed: ' + out0j[-300:]

        # 1. first cached build: all misses, identical MLIR
        rc, out, err = run(args.sclang,
                           ['--cache-dir=' + cache, '--cache-stats',
                            '--emit=mlir'] + files)
        assert rc == 0, err
        hits, misses, total = cache_stats(err)
        assert (hits, misses, total) == (0, 4, 4), (hits, misses, total)
        assert out == out0, 'first cached build differs from uncached'
        print('PASS first build: 0 hits / 4 misses, output identical')

        # 2. second cached build: all hits, identical MLIR
        rc, out, err = run(args.sclang,
                           ['--cache-dir=' + cache, '--cache-stats',
                            '--emit=mlir'] + files)
        assert rc == 0, err
        hits, misses, total = cache_stats(err)
        assert (hits, misses, total) == (4, 0, 4), (hits, misses, total)
        assert out == out0, 'cached (all-hit) build differs from uncached'
        print('PASS second build: 4 hits / 0 misses, output identical')

        # 3. edit one file: one miss, JIT reflects the edit, MLIR identical
        with open(fc, 'w') as f:
            f.write(FC.replace('x + 10', 'x + 20'))
        rc, out, err = run(args.sclang,
                           ['--cache-dir=' + cache, '--cache-stats',
                            '--emit=mlir'] + files)
        assert rc == 0, err
        hits, misses, total = cache_stats(err)
        assert (hits, misses, total) == (3, 1, 4), (hits, misses, total)
        rc0, out0, _ = run(args.sclang, ['--emit=mlir'] + files)
        assert rc0 == 0 and out == out0, 'edited cached build differs'
        rc, outj, err = run(args.sclang,
                            ['--cache-dir=' + cache] + jit + files)
        assert rc == 0, err
        rc0j, out0j, _ = run(args.sclang, jit + files)
        assert rc0j == 0 and outj == out0j, \
            'cached JIT differs from uncached JIT:\n%s\n%s' % (outj, out0j)
        assert 'scan 0: mw=25' in outj, outj  # 0+5+20
        print('PASS after edit: 3 hits / 1 miss, JIT values updated:',
              ' '.join(outj.split()))

        # 4. corrupt all cache entries: treated as misses, output still correct
        entries = [os.path.join(cache, e) for e in os.listdir(cache)
                   if e.endswith('.mlir')]
        assert len(entries) >= 4, entries  # 4 files + 1 edited version
        for e in entries:
            with open(e, 'w') as f:
                f.write('garbage {{{')
        rc, out, err = run(args.sclang,
                           ['--cache-dir=' + cache, '--cache-stats',
                            '--emit=mlir'] + files)
        assert rc == 0, err
        hits, misses, total = cache_stats(err)
        assert (hits, misses, total) == (0, 4, 4), (hits, misses, total)
        rc0, out0, _ = run(args.sclang, ['--emit=mlir'] + files)
        assert out == out0, 'corrupted-entry rebuild differs from uncached'
        print('PASS corrupted entries rebuilt (4 misses), output identical')

        # 5. duplicate program unit across files is still an error
        with open(os.path.join(tmp, 'dup.scl'), 'w') as f:
            f.write(FC)
        rc, out, err = run(args.sclang,
                           ['--cache-dir=' + cache, '--emit=mlir',
                            fc, os.path.join(tmp, 'dup.scl')])
        assert rc == 6 and 'duplicate program unit' in err, (rc, err[:200])
        print('PASS duplicate unit across files diagnosed')

        print('project_cache: all PASS')
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
