#!/usr/bin/env python3
# Invalid-input (robustness) fuzzer: mutate the repo's own test corpus
# (.scl/.awl/.mlir) with byte flips, deletions, insertions, truncation and
# pure garbage, run the compiler front ends on the results, and assert the
# tools NEVER die by signal (segfault/trap/abort) or hang. Invalid input
# must produce a diagnostic and a clean non-zero exit — see AGENTS.md
# "Robustness against invalid input" and docs/testing.md.
#
# Usage:
#   python3 test/conformance/fuzz_invalid.py --sclang build.23/bin/sclang \
#       [--sclang-gen build.23/bin/sclang-gen] [--iters 500] [--seed 1]
#
# Exit 0 = no crashes; exit 1 = at least one signal/hang (prints the
# offending input path, which is kept under the tmp dir for reproduction).
import argparse
import glob
import os
import random
import subprocess
import sys
import tempfile

TIMEOUT_S = 20


def seed_files(repo):
    pats = ["test/sclang/*.scl", "test/AwlGen/*.awl", "test/SclGen/MLIR/*.scl",
            "test/SclGen/Parse/*.scl", "test/SclTransforms/*.mlir",
            "test/SclTransforms/*.scl"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(repo, p))
    return out


def mutate(rng, data):
    b = bytearray(data)
    for _ in range(rng.randint(1, 25)):
        if not b:
            break
        op = rng.random()
        pos = rng.randrange(len(b))
        if op < 0.35:
            b[pos] = rng.randrange(256)
        elif op < 0.65:
            del b[pos:pos + rng.randint(1, 40)]
        else:
            b[pos:pos] = bytes(rng.randrange(256)
                               for _ in range(rng.randint(1, 16)))
    if rng.random() < 0.35:
        b = b[:rng.randrange(len(b) + 1)]
    if rng.random() < 0.1:  # pure garbage
        b = bytearray(rng.randrange(256) for _ in range(rng.randint(0, 200)))
    return bytes(b)


def tool_args(sclang, sclang_gen, path, ext, tolerant=False):
    if ext == ".awl":
        return [sclang_gen, "-x=awl", "--emit=mlir-std", path]
    if ext == ".mlir":
        return [sclang_gen, "--emit=mlir-std", path]
    args = [sclang, "--emit=jit", "--jit-cycles=1"]
    if tolerant:
        args.append("--parse-tolerant")  # recovery paths must not crash/hang either
    return args + [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    ap.add_argument('--sclang-gen', default=None)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tolerant', action='store_true',
                    help='also fuzz with --parse-tolerant (error-recovery paths)')
    args = ap.parse_args()
    if not args.sclang_gen:
        args.sclang_gen = os.path.join(os.path.dirname(args.sclang),
                                      'sclang-gen')
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    seeds = seed_files(repo)
    if not seeds:
        print('no seed files found (run from the repo root layout)')
        return 1
    rng = random.Random(args.seed)
    crashes = []
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.iters):
            src = rng.choice(seeds)
            data = mutate(rng, open(src, 'rb').read())
            ext = os.path.splitext(src)[1]
            path = os.path.join(tmp, 'f%d%s' % (i, ext))
            open(path, 'wb').write(data)
            try:
                r = subprocess.run(tool_args(args.sclang, args.sclang_gen,
                                             path, ext, args.tolerant),
                                   capture_output=True, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                crashes.append((i, src, 'TIMEOUT', path))
                continue
            n += 1
            if r.returncode < 0:  # killed by a signal = real crash
                crashes.append((i, src, r.returncode, path))
    print('fuzz-invalid: %d mutations, %d crashes (seed %d%s)'
          % (n, len(crashes), args.seed, ', tolerant' if args.tolerant else ''))
    for i, src, rc, path in crashes[:10]:
        print('  CRASH iter=%d seed-file=%s rc=%s input-kept=%s'
              % (i, src, rc, path))
    return 1 if crashes else 0


if __name__ == '__main__':
    sys.exit(main())
