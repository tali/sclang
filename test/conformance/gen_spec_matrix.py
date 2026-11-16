#!/usr/bin/env python3
# Spec-conformance matrix generator. Scans the sources for SPEC tags and
# regenerates the matrix section of docs/spec-conformance.md between the
# SPEC-MATRIX markers, so the documented conformance cannot silently drift
# from the code. See docs/spec-conformance.md ("How to trace") and AGENTS.md.
#
# Tag format (in any // comment):
#   // SPEC[AWL] IEC 61131-3 STL, load instruction (L)          -> full
#   // SPEC[AWL] Siemens AWL manual, timer instructions (approx) -> approx
#   // SPEC[SCL] IEC 61131-3 ST, CASE statement (gap)            -> gap
# The language tag is AWL/STL (IEC 61131-3 instruction list, Siemens AWL)
# or SCL/ST (IEC 61131-3 structured text, Siemens SCL). The status in the
# trailing parentheses is optional and defaults to "full":
# full = implemented per the reference, approx = implemented with a
# documented approximation, gap = known missing/unsupported.
#
# Usage: gen_spec_matrix.py [--check]
#   (default) rewrites docs/spec-conformance.md in place
#   --check   exits 1 if the checked-in file would change (for lit tests)

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DOC = os.path.join(ROOT, 'docs', 'spec-conformance.md')
BEGIN = '<!-- SPEC-MATRIX:BEGIN -->'
END = '<!-- SPEC-MATRIX:END -->'

# Sources scanned for SPEC tags, in display order. Paths relative to ROOT.
SCAN = [
    ('AWL frontend (mnemonic dispatch)', 'lib/AwlGen/AwlGen.cpp'),
    ('AWL dialect ops', 'include/sclang/AwlDialect/AwlOps.td'),
    ('AWL/SCL -> std lowering', 'lib/SclTransforms/LowerToStd.cpp'),
    ('AWL -> text (STL) emission', 'lib/AwlDialect/Printer.cpp'),
    ('SCL frontend', 'lib/SclGen/MLIRGen.cpp'),
    ('SCL -> AWL lowering', 'lib/SclTransforms/LowerToAWL.cpp'),
    ('simulation runtime', 'lib/Runtime/SimRuntime.cpp'),
    ('AWL lit tests', 'test/AwlGen'),
]

TAG = re.compile(r'//+\s*SPEC\[([A-Za-z]+)\]\s*(.+?)(?:\s*\((full|approx|gap)\))?\s*$')

# Known external conformance corpora groups (awlsim test suite layout), used
# to link the spec matrix to the executable test corpora.
CORPORA = [
    ('awlsim-test tc100_instructions', 'instruction set semantics'),
    ('awlsim-test tc150_startup', 'startup OBs / CPU features'),
    ('awlsim-test tc200_datatypes', 'data types: arrays, structs, strings, ANY/POINTER'),
    ('awlsim-test tc300_cpu', 'CPU behavior: recursion, status word'),
    ('awlsim-test tc999_projects', 'whole projects (shiftregister, ringbuffer, ...)'),
]


def scan_file(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for n, line in enumerate(f, 1):
            m = TAG.search(line)
            if m:
                lang = m.group(1).upper()
                ref = m.group(2).strip()
                status = m.group(3) or 'full'
                rows.append((lang, ref, status, n))
    return rows


def scan():
    """Return [(section, [(lang, ref, status, relpath, line)])]."""
    out = []
    for section, rel in SCAN:
        full = os.path.join(ROOT, rel)
        rows = []
        if os.path.isdir(full):
            for fn in sorted(os.listdir(full)):
                if fn.endswith(('.awl', '.mlir', '.scl')):
                    for lang, ref, status, n in scan_file(os.path.join(full, fn)):
                        rows.append((lang, ref, status, os.path.join(rel, fn), n))
        elif os.path.isfile(full):
            for lang, ref, status, n in scan_file(full):
                rows.append((lang, ref, status, rel, n))
        if rows:
            out.append((section, rows))
    return out


def link(rel, line):
    # GitHub-style relative source link from docs/ to the file:line
    return '[%s:%d](%s#L%d)' % (rel, line, '../' + rel, line)


def render(sections):
    lines = []
    lines.append('| language | spec reference | status | code |')
    lines.append('|---|---|---|---|')
    for section, rows in sections:
        for i, (lang, ref, status, rel, n) in enumerate(rows):
            sec = section if i == 0 else ''
            lines.append('| %s | %s | %s | %s |' % (lang, ref, status, link(rel, n)))
    return '\n'.join(lines)


def main():
    check = '--check' in sys.argv
    with open(DOC, encoding='utf-8') as f:
        doc = f.read()
    if BEGIN not in doc or END not in doc:
        sys.exit('docs/spec-conformance.md lacks the %s/%s markers' % (BEGIN, END))
    matrix = render(scan())
    head, rest = doc.split(BEGIN, 1)
    tail = rest.split(END, 1)[1] if END in rest else ''
    new = head + BEGIN + '\n\n' + matrix + '\n\n' + END + tail
    if check:
        if new == doc:
            print('spec matrix up to date')
            return 0
        print('spec matrix STALE: run test/conformance/gen_spec_matrix.py', file=sys.stderr)
        return 1
    with open(DOC, 'w', encoding='utf-8') as f:
        f.write(new)
    n = sum(len(rows) for _, rows in scan())
    print('wrote %s (%d SPEC tags)' % (os.path.relpath(DOC, ROOT), n))
    return 0


if __name__ == '__main__':
    sys.exit(main())
