#!/usr/bin/env python3
# Semantic-editing test for `sclang --lsp`: scripted editing sessions (not
# random mutations) that check diagnostics track the *semantics* of the
# document as it is being typed / refactored:
#
#   A. typing a POU incrementally:
#        incomplete unit            -> parse diagnostic (SCL code)
#        syntactically complete,
#        but variable undeclared    -> semantic diagnostic ("unknown
#                                      variable", MLIR code) at the use site
#        declaration added          -> diagnostics clear
#        two bad statements in two
#        different POUs             -> BOTH reported in a single pass
#                                      (tolerant MLIRGen)
#   B. renaming a variable with ranged (editor-style) didChange edits:
#        rename one use             -> one "unknown variable" at that use
#        rename only the decl      -> diagnostics at every remaining use
#        rename all occurrences in one multi-change didChange -> clean
#   C. after the document is clean again, didSave compiles the full pipeline
#      and JIT-runs it, returning the scan results.
#
# Usage: python3 test/conformance/lsp_semantic.py --sclang build.23/bin/sclang
import argparse
import sys

from lsp_client import Lsp

# Client-side mirror of each open document, so ranged edits can be computed
# the way an editor computes them (find the identifier on a line).
DOCS = {}
VERSIONS = {}


def bump(uri):
    VERSIONS[uri] = VERSIONS.get(uri, 0) + 1
    return VERSIONS[uri]


def open_doc(lsp, uri, text):
    DOCS[uri] = text
    return lsp.open_doc(uri, text)


def set_text(lsp, uri, text):
    DOCS[uri] = text
    return lsp.set_text(uri, bump(uri), text)


def edit_on_line(lsp, uri, line, old, new, occurrence=0):
    """Ranged didChange replacing the `occurrence`-th `old` on `line`
    (0-based line, 0-based character positions, like an editor rename)."""
    lines = DOCS[uri].split('\n')
    l = lines[line]
    start = -1
    for _ in range(occurrence + 1):
        start = l.index(old, start + 1)
    rng = {"start": {"line": line, "character": start},
           "end": {"line": line, "character": start + len(old)}}
    lsp.notify("textDocument/didChange",
               {"textDocument": {"uri": uri, "version": bump(uri)},
                "contentChanges": [{"range": rng, "text": new}]})
    lines[line] = l[:start] + new + l[start + len(old):]
    DOCS[uri] = '\n'.join(lines)
    return lsp.wait_notification("textDocument/publishDiagnostics")


def diags_of(msg):
    return msg["params"]["diagnostics"]


def summarize(diags):
    return sorted((d["range"]["start"]["line"], d.get("code", ""),
                   d["message"].split(":")[-1].strip()) for d in diags)


def check(cond, what, got):
    assert cond, "%s\n  got: %s" % (what, got)
    print("PASS", what)


def scenario_typing(lsp):
    print("-- scenario A: typing a POU incrementally")
    uri = "file:///tmp/lsp_sem_typing.scl"

    # A1: half-typed unit (no end_function) -> parse error, SCL code
    d = diags_of(open_doc(lsp, uri, "function f : int\nbegin\n  x := 1;\n"))
    check(len(d) >= 1 and all(str(x.get("code", "")).startswith("SCL") for x in d),
          "incomplete unit gives parse diagnostic(s) with SCL codes", summarize(d))

    # A2: structure complete, but x is not declared -> one semantic diagnostic
    # pointing at the use site (0-based line 2)
    d = diags_of(set_text(lsp, uri,
                          "function f : int\nbegin\n  x := 1;\nend_function\n"))
    check(len(d) == 1 and d[0]["range"]["start"]["line"] == 2
          and d[0].get("code") == "MLIR" and "unknown variable 'x'" in d[0]["message"],
          "undeclared variable -> MLIR diagnostic at the use site (line 2)",
          summarize(d))

    # A3: add the declaration -> clean
    d = diags_of(set_text(lsp, uri,
                          "function f : int\nvar\n  x : int;\nend_var\n"
                          "begin\n  x := 1;\nend_function\n"))
    check(d == [], "declaration added -> diagnostics cleared", summarize(d))

    # A4: two bad statements in two different POUs -> both reported at once
    d = diags_of(set_text(lsp, uri,
                          "function f : int\nvar\n  x : int;\nend_var\n"
                          "begin\n  x := 1;\n  x := y;\nend_function\n"
                          "function g : int\nbegin\n  g := z;\nend_function\n"))
    s = summarize(d)
    check(len(d) == 2 and s[0][0] == 6 and s[1][0] == 10
          and all("unknown variable" in m for _, _, m in s),
          "errors in two POUs reported in a single pass (lines 6 and 10)", s)


def scenario_rename(lsp):
    print("-- scenario B: renaming a variable via ranged edits")
    uri = "file:///tmp/lsp_sem_rename.scl"
    text = ("function f : int\n"
            "var\n"
            "  count : int;\n"          # line 2
            "end_var\n"
            "begin\n"
            "  count := count + 1;\n"   # line 5 (two uses)
            "  f := count;\n"           # line 6
            "end_function\n")
    d = diags_of(open_doc(lsp, uri, text))
    check(d == [], "initial document is clean", summarize(d))

    # rename only the last use (line 6) -> one diagnostic at that use
    d = diags_of(edit_on_line(lsp, uri, 6, "count", "cnt"))
    check(len(d) == 1 and d[0]["range"]["start"]["line"] == 6
          and "unknown variable 'cnt'" in d[0]["message"],
          "renaming one use -> 'unknown variable' at that use (line 6)",
          summarize(d))

    # rename only the declaration (line 2) -> the remaining `count` uses
    # error (one diagnostic per bad statement: line 5 fails at its LHS;
    # line 6 already says `cnt` and is fine again)
    d = diags_of(edit_on_line(lsp, uri, 2, "count", "cnt"))
    s = summarize(d)
    check(len(d) == 1 and s[0][0] == 5 and "unknown variable 'count'" in s[0][2],
          "renaming only the decl -> 'unknown variable' at the remaining use (line 5)", s)

    # rename both uses on line 5 in a single multi-change didChange -> clean
    lines = DOCS[uri].split('\n')
    l = lines[5]
    c1 = l.index("count")
    c2 = l.index("count", c1 + 1)
    changes = [{"range": {"start": {"line": 5, "character": c1},
                          "end": {"line": 5, "character": c1 + 5}}, "text": "cnt"},
               {"range": {"start": {"line": 5, "character": c2},
                          "end": {"line": 5, "character": c2 + 5}}, "text": "cnt"}]
    lsp.notify("textDocument/didChange",
               {"textDocument": {"uri": uri, "version": bump(uri)},
                "contentChanges": changes})
    lines[5] = l[:c1] + "cnt" + l[c1 + 5:c2] + "cnt" + l[c2 + 5:]
    DOCS[uri] = '\n'.join(lines)
    d = diags_of(lsp.wait_notification("textDocument/publishDiagnostics"))
    check(d == [], "all occurrences renamed -> diagnostics cleared", summarize(d))


def scenario_save_runs(lsp):
    print("-- scenario C: didSave JIT-runs the fixed document")
    uri = "file:///tmp/lsp_sem_save.scl"
    good = ("function OB1 : int\nbegin\n  %MW500 := %MW500 + 1;\nend_function\n"
            "function _mlir_ciface_OB1 : void\nbegin\n  OB1();\nend_function\n")
    d = diags_of(open_doc(lsp, uri, good))
    check(d == [], "runnable document is clean", summarize(d))
    lsp.notify("textDocument/didSave", {"textDocument": {"uri": uri}})
    m = lsp.wait_notification("sclang/scanResults")
    res = m["params"]
    out = res.get("output") or ""
    check(res.get("compileOk") is True and res.get("jitRc") == 0
          and "scan 0: mw=1" in out and "scan 1: mw=2" in out,
          "didSave ran the JIT scans", " | ".join(out.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    args = ap.parse_args()
    lsp = Lsp(args.sclang, ['--jit-cycles=2', '--scan-io=out:mw=M/500/2'])
    lsp.initialize()
    scenario_typing(lsp)
    scenario_rename(lsp)
    scenario_save_runs(lsp)
    lsp.shutdown()
    print("lsp_semantic: all PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
