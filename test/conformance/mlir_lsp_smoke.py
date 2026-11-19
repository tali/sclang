#!/usr/bin/env python3
# Smoke test for `sclang-lsp-server` (the standard mlir-lsp-server with the
# scl/awl dialects registered, sclang-lsp-server/sclang-lsp-server.cpp).
# It serves *MLIR text* documents (.mlir) — for SCL/AWL source files use
# `sclang --lsp` (tested by lsp_smoke.py / lsp_semantic.py).
#
# Checks over stdio JSON-RPC:
#   1. initialize advertises hover/definition capabilities,
#   2. a valid scl-dialect .mlir document gets empty diagnostics,
#   3. a broken document gets diagnostics,
#   4. textDocument/hover over an scl.function name returns content,
#   5. textDocument/definition on a call resolves to the callee (Symbol
#      interfaces our ops implement).
#
# Usage: python3 test/conformance/mlir_lsp_smoke.py --sclang-lsp-server build.23/bin/sclang-lsp-server
import argparse
import sys

from lsp_client import Lsp

GOOD = """module {
  scl.function @callee(%arg0: !scl.int) -> !scl.int {
    %r = scl.tmpvar "r" : !scl.address<!scl.int>
    scl.store %r : !scl.address<!scl.int> = %arg0 : !scl.int
    scl.return.value %r : !scl.address<!scl.int>
  }
  scl.function @caller() -> !scl.int {
    %c = scl.constant 7 : i16 -> !scl.int
    %0 = scl.call.fc @callee(%c) {argNames = ["a"]} : (!scl.int) -> !scl.int
    %r = scl.tmpvar "r" : !scl.address<!scl.int>
    scl.store %r : !scl.address<!scl.int> = %0 : !scl.int
    scl.return.value %r : !scl.address<!scl.int>
  }
}
"""

BAD = "module {\n  scl.function @broken( -> !scl.int {\n}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang-lsp-server', required=True)
    args = ap.parse_args()
    lsp = Lsp(args.sclang_lsp_server, lsp_flag=False)
    r = lsp.initialize()
    caps = r["result"]["capabilities"]
    assert caps.get("hoverProvider") is True, caps
    assert caps.get("definitionProvider") is True, caps
    print("PASS initialize: hover+definition advertised")

    uri = "file:///tmp/mlir_lsp_good.mlir"
    d = lsp.open_doc(uri, GOOD)
    diags = d["params"]["diagnostics"]
    assert d["params"]["uri"] == uri and diags == [], d
    print("PASS valid scl-dialect document: no diagnostics")

    uri2 = "file:///tmp/mlir_lsp_bad.mlir"
    d = lsp.open_doc(uri2, BAD)
    assert d["params"]["diagnostics"], "expected diagnostics for broken mlir"
    print("PASS broken document: %d diagnostic(s)" % len(d["params"]["diagnostics"]))

    # hover over the callee name on line 1 (0-based), character 18 ("callee")
    r = lsp.request("textDocument/hover", {
        "textDocument": {"uri": uri},
        "position": {"line": 1, "character": 18}}, 10)
    assert "result" in r and r["result"], r
    contents = r["result"]["contents"]
    text = contents.get("value", "") if isinstance(contents, dict) else str(contents)
    assert "scl.function" in text, text
    print("PASS hover on scl.function:", text.splitlines()[0][:60])

    # definition of the call target on line 8 (`scl.call.fc @callee`)
    r = lsp.request("textDocument/definition", {
        "textDocument": {"uri": uri},
        "position": {"line": 8, "character": 24}}, 11)
    assert "result" in r and r["result"], r
    loc = r["result"][0] if isinstance(r["result"], list) else r["result"]
    assert loc["uri"] == uri and loc["range"]["start"]["line"] == 1, loc
    print("PASS definition: call @callee -> declaration on line 1")

    # references of @callee: the declaration (line 1) + the call use (line 8)
    r = lsp.request("textDocument/references", {
        "textDocument": {"uri": uri},
        "position": {"line": 1, "character": 18},
        "context": {"includeDeclaration": True}}, 12)
    refs = r.get("result") or []
    lines = sorted(x["range"]["start"]["line"] for x in refs)
    assert lines == [1, 8], refs
    print("PASS references: @callee used on lines", lines)

    lsp.shutdown()
    print("mlir_lsp_smoke: all PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
