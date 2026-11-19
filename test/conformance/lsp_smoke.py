#!/usr/bin/env python3
# Smoke test for `sclang --lsp` (the minimal LSP server in sclang/Lsp.cpp):
# drives it over stdio with JSON-RPC and checks
#   1. a document with a syntax error gets publishDiagnostics with the
#      right 0-based line,
#   2. fixing the document via didChange clears the diagnostics,
#   3. didSave on a runnable OB1 document runs the JIT scans and returns
#      `sclang/scanResults` with the scan lines,
#   4. an incremental (ranged) didChange is applied correctly.
#
# Usage: python3 test/conformance/lsp_smoke.py --sclang build.23/bin/sclang
import argparse
import json
import subprocess
import sys


def frame(obj):
    body = json.dumps(obj).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


class Lsp:
    def __init__(self, sclang):
        # --scan-io makes the JIT print `scan N: mw=value` per cycle so the
        # test can check the *executed* result, not just the exit code.
        self.p = subprocess.Popen([sclang, '--lsp', '--jit-cycles=2',
                                   '--scan-io=out:mw=M/500/2'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.buf = b""

    def send(self, obj):
        self.p.stdin.write(frame(obj))
        self.p.stdin.flush()

    def recv(self, timeout=30):
        # read one framed message
        while b"\r\n\r\n" not in self.buf:
            chunk = self.p.stdout.read1(65536)
            if not chunk:
                raise EOFError("server closed stdout")
            self.buf += chunk
        head, rest = self.buf.split(b"\r\n\r\n", 1)
        length = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        while len(rest) < length:
            chunk = self.p.stdout.read1(65536)
            if not chunk:
                raise EOFError("server closed stdout mid-body")
            rest += chunk
        self.buf = rest[length:]
        return json.loads(rest[:length])

    def request(self, method, params, id):
        self.send({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
        while True:
            m = self.recv()
            if m.get("id") == id:
                return m

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def wait_notification(self, method, timeout=30):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.recv()
            if m.get("method") == method:
                return m
        raise TimeoutError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sclang', required=True)
    args = ap.parse_args()
    lsp = Lsp(args.sclang)
    r = lsp.request("initialize", {"processId": None, "rootUri": None,
                                   "capabilities": {}}, 1)
    assert "result" in r and r["result"]["capabilities"]["textDocumentSync"] == 1, r
    lsp.notify("initialized", {})

    uri = "file:///tmp/lsp_smoke.scl"
    bad = "function f : int\nvar\n  x : int;\nend_var\nbegin\n  x := ;\nend_function\n"
    lsp.notify("textDocument/didOpen",
               {"textDocument": {"uri": uri, "languageId": "scl", "version": 1,
                                 "text": bad}})
    d = lsp.wait_notification("textDocument/publishDiagnostics")
    diags = d["params"]["diagnostics"]
    assert diags, "expected a diagnostic for the syntax error"
    # error is on line 6 (`x := ;`), 0-based line 5
    assert diags[0]["range"]["start"]["line"] == 5, diags
    assert "expected 'expression'" in diags[0]["message"], diags
    print("PASS syntax error diagnostic: line", diags[0]["range"]["start"]["line"],
          "-", diags[0]["message"][:60])

    # incremental change: fix line 6 by replacing `;` after `x := ` with `1`
    lsp.notify("textDocument/didChange", {
        "textDocument": {"uri": uri, "version": 2},
        "contentChanges": [{"range": {"start": {"line": 5, "character": 7},
                                      "end": {"line": 5, "character": 8}},
                            "text": "1;"}]})
    d = lsp.wait_notification("textDocument/publishDiagnostics")
    assert d["params"]["diagnostics"] == [], d
    print("PASS didChange cleared diagnostics")

    # full-text change to a runnable program, then didSave runs the JIT.
    # %MW500 addresses M byte 500 in this compiler, matching the scan slot.
    good = ("function OB1 : int\nbegin\n  %MW500 := %MW500 + 1;\nend_function\n"
            "function _mlir_ciface_OB1 : void\nbegin\n  OB1();\nend_function\n")
    lsp.notify("textDocument/didChange", {
        "textDocument": {"uri": uri, "version": 3},
        "contentChanges": [{"text": good}]})
    d = lsp.wait_notification("textDocument/publishDiagnostics")
    assert d["params"]["diagnostics"] == [], d
    lsp.notify("textDocument/didSave", {"textDocument": {"uri": uri}})
    m = lsp.wait_notification("sclang/scanResults")
    res = m["params"]
    assert res.get("compileOk") is True, res
    assert res.get("jitRc") == 0, res
    out = res.get("output") or ""
    assert "scan 0: mw=1" in out and "scan 1: mw=2" in out, out
    print("PASS didSave JIT run:", " | ".join(out.split()))

    # error-tolerant parsing in LSP mode: a document with several errors on
    # different lines gets one diagnostic per error (not just the first)
    uri2 = "file:///tmp/lsp_smoke2.scl"
    multi_err = ("function f1 : int\nvar\n  x : int;\nend_var\nbegin\n"
                 "  x := ;\n  x := 1 + ;\nend_function\n"
                 "123 garbage\n"
                 "function f3 : int\nbegin\n  f3 := 1;\nend_function\n")
    lsp.notify("textDocument/didOpen",
               {"textDocument": {"uri": uri2, "languageId": "scl", "version": 1,
                                 "text": multi_err}})
    d = lsp.wait_notification("textDocument/publishDiagnostics")
    assert d["params"]["uri"] == uri2, d
    diags = d["params"]["diagnostics"]
    lines = sorted(x["range"]["start"]["line"] for x in diags)
    assert len(diags) >= 3 and lines == [5, 6, 8], diags  # 0-based: 6,7,9
    print("PASS tolerant multi-error diagnostics: lines", lines)

    r = lsp.request("shutdown", None, 2)
    assert r.get("result") is None, r
    lsp.notify("exit", {})
    lsp.p.wait(timeout=10)
    print("lsp_smoke: all PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
