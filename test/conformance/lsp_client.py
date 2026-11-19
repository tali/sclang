#!/usr/bin/env python3
# Shared JSON-RPC client for driving `sclang --lsp` (sclang/Lsp.cpp) from
# tests. Used by lsp_smoke.py and lsp_semantic.py.
#
# Usage: from lsp_client import Lsp; lsp = Lsp(sclang_path, extra_args=[...])
import json
import subprocess
import time


def frame(obj):
    body = json.dumps(obj).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


class Lsp:
    def __init__(self, sclang, extra_args=(), lsp_flag=True):
        # lsp_flag=False for servers that speak LSP without a flag
        # (e.g. sclang-lsp-server).
        args = [sclang] + (['--lsp'] if lsp_flag else []) + list(extra_args)
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)
        self.buf = b""

    def send(self, obj):
        self.p.stdin.write(frame(obj))
        self.p.stdin.flush()

    def recv(self, timeout=30):
        # read one framed message
        deadline = time.time() + timeout
        while b"\r\n\r\n" not in self.buf:
            if time.time() > deadline:
                raise TimeoutError("header")
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
            if time.time() > deadline:
                raise TimeoutError("body")
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
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.recv(timeout=timeout)
            if m.get("method") == method:
                return m
        raise TimeoutError(method)

    def initialize(self):
        r = self.request("initialize", {"processId": None, "rootUri": None,
                                        "capabilities": {}}, 1)
        assert "result" in r, r
        self.notify("initialized", {})
        return r

    def open_doc(self, uri, text, version=1):
        self.notify("textDocument/didOpen",
                    {"textDocument": {"uri": uri, "languageId": "scl",
                                      "version": version, "text": text}})
        return self.wait_notification("textDocument/publishDiagnostics")

    def set_text(self, uri, version, text):
        """Full-document didChange; returns the publishDiagnostics message."""
        self.notify("textDocument/didChange",
                    {"textDocument": {"uri": uri, "version": version},
                     "contentChanges": [{"text": text}]})
        return self.wait_notification("textDocument/publishDiagnostics")

    def shutdown(self):
        r = self.request("shutdown", None, 999)
        assert r.get("result") is None, r
        self.notify("exit", {})
        self.p.wait(timeout=10)
