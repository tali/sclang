//===----------------------------------------------------------------------===//
// Lsp.cpp - Minimal LSP server for sclang (JSON-RPC over stdio)
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `sclang --lsp` speaks a minimal LSP subset on stdin/stdout:
//
//   initialize / initialized / shutdown / exit
//   textDocument/didOpen, didChange, didSave, didClose
//   -> textDocument/publishDiagnostics   (parse + MLIRGen + verifier errors,
//                                           with 0-based LSP line/character)
//   didSave (and the custom `sclang/runScans` request)
//   -> runs the full lowering pipeline + JIT scan loop on the document and
//      sends window/logMessage plus a `sclang/scanResults` notification with
//      the per-scan output ("scan i: name=value ..." lines).
//
// Incrementality: the parser does ~100k lines in ~50ms, so per-change we
// simply re-parse the whole document in the persistent process (no
// tree-sitter-style green tree needed at SCL file sizes); the expensive
// part (lowering + JIT) runs only on save / explicit run.
//
// See AGENTS.md "LSP mode" and test/conformance/lsp_smoke.py.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace sclang {

// From sclang.cpp:
struct LspDiag {
  int line;
  int col;
  std::string message;
  bool warning;
  std::string code;
};
extern std::atomic<bool> g_lspShutdown;
int lspCompile(mlir::MLIRContext &ctx, llvm::StringRef text, llvm::StringRef name,
               bool fullPipeline, std::vector<LspDiag> &diags,
               mlir::OwningOpRef<mlir::ModuleOp> &module);
int lspRunJit(mlir::ModuleOp module, unsigned cycles, llvm::raw_ostream &out);
void registerSclangDialects(mlir::MLIRContext &context);

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

struct Document {
  std::string text;
  std::string name; // path without the file:// prefix
};

class LspServer {
public:
  explicit LspServer(unsigned jitCycles) : jitCycles(jitCycles), msg(nullptr) {
    // The context must know the scl/awl/std dialects or op creation in
    // mlirGen dereferences a null dialect (crashes in release builds).
    registerSclangDialects(context);
  }

  int run() {
    while (readMessage()) {
      if (!dispatch())
        break; // exit requested
    }
    return 0;
  }

private:
  unsigned jitCycles;
  mlir::MLIRContext context;
  std::map<std::string, Document> docs;

  // ---- JSON-RPC framing ----

  bool readMessage() {
    // Headers until an empty line, then Content-Length bytes of body.
    size_t contentLength = 0;
    std::string line;
    while (true) {
      line.clear();
      int c;
      while ((c = std::fgetc(stdin)) != EOF && c != '\n') {
        if (c != '\r')
          line += (char)c;
      }
      if (c == EOF && line.empty())
        return false;
      if (line.empty())
        break;
      if (line.rfind("Content-Length:", 0) == 0)
        contentLength = std::stoull(line.substr(15));
    }
    if (contentLength == 0)
      return false;
    body.assign(contentLength, '\0');
    size_t got = std::fread(body.data(), 1, contentLength, stdin);
    if (got != contentLength)
      return false;
    auto parsed = llvm::json::parse(body);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return true; // skip malformed message
    }
    msg = std::move(*parsed);
    return true;
  }

  void send(const Value &v) {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << v;
    os.flush();
    llvm::outs() << "Content-Length: " << out.size() << "\r\n\r\n" << out;
    llvm::outs().flush();
  }

  void respond(const Value &id, const Value &result) {
    Object r;
    r["jsonrpc"] = "2.0";
    r["id"] = id;
    r["result"] = result;
    send(Value(std::move(r)));
  }

  void respondError(const Value &id, int code, llvm::StringRef msg) {
    Object e;
    e["code"] = code;
    e["message"] = msg.str();
    Object r;
    r["jsonrpc"] = "2.0";
    r["id"] = id;
    r["error"] = Value(std::move(e));
    send(Value(std::move(r)));
  }

  void notify(llvm::StringRef method, const Value &params) {
    Object n;
    n["jsonrpc"] = "2.0";
    n["method"] = method.str();
    n["params"] = params;
    send(Value(std::move(n)));
  }

  // ---- dispatch ----

  bool dispatch() {
    auto *obj = msg.getAsObject();
    if (!obj)
      return true;
    llvm::StringRef method;
    if (auto m = obj->getString("method"))
      method = *m;
    Value id = Value(0); // requests carry an id; notifications don't
    bool isRequest = obj->get("id") != nullptr;
    if (isRequest)
      id = (*obj)["id"];
    const Object *params = obj->getObject("params");

    if (method == "initialize") {
      Object caps;
      caps["textDocumentSync"] = 1; // full-document sync
      Object result;
      result["capabilities"] = Value(std::move(caps));
      Object info;
      info["name"] = "sclang-lsp";
      info["version"] = "0.1";
      result["serverInfo"] = Value(std::move(info));
      respond(id, Value(std::move(result)));
    } else if (method == "shutdown") {
      respond(id, Value(nullptr));
    } else if (method == "exit") {
      g_lspShutdown = true; // cancel any in-flight parse
      std::exit(0);
    } else if (method == "textDocument/didOpen") {
      if (!params)
        return true;
      const Object *td = params->getObject("textDocument");
      if (!td)
        return true;
      std::string uri = td->getString("uri")->str();
      Document doc{td->getString("text")->str(), pathFromUri(uri)};
      docs[uri] = std::move(doc);
      publishDiagnostics(uri);
    } else if (method == "textDocument/didChange") {
      if (!params)
        return true;
      std::string uri = params->getObject("textDocument")->getString("uri")->str();
      auto it = docs.find(uri);
      if (it == docs.end())
        return true;
      if (auto *changes = params->getArray("contentChanges"))
        applyChanges(it->second.text, *changes);
      publishDiagnostics(uri);
    } else if (method == "textDocument/didSave") {
      if (params)
        if (auto *td = params->getObject("textDocument")) {
          std::string uri = td->getString("uri")->str();
          if (auto t = params->getString("text"))
            docs[uri].text = t->str();
          runScansAndNotify(uri);
        }
    } else if (method == "textDocument/didClose") {
      if (!params)
        return true;
      std::string uri = params->getObject("textDocument")->getString("uri")->str();
      docs.erase(uri);
      // Clear the squiggles for the closed document.
      Object p;
      p["uri"] = uri;
      p["diagnostics"] = Value(Array());
      notify("textDocument/publishDiagnostics", Value(std::move(p)));
    } else if (method == "sclang/runScans") {
      // Custom request: compile fully and JIT-run; respond with the output.
      std::string uri;
      std::string text;
      if (params)
        if (auto *td = params->getObject("textDocument")) {
          uri = td->getString("uri")->str();
          auto it = docs.find(uri);
          if (it != docs.end())
            text = it->second.text;
        }
      Object result = runScans(uri, text);
      respond(id, Value(std::move(result)));
    } else if (isRequest) {
      respondError(id, -32601, "method not found");
    }
    return true;
  }

  // ---- helpers ----

  static std::string pathFromUri(llvm::StringRef uri) {
    if (uri.starts_with("file://"))
      return uri.substr(7).str();
    return uri.str();
  }

  // Byte offset of an LSP (line, character) position. Positions are treated
  // as byte offsets within the line — fine for ASCII SCL/AWL.
  static size_t offsetOf(llvm::StringRef text, const Object *p) {
    long line = *p->getInteger("line");
    long col = *p->getInteger("character");
    size_t off = 0;
    for (long l = 0; l < line && off < text.size(); ++l) {
      size_t nl = text.find('\n', off);
      off = (nl == std::string::npos) ? text.size() : nl + 1;
    }
    return off + std::min<size_t>(col, text.size() - off);
  }

  // Apply a batch of contentChanges. Per the LSP spec a didChange may carry
  // several changes whose ranges all refer to the document state *before* the
  // batch (this is what editors send for multi-cursor edits / rename), so all
  // offsets are computed against the original text and applied back-to-front.
  // A change without a range replaces the whole document.
  static void applyChanges(std::string &text, const Array &changes) {
    struct Edit {
      size_t s, e;
      std::string ins;
    };
    llvm::SmallVector<Edit, 4> edits;
    for (const Value &ch : changes) {
      const Object *c = ch.getAsObject();
      if (!c)
        continue;
      auto newText = c->getString("text");
      if (!newText)
        continue;
      const Object *range = c->getObject("range");
      if (!range) {
        text = newText->str();
        return;
      }
      const Object *start = range->getObject("start");
      const Object *end = range->getObject("end");
      if (!start || !end)
        continue;
      size_t s = offsetOf(text, start), e = offsetOf(text, end);
      if (s > e)
        std::swap(s, e);
      edits.push_back({s, e, newText->str()});
    }
    llvm::sort(edits, [](const Edit &a, const Edit &b) { return a.s > b.s; });
    for (const Edit &ed : edits)
      text.replace(ed.s, ed.e - ed.s, ed.ins);
  }

  Value diagnosticsToJson(const std::vector<LspDiag> &diags) {
    Array arr;
    for (auto &d : diags) {
      Object range, spos, epos;
      spos["line"] = std::max(0, d.line - 1);
      spos["character"] = std::max(0, d.col - 1);
      epos["line"] = std::max(0, d.line - 1);
      epos["character"] = std::max(0, d.col - 1) + 1;
      range["start"] = Value(std::move(spos));
      range["end"] = Value(std::move(epos));
      Object diag;
      diag["range"] = Value(std::move(range));
      diag["severity"] = d.warning ? 2 : 1;
      diag["source"] = "sclang";
      diag["code"] = d.code;
      diag["message"] = d.message;
      arr.push_back(Value(std::move(diag)));
    }
    return Value(std::move(arr));
  }

  void publishDiagnostics(const std::string &uri) {
    auto it = docs.find(uri);
    if (it == docs.end())
      return;
    std::vector<LspDiag> diags;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    lspCompile(context, it->second.text, it->second.name, /*fullPipeline=*/false,
               diags, module);
    Object p;
    p["uri"] = uri;
    p["diagnostics"] = diagnosticsToJson(diags);
    notify("textDocument/publishDiagnostics", Value(std::move(p)));
  }

  Object runScans(const std::string &uri, const std::string &text) {
    std::vector<LspDiag> diags;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    Object result;
    result["uri"] = uri;
    if (text.empty()) {
      result["error"] = "document not open";
      return result;
    }
    int rc = lspCompile(context, text, pathFromUri(uri), /*fullPipeline=*/true,
                       diags, module);
    result["compileOk"] = (rc == 0 && module);
    if (rc != 0 || !module) {
      result["diagnostics"] = diagnosticsToJson(diags);
      return result;
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    int jrc = lspRunJit(*module, jitCycles, os);
    os.flush();
    result["jitRc"] = jrc;
    result["output"] = out;
    return result;
  }

  void runScansAndNotify(const std::string &uri) {
    auto it = docs.find(uri);
    if (it == docs.end())
      return;
    Object result = runScans(uri, it->second.text);
    notify("sclang/scanResults", Value(std::move(result)));
  }

  Value msg;
  std::string body;
};

} // namespace

int runLspServer(unsigned jitCycles) {
  LspServer server(jitCycles);
  return server.run();
}

} // namespace sclang
