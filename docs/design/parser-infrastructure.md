# Parser infrastructure: comparison and reuse analysis

Question: how does our hand-written recursive-descent parser compare to
state-of-the-art parser infrastructure, and can we reuse e.g. tree-sitter?

Status: analysis (no code change). Our parser stays hand-written; see the
verdict at the bottom.

## What we have

Hand-written lexer + recursive-descent parser (`include/sclang/SclGen/`),
header-only, produces our AST directly, zero dependencies. ~50ms per 100k
lines (no backtracking). Strict mode (abort at first error) for batch;
tolerant mode (`--parse-tolerant`, always on in `--lsp`) with panic-mode
recovery at unit/statement/declaration boundaries + `ErrorExprAST`
placeholders + a 100-error cap — multiple precise "expected X but has Y
(line,col)" diagnostics per pass. See AGENTS.md "Robustness"/"LSP mode".

## Comparison

| | ours (RD) | tree-sitter | Roslyn / rust-analyzer (rowan) | bison/GLR (`error` token) | PEG/packrat (pest) |
|---|---|---|---|---|---|
| grammar | implicit in code | declarative grammar file | implicit in code | declarative grammar file | declarative grammar file |
| incremental reparse | no (full reparse ~50ms/100k lines — fine at PLC sizes) | yes, green/red tree, O(edit) | yes, immutable trees | no | no (packrat memoizes, still full) |
| error tolerance | panic-mode sync + error nodes (new tolerant mode) | built-in: ERROR/MISSING nodes, always produces a tree | built-in error nodes | phrase-level via `error` productions | limited (PEG cut operators) |
| diagnostic quality | precise "expected X but has Y @ line:col" | ERROR node spans, coarser | good | decent | "expected one of ..." lists |
| deps/embedding | none | C lib (MIT), embeddable in C++ | C#/Rust only — not embeddable | C lib | Rust crate |
| speed | fast (no backtracking) | fast | fast | table-driven, fast | can degrade w/o memo |

## What the PLC ecosystem actually does (checked in the local corpora)

- PLC-lang-rusty (Rust): hand-written lexer (logos) + hand-written RD parser
  crates (`compiler/plc_lexer`, `plc_parser`) — same design as ours.
- awlsim (Python): hand-written parser (`awlsim/core/symbolparser.py` etc.).
- awltool (Ruby): hand-written parser (`lib/awltool/parser/`).
None use tree-sitter or any grammar framework in the compiler.

## Can we reuse tree-sitter?

Two different use cases:

1. **In the compiler** (replace/augment our parser): technically possible —
   tree-sitter is a small MIT C library, embeddable in our C++ — but it
   requires (a) a maintained grammar file: public grammars cover IEC
   61131-3 ST (gmahlen/tree-sitter-iec61131-3-st, tmatijevich/
   tree-sitter-structured-text) but NOT the Siemens SCL dialect (% directs,
   FC/FB/OB blocks, TIA attributes, S7_SetPoint...), so we would write and
   maintain our own grammar = a second source of truth next to the RD code;
   (b) a CST→AST bridge into MLIRGen (tree-sitter gives a concrete syntax
   tree, not our AST) or a rewrite of MLIRGen against the CST; (c) we would
   lose our precise diagnostics (tree-sitter reports ERROR/MISSING node
   spans, not "expected X but has Y"). The main benefit — incremental
   reparse — only pays off above ~100k lines per file (we measure ~50ms for
   a full 100k-line reparse, and per-keystroke work in the LSP is one
   document). Verdict: **not worth it for the compiler** at PLC file sizes.
2. **Editor-facing only**: publishing a tree-sitter grammar for SCL (start
   from an IEC ST grammar + Siemens dialect extensions) gives
   highlighting/folding/outline in Zed/Helix/Neovim for free, without the
   compiler depending on it (the grammar may stay approximate — it is only
   used for editor features). A TextMate grammar is the zero-code
   alternative for VS Code. This is the sweet spot if we want tree-sitter
   at all: **as a separate editor artifact, never wired into the compiler**.

## Verdict

Keep the hand-written RD parser in the compiler (it is fast, dependency-free,
produces our AST directly, and now has error-tolerant mode with precise
diagnostics). If editor integration beyond our LSP is wanted, add a
tree-sitter (or TextMate) grammar as a standalone artifact. Revisit only if
profiling shows >100k-line single files in the interactive loop — then
tree-sitter's incremental reparse (or a green-tree layer on our AST) would
be the thing to adopt, not the whole tree-sitter pipeline.
