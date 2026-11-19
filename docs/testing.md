# Testing methodology

The compiler is verified by layers, from syntax to executed values. Each
layer exists because the layer below it cannot catch its failure class:
compile-only tests cannot catch wrong values, and JIT tests cannot tell you
*which* lowering stage is wrong.

## Tiers

| Tier | Runner | What it checks |
|------|--------|----------------|
| lit (FileCheck) | `build.23/bin/llvm-lit test` (131 tests) | IR shape per pass (`--emit=mlir`, `mlir-std`, `awl-mlir`), AWL text output, and JIT-executed values via `--scan-io`/`--scan-inputs` scan lines and `--dump-areas` |
| AWL/SCL compile corpora | `run_conformance.py` compile tiers | every public `.awl`/`.scl` corpus file parses and lowers |
| JIT smoke | `run_conformance.py` jit tier | corpus files with `OB1` run under the JIT without crashing |
| Oracle vs awlsim | `run_conformance.py` oracle tier | `__ASSERT(...)` lines in awlsim-test `.awl` files: our sim runtime must agree with awlsim's own expectations |
| ST trace conformance | `run_st_traces.py` (92 IEC 61131-3 `.st`/`.trace` pairs) | value-level: per-scan outputs compared to the published trace |
| Differential cross | same, `--cross` (default) | each trace wrapper run twice: `--emit=jit` vs `--emit=jit-awl` (in-memory awl-dialect lowering); scan lines must be identical |
| Fuzz | `fuzz_st.py` | seeded random per-scan inputs; SCL->JIT vs SCL->jit-awl must agree |

Run everything:

```
cd build.23 && ninja sclang sclang-gen && ./bin/llvm-lit ../test
python3 test/conformance/run_conformance.py --sclang build.23/bin/sclang
python3 test/conformance/run_st_traces.py --sclang build.23/bin/sclang --cross   # add --jobs N to parallelize (default min(8, cores); 10s -> 3.5s)
python3 test/conformance/fuzz_st.py --sclang build.23/bin/sclang --scans 5 --seed 7
```

## Why the differential cross exists

The SCL->AWL lowering was originally only verifiable through the AWL *text*
round-trip (`sclang-gen --emit=awl` | `-x=awl --emit=jit`). That path hides
lowering bugs behind the printer/parser: the text printer models the S7
accumulator stack (T pops ACCU1 in the printer model, while awlsim's T does
not pop), so print+reparse can "fix" or mask dialect-level mistakes. The
primary path is now the in-memory one (`--emit=jit-awl`): LowerToAWL produces
awl-dialect ops that go straight through LowerToStd/LLVM/JIT. The text
round-trip remains available as `--emit=awl` (export) and `--cross-text`
(opt-in extra check) but is no longer the verification vehicle.

Cross-skips (not failures): constructs the AWL backend cannot represent
(strings, LREAL, date/time, FB builtins in SCL blocks, VAR initializers,
implementation-defined traces). The skip is decided by running only the
`--emit=awl-mlir` lowering first; if that fails or reports
'unsupported'/'not representable'/'unknown FB member', the test is skipped.

## Why the edge-case lit tests exist (scenario rationale)

`test/sclang/jit_edge_*.scl` pin value-level behavior in the areas where the
lowering was historically wrong. Each scenario targets a found bug class:

- **Operand temps across control flow** (`jit_awl_direct.scl`, plus the
  `control-flow/*` traces): computed values are spilled to M-area temps at
  their definition and operand temps are re-stored at *every* use, because a
  spill store inside a loop body that runs zero times must not leave a stale
  temp for a later use (this was the `for_exit` divergence: a CSE-shared
  constant's temp was only stored inside the first loop).
- **Signed vs unsigned widening** (`ops/conversions` trace,
  `jit_edge_arith.scl`): S7 `L` zero-extends, so sign extension of i16->i32
  needs SLD 16 + **SSD** 16 (SRD is a *logical* shift in S7/awlsim — using
  SRD zero-fills). Unsigned sources (BYTE/WORD/UINT = LogicalType) must set
  the `zext` attr on awl.cast instead.
- **MOD/DIV signs** (`jit_edge_arith.scl`): IEC 61131-3 trunc-toward-zero;
  -5 MOD 4 = -1 (dividend's sign), 5 / -4 = -1.
- **REAL f32 edges** (`jit_edge_arith.scl`): div-by-zero gives +-inf by the
  dividend's sign and 0/0 = NaN (awlsim /R rule), 0.1+0.2 = 0x3E99999A, and
  REAL_TO_INT ties round half away from zero (impl-defined in IEC table 22,
  pinned).
- **Strings** (`jit_edge_string.scl`): LEN('')=0, CONCAT truncates at the
  254-char max, '$27'/'$$' escapes produce single characters.
- **Timers** (`jit_edge_timer.scl`): TON PT=100ms Q timing; PT=0 is
  implementation-defined (awlsim resets; we reset on the first scan) —
  pinned with a comment rather than asserted as standard.
- **Control flow** (`jit_edge_control.scl`): nested FOR with CONTINUE/EXIT
  (deterministic sum/cnt) and CASE with range+list selectors.

## Coverage notes

There are no C++ unit tests (no gtest); the lit + harness tiers above are the
test suite. The 92-trace conformance corpus plus fuzzing covers the common
paths; the `jit_edge_*` lit tests exist precisely for the corners the fixed
traces do not reach (negative MOD, inf/NaN, string truncation, PT=0 timers).
