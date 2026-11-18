// External conformance suite: compiles public SCL/AWL corpora (awlsim test
// projects, GitHub SCL samples) with sclang and enforces minimum pass rates.
// The corpora are NOT in the repo; the runner SKIPs (exit 0) when the root
// is absent. See docs/conformance.md.
// RUN: python3 %S/run_conformance.py --sclang sclang
