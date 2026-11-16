// Spec-conformance matrix freshness: the generated matrix section of
// docs/spec-conformance.md must match the SPEC[...] tags in the sources.
// Regenerate after adding/changing tags:
//   python3 test/conformance/gen_spec_matrix.py
// See docs/spec-conformance.md ("How to trace").
// RUN: python3 %S/gen_spec_matrix.py --check
