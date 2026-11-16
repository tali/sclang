# Spec conformance and traceability (AWL/STL, SCL/ST)

This document states which official specifications sclang conforms to, and
makes every claim traceable from the spec to the code and the tests.

## Which documents define the languages

| sclang input | IEC 61131-3 language | Siemens reference |
|---|---|---|
| `.awl` (AWL) | STL — *Structured Text List* / instruction list (IEC 61131-3) | *AWL for S7-300/400* manual (statement list); AWL and STL are the same language in different naming |
| `.scl` (SCL) | ST — *Structured Text* (IEC 61131-3) | *Structured Control Language (SCL) for S7-300/400* manual (PDF link in [README.md](../README.md#references)) |

Links: IEC 61131-3 (see README REFERENCES), the Siemens SCL manual, and
[awlsim](https://github.com/mbuesch/awlsim) as the *executable* reference
semantics — awlsim's own test suite (`awlsim-test/tests/tc*`) is used as an
external conformance corpus, see [conformance.md](conformance.md).

Conformance status vocabulary (used in the `SPEC[...]` tags and the matrix):

- **full** — implemented to match the reference semantics (checked against
  awlsim behavior and its `__ASSERT` oracles where available);
- **approx** — implemented with a documented approximation (the tag or the
  linked design doc says which);
- **gap** — known missing/unsupported.

## How to trace a spec item to the code

Every feature in the sources carries a greppable tag in a `//` comment:

```
// SPEC[AWL] IEC 61131-3 STL load instruction (L), Siemens AWL manual
// SPEC[SCL] IEC 61131-3 ST CASE statement (approx)
```

Format: `SPEC[<lang>] <reference> [<status>]`, where `<lang>` is `AWL`/`STL`
or `SCL`/`ST` and `<status>` is `full` (default), `approx` or `gap`. The
status may carry a reason — `(approx: truncation)` — which stays in the
reference text and still counts as that status, not as `full`.

To trace: `grep -rn "SPEC\[" lib include test` lists every tagged site with
file:line. The matrix below is **generated** from those tags by
`test/conformance/gen_spec_matrix.py` (run it after adding/changing tags;
the lit test `test/conformance/spec_matrix.mlir` fails in `check-sclang`
when the checked-in matrix is stale, so it cannot silently drift).

Reading one matrix row: language → spec reference → status → code location
(the `file:line` link jumps to the handler on GitHub). From the handler you
can follow the dialect op (`include/sclang/AwlDialect/AwlOps.td`), its
lowering (`lib/SclTransforms/LowerToStd.cpp`) and the runtime function
(`lib/Runtime/SimRuntime.cpp`, C ABI in `docs/decisions/002-runtime-c-abi.md`),
plus the lit tests under `test/` and the external corpus tests listed in
[conformance.md](conformance.md).

## Generated conformance matrix

<!-- SPEC-MATRIX:BEGIN -->

| language | spec reference | status | code |
|---|---|---|---|
| AWL | IEC 61131-3 STL block end (BE/BEA/BEU), Siemens AWL manual "Block end" | full | [lib/AwlGen/AwlGen.cpp:2723](../lib/AwlGen/AwlGen.cpp#L2723) |
| AWL | IEC 61131-3 STL conditional block end (BEB/BEC), Siemens AWL manual | full | [lib/AwlGen/AwlGen.cpp:2729](../lib/AwlGen/AwlGen.cpp#L2729) |
| AWL | IEC 61131-3 STL no-operation (NOP) | full | [lib/AwlGen/AwlGen.cpp:2740](../lib/AwlGen/AwlGen.cpp#L2740) |
| AWL | IEC 61131-3 STL load (L), Siemens AWL manual "Load and transfer" — see parseLoad | full | [lib/AwlGen/AwlGen.cpp:2748](../lib/AwlGen/AwlGen.cpp#L2748) |
| AWL | IEC 61131-3 STL transfer (T), Siemens AWL manual "Load and transfer" — see parseStore | full | [lib/AwlGen/AwlGen.cpp:2751](../lib/AwlGen/AwlGen.cpp#L2751) |
| AWL | IEC 61131-3 STL boolean logic ops (U/UN/A/AN/O/ON) + STW comparisons (U ==0 etc.), Siemens AWL manual "Logic operations" | full | [lib/AwlGen/AwlGen.cpp:2754](../lib/AwlGen/AwlGen.cpp#L2754) |
| AWL | IEC 61131-3 STL NOT (negate RLO) | full | [lib/AwlGen/AwlGen.cpp:2804](../lib/AwlGen/AwlGen.cpp#L2804) |
| AWL | IEC 61131-3 STL SET (RLO := 1) | full | [lib/AwlGen/AwlGen.cpp:2816](../lib/AwlGen/AwlGen.cpp#L2816) |
| AWL | IEC 61131-3 STL CLR (RLO := 0) | full | [lib/AwlGen/AwlGen.cpp:2824](../lib/AwlGen/AwlGen.cpp#L2824) |
| AWL | Siemens AWL manual "Opening a data block" (AUF/OPN DB/DI, incl. AUF DB [LW n]) | full | [lib/AwlGen/AwlGen.cpp:2833](../lib/AwlGen/AwlGen.cpp#L2833) |
| AWL | Siemens AWL manual "Address register instructions" (LAR1/LAR2/+ARn; LAR loads the operand content as pointer, awlsim insn_larN) | full | [lib/AwlGen/AwlGen.cpp:2869](../lib/AwlGen/AwlGen.cpp#L2869) |
| AWL | Siemens AWL manual "Address register instructions" (TAR1/TAR2/TAR; TAR stores the AR as a raw S7 pointer, awlsim insn_tarN) | full | [lib/AwlGen/AwlGen.cpp:2949](../lib/AwlGen/AwlGen.cpp#L2949) |
| AWL | IEC 61131-3 STL bit set/reset (S/R), Siemens AWL manual "Bit logic" | full | [lib/AwlGen/AwlGen.cpp:3049](../lib/AwlGen/AwlGen.cpp#L3049) |
| AWL | IEC 61131-3 STL word/dword logic (AW/OW/XOW/AD/OD/XOD), Siemens AWL manual "Word logic" | full | [lib/AwlGen/AwlGen.cpp:3054](../lib/AwlGen/AwlGen.cpp#L3054) |
| AWL | Siemens AWL manual "Timer instructions" (SP/SE/SS/SV/SI/SD, S5TIME), runtime scl_rt_timer_* | full | [lib/AwlGen/AwlGen.cpp:3058](../lib/AwlGen/AwlGen.cpp#L3058) |
| AWL | Siemens AWL manual "Counter instructions" (ZV/ZR/CU/CD), runtime scl_rt_counter_op | full | [lib/AwlGen/AwlGen.cpp:3062](../lib/AwlGen/AwlGen.cpp#L3062) |
| AWL | IEC 61131-3 STL unconditional/conditional jumps (SPA/SPB/SPN/SPBN/JU/JB/JNB/JBI/JC/LOOP), Siemens AWL manual "Jumps" | full | [lib/AwlGen/AwlGen.cpp:3065](../lib/AwlGen/AwlGen.cpp#L3065) |
| AWL | Siemens AWL manual "SAVE" (RLO -> BR status bit) | full | [lib/AwlGen/AwlGen.cpp:3068](../lib/AwlGen/AwlGen.cpp#L3068) |
| AWL | Siemens AWL S5-compat SF (timer/counter enable) | approx | [lib/AwlGen/AwlGen.cpp:3078](../lib/AwlGen/AwlGen.cpp#L3078) |
| AWL | IEC 61131-3 STL TAK (exchange accumulators), Siemens AWL manual "Accumulator instructions" | full | [lib/AwlGen/AwlGen.cpp:3087](../lib/AwlGen/AwlGen.cpp#L3087) |
| AWL | IEC 61131-3 STL NEGI/NEGD (two's-complement ACCU1) | full | [lib/AwlGen/AwlGen.cpp:3093](../lib/AwlGen/AwlGen.cpp#L3093) |
| AWL | Siemens AWL manual "Shift and rotate" (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA), awl.shift | full | [lib/AwlGen/AwlGen.cpp:3101](../lib/AwlGen/AwlGen.cpp#L3101) |
| AWL | Siemens AWL manual "Real math" (ABS/NEGR/SIN/COS/TAN/...), awl.real_math (approx: float32 host math) | approx | [lib/AwlGen/AwlGen.cpp:3106](../lib/AwlGen/AwlGen.cpp#L3106) |
| AWL | Siemens AWL manual "Value conversions" TRUNC/RND (approx: truncation) | approx | [lib/AwlGen/AwlGen.cpp:3111](../lib/AwlGen/AwlGen.cpp#L3111) |
| AWL | Siemens AWL manual "Value conversions" DTR (dword->real), plus BTI/BTD/ITB/DTB BCD and ITD/DTI/INVI/INVD below | full | [lib/AwlGen/AwlGen.cpp:3138](../lib/AwlGen/AwlGen.cpp#L3138) |
| AWL | Siemens AWL manual "Enabling" FR (timer/counter), runtime scl_rt_timer_fr | full | [lib/AwlGen/AwlGen.cpp:3144](../lib/AwlGen/AwlGen.cpp#L3144) |
| AWL | Siemens AWL manual TDB (swap DB/DI registers) (approx: not modelled) | approx | [lib/AwlGen/AwlGen.cpp:3154](../lib/AwlGen/AwlGen.cpp#L3154) |
| AWL | Siemens AWL manual "Opening a data block" OPN (= AUF) | full | [lib/AwlGen/AwlGen.cpp:3165](../lib/AwlGen/AwlGen.cpp#L3165) |
| AWL | awlsim test directive __STWRST (reset status word) | full | [lib/AwlGen/AwlGen.cpp:3198](../lib/AwlGen/AwlGen.cpp#L3198) |
| AWL | awlsim test directive __FEATURE t [, v] (nrAccus / OB-temp presets), runtime scl_rt_feature | full | [lib/AwlGen/AwlGen.cpp:3206](../lib/AwlGen/AwlGen.cpp#L3206) |
| AWL | awlsim test directive __SLEEP ms (advance virtual clock), runtime scl_rt_sleep | full | [lib/AwlGen/AwlGen.cpp:3226](../lib/AwlGen/AwlGen.cpp#L3226) |
| AWL | Siemens AWL manual LC (load timer/counter current value, BCD-coded) | full | [lib/AwlGen/AwlGen.cpp:3236](../lib/AwlGen/AwlGen.cpp#L3236) |
| AWL | Siemens AWL manual FP/FN edge memory (approx: approximated by U/UN — no edge memory) | approx | [lib/AwlGen/AwlGen.cpp:3255](../lib/AwlGen/AwlGen.cpp#L3255) |
| AWL | Siemens AWL manual MCR (master control relay) (approx: bracket only) | approx | [lib/AwlGen/AwlGen.cpp:3258](../lib/AwlGen/AwlGen.cpp#L3258) |
| AWL | Siemens AWL manual BLD <n> (display code — no-op) | full | [lib/AwlGen/AwlGen.cpp:3261](../lib/AwlGen/AwlGen.cpp#L3261) |
| AWL | Siemens AWL manual MCR( / MCR ) nesting | approx | [lib/AwlGen/AwlGen.cpp:3267](../lib/AwlGen/AwlGen.cpp#L3267) |
| AWL | IEC 61131-3 STL SPBB/SPB-branch save (save RLO into BR) | full | [lib/AwlGen/AwlGen.cpp:3286](../lib/AwlGen/AwlGen.cpp#L3286) |
| AWL | Siemens AWL manual ENT/PUSH (push accumulators), POP/LEAVE below | full | [lib/AwlGen/AwlGen.cpp:3289](../lib/AwlGen/AwlGen.cpp#L3289) |
| AWL | Siemens AWL manual POP/LEAVE (pop accumulators) | full | [lib/AwlGen/AwlGen.cpp:3294](../lib/AwlGen/AwlGen.cpp#L3294) |
| AWL | Siemens AWL manual INC/DEC (ACCU1 +/- n) | full | [lib/AwlGen/AwlGen.cpp:3303](../lib/AwlGen/AwlGen.cpp#L3303) |
| AWL | Siemens AWL manual "Value conversions" ITD/DTI | full | [lib/AwlGen/AwlGen.cpp:3316](../lib/AwlGen/AwlGen.cpp#L3316) |
| AWL | Siemens AWL manual INVI/INVD (one's complement) | full | [lib/AwlGen/AwlGen.cpp:3324](../lib/AwlGen/AwlGen.cpp#L3324) |
| AWL | Siemens AWL manual MOD (remainder) | full | [lib/AwlGen/AwlGen.cpp:3334](../lib/AwlGen/AwlGen.cpp#L3334) |
| AWL | Siemens AWL manual TAD (dword accumulator exchange) | full | [lib/AwlGen/AwlGen.cpp:3351](../lib/AwlGen/AwlGen.cpp#L3351) |
| AWL | IEC 61131-3 STL X/XN (XOR RLO with bit) | full | [lib/AwlGen/AwlGen.cpp:3364](../lib/AwlGen/AwlGen.cpp#L3364) |
| AWL | Siemens AWL manual SPL (jump to label via ACCU1) | full | [lib/AwlGen/AwlGen.cpp:3401](../lib/AwlGen/AwlGen.cpp#L3401) |
| AWL | IEC 61131-3 STL conditional jumps JNB/JBI/SPBNB/SPBI/SPBIN (with BR save) | full | [lib/AwlGen/AwlGen.cpp:3435](../lib/AwlGen/AwlGen.cpp#L3435) |
| AWL | IEC 61131-3 STL conditional jumps JB/JBI/JC (RLO/BR conditions) | full | [lib/AwlGen/AwlGen.cpp:3439](../lib/AwlGen/AwlGen.cpp#L3439) |
| AWL | IEC 61131-3 STL LOOP (decrement ACCU1, jump if != 0) | full | [lib/AwlGen/AwlGen.cpp:3442](../lib/AwlGen/AwlGen.cpp#L3442) |
| AWL | IEC 61131-3 STL zero/plus/minus jumps (SPZ/SPPZ/SPMZ/SPS/SPP/SPM) | full | [lib/AwlGen/AwlGen.cpp:3456](../lib/AwlGen/AwlGen.cpp#L3456) |
| AWL | Siemens AWL manual "Value conversions" BTI/BTD (BCD -> binary) | full | [lib/AwlGen/AwlGen.cpp:3478](../lib/AwlGen/AwlGen.cpp#L3478) |
| AWL | Siemens AWL manual "Value conversions" ITB/DTB (binary -> BCD) | full | [lib/AwlGen/AwlGen.cpp:3486](../lib/AwlGen/AwlGen.cpp#L3486) |
| AWL | Siemens AWL manual TAW (ACCU1 -> status word) | approx | [lib/AwlGen/AwlGen.cpp:3494](../lib/AwlGen/AwlGen.cpp#L3494) |
| AWL | awlsim test directive __ASSERT<cmp> (value oracles, runtime scl_rt_assert_cmp[_r]) | full | [lib/AwlGen/AwlGen.cpp:3503](../lib/AwlGen/AwlGen.cpp#L3503) |
| AWL | IEC 61131-3 STL block calls (CALL/UC/CC, incl. parameterized CALL and indirect UC FC [MW x]), see parseCall | full | [lib/AwlGen/AwlGen.cpp:3506](../lib/AwlGen/AwlGen.cpp#L3506) |
| AWL | Siemens AWL manual "Indirect addressing" ([AR1, P#x.y] area-relative, memory-indirect [LD n]/[#var] base = operand value, T/Z [..] = block number) | full | [lib/AwlGen/AwlGen.cpp:3789](../lib/AwlGen/AwlGen.cpp#L3789) |
| AWL | IEC 61131-3 STL / Siemens AWL manual "Addressing modes": absolute (M 10.0, DB1.DBW 4), indirect ([AR1, P#x.y], [LD n], [#var]), symbolic, P# pointer literals | full | [lib/AwlGen/AwlGen.cpp:3941](../lib/AwlGen/AwlGen.cpp#L3941) |
| AWL | Siemens AWL manual symbolic DB members (DBn.SYMBOL...) resolved through the DATA_BLOCK layout (approx: a member missing from a declared DB resolves to DBW 0/DBX 0.0 of that DB) | approx | [lib/AwlGen/AwlGen.cpp:5417](../lib/AwlGen/AwlGen.cpp#L5417) |
| AWL | Siemens AWL manual symbolic DB members of undeclared DBs, auto-assigned offsets in order of first use (awlsim symbol tables) -> autoDbCursor branch of resolveDbMember (approx: DB contents are runtime memory, so the synthetic offset only fixes the address, not the value) | approx | [lib/AwlGen/AwlGen.cpp:5436](../lib/AwlGen/AwlGen.cpp#L5436) |
| AWL | Siemens AWL manual "Calling blocks" (CALL/UC/CC with parameter lists := / =>, FB instance DB, multi-instance via AR2) | full | [lib/AwlGen/AwlGen.cpp:6087](../lib/AwlGen/AwlGen.cpp#L6087) |
| AWL | Siemens AWL manual "Addressing": absolute memory operand -> !scl.direct<area, db, byte, bit, width> (see docs/design/memory-areas.md) | full | [include/sclang/AwlDialect/AwlOps.td:51](../include/sclang/AwlDialect/AwlOps.td#L51) |
| AWL | IEC 61131-3 STL load (L) / status-word reads -> awl.load | full | [include/sclang/AwlDialect/AwlOps.td:94](../include/sclang/AwlDialect/AwlOps.td#L94) |
| AWL | IEC 61131-3 STL transfer (T) / bit set-reset (S/R) -> awl.store | full | [include/sclang/AwlDialect/AwlOps.td:107](../include/sclang/AwlDialect/AwlOps.td#L107) |
| AWL | Siemens AWL manual shift/rotate (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA) -> awl.shift | full | [include/sclang/AwlDialect/AwlOps.td:197](../include/sclang/AwlDialect/AwlOps.td#L197) |
| AWL | Siemens AWL manual real arithmetic (ABS/NEGR/SIN/...) -> awl.real_math | approx | [include/sclang/AwlDialect/AwlOps.td:207](../include/sclang/AwlDialect/AwlOps.td#L207) |
| AWL | IEC 61131-3 STL block call (CALL/UC/CC) -> awl.call (FC VL params, FB instance DB, SFC/SFB runtime) | full | [include/sclang/AwlDialect/AwlOps.td:254](../include/sclang/AwlDialect/AwlOps.td#L254) |
| AWL | awlsim test oracle __ASSERT<cmp> -> awl.assert/assert_r -> scl_rt_assert_cmp[_r] | full | [include/sclang/AwlDialect/AwlOps.td:379](../include/sclang/AwlDialect/AwlOps.td#L379) |
| AWL | awlsim __FEATURE t [, v] -> awl.feature -> scl_rt_feature | full | [include/sclang/AwlDialect/AwlOps.td:430](../include/sclang/AwlDialect/AwlOps.td#L430) |
| AWL | Siemens AWL manual AUF/OPN DB n -> awl.set_db / set_dbv / set_div (DB/DI registers, ABI v5) | full | [include/sclang/AwlDialect/AwlOps.td:449](../include/sclang/AwlDialect/AwlOps.td#L449) |
| AWL | Siemens AWL manual LARn/TARn -> awl.ar_set/ar_get/ar_raw (packed internal AR, raw S7 on TAR to memory) | full | [include/sclang/AwlDialect/AwlOps.td:489](../include/sclang/AwlDialect/AwlOps.td#L489) |
| AWL | Siemens AWL manual indirect addressing [ARn, P#x.y] -> awl.indirect (AR + bit offset) | full | [include/sclang/AwlDialect/AwlOps.td:526](../include/sclang/AwlDialect/AwlOps.td#L526) |
| AWL | Siemens AWL manual indirect load (L B/W/D [ARn, P#x.y]) -> awl.load_indirect | full | [include/sclang/AwlDialect/AwlOps.td:535](../include/sclang/AwlDialect/AwlOps.td#L535) |
| AWL | Siemens AWL manual indirect store (T B/W/D [ARn, P#x.y]) -> awl.store_indirect | full | [include/sclang/AwlDialect/AwlOps.td:544](../include/sclang/AwlDialect/AwlOps.td#L544) |
| SCL | IEC 61131-3 ST organization block (OB) -> void function with scan-cycle hooks (ADR 002) | full | [lib/SclGen/MLIRGen.cpp:464](../lib/SclGen/MLIRGen.cpp#L464) |
| SCL | IEC 61131-3 ST statement list: assignment, call, IF/ELSIF/ELSE, | full | [lib/SclGen/MLIRGen.cpp:786](../lib/SclGen/MLIRGen.cpp#L786) |
| SCL | IEC 61131-3 ST assignment (:=), incl. string/array copies -> see mlirGenStringAssignment and the array-copy path below | full | [lib/SclGen/MLIRGen.cpp:1114](../lib/SclGen/MLIRGen.cpp#L1114) |
| SCL | IEC 61131-3 ST type conversions (INT_TO_WORD, CHAR_TO_INT, ...) -> mlirGenBuiltinConversion; string functions (LEN/CONCAT/...) -> mlirGenStringBuiltin | full | [lib/SclGen/MLIRGen.cpp:2268](../lib/SclGen/MLIRGen.cpp#L2268) |
| SCL | IEC 61131-3 ST string functions LEN/LEFT/RIGHT/MID/FIND/CONCAT/DELETE/INSERT/REPLACE and <type>_TO_STRING/STRING_TO_<type> -> mlirGenStringBuiltin (approx: flat zero-padded buffers with C-string length; STRING_TO_REAL/REAL_TO_STRING/VAL_STRG remain external stubs) | approx | [lib/SclGen/MLIRGen.cpp:2370](../lib/SclGen/MLIRGen.cpp#L2370) |
| AWL | IEC 61131-3 STL memory areas (P/Q/M/DB/DI/L/T/C, big-endian S7 storage — docs/decisions/001) -> scl_rt_read/read_bit (C ABI, docs/decisions/002) | full | [lib/Runtime/SimRuntime.cpp:258](../lib/Runtime/SimRuntime.cpp#L258) |
| AWL | IEC 61131-3 STL memory areas, stores -> scl_rt_write/write_bit (C ABI) | full | [lib/Runtime/SimRuntime.cpp:308](../lib/Runtime/SimRuntime.cpp#L308) |
| AWL | Siemens AWL manual local data stack (L/VL frames per block call, incl. FC param area and 256-depth guard) -> scl_rt_l_push/l_pop/param_in/param_out | full | [lib/Runtime/SimRuntime.cpp:562](../lib/Runtime/SimRuntime.cpp#L562) |
| AWL | Siemens AWL manual address registers AR1/AR2 -> scl_rt_ar_get/ar_set/ar_add/ar_raw (packed internal form) | full | [lib/Runtime/SimRuntime.cpp:1053](../lib/Runtime/SimRuntime.cpp#L1053) |
| AWL | Siemens AWL manual indirect addressing -> scl_rt_read_indirect/write_indirect (raw S7 <-> packed decode) | full | [lib/Runtime/SimRuntime.cpp:1115](../lib/Runtime/SimRuntime.cpp#L1115) |
| AWL | Siemens AWL manual system functions SFC/SFB -> scl_rt_sfc/scl_rt_sfb (runtime-staged params in the marker area) | full | [lib/Runtime/SimRuntime.cpp:1155](../lib/Runtime/SimRuntime.cpp#L1155) |
| AWL | Siemens AWL manual status word (STW) cleared at OB entry (awlsim statusWord.reset) -> scl_rt_ob_start | full | [lib/Runtime/SimRuntime.cpp:1707](../lib/Runtime/SimRuntime.cpp#L1707) |
| AWL | awlsim __FEATURE t [, v] -> scl_rt_feature (nrAccus, OB-temp presets) | full | [lib/Runtime/SimRuntime.cpp:1740](../lib/Runtime/SimRuntime.cpp#L1740) |
| AWL | IEC 61131-3 STL memory areas P/Q/M/DB/DI/L/T/C and direct operands | full | [test/AwlGen/areas.awl:2](../test/AwlGen/areas.awl#L2) |
| AWL | awlsim __ASSERT oracle directives -> awl.assert -> scl_rt_assert_cmp | full | [test/AwlGen/assert.awl:2](../test/AwlGen/assert.awl#L2) |
| AWL | IEC 61131-3 STL block calls incl. FB instance DBs and parameter passing | full | [test/AwlGen/call_fb.awl:2](../test/AwlGen/call_fb.awl#L2) |
| AWL | Siemens AWL manual "Addressing": DB register (AUF/OPN), ARs (LAR/TAR), indirect [AR1, P#x.y], DATA_BLOCK init | full | [test/AwlGen/db_indirect.awl:5](../test/AwlGen/db_indirect.awl#L5) |
| AWL | IEC 61131-3 STL instruction set: logic, arithmetic, compare, jumps | full | [test/AwlGen/ops.awl:2](../test/AwlGen/ops.awl#L2) |
| AWL | Siemens AWL manual shift/rotate + real math | full | [test/AwlGen/shift_math.awl:2](../test/AwlGen/shift_math.awl#L2) |
| AWL | Siemens AWL manual timer instructions (SP/SE/SS/SV/SI/SD) + runtime | full | [test/AwlGen/timer.awl:2](../test/AwlGen/timer.awl#L2) |

<!-- SPEC-MATRIX:END -->

## External executable conformance

The corpora groups below (awlsim test-suite layout, cloned outside the repo,
see [conformance.md](conformance.md)) exercise the matrix rows at runtime
with `__ASSERT` value oracles:

- tc100_instructions — instruction set semantics
- tc150_startup — startup OBs / CPU features
- tc200_datatypes — data types (arrays, structs, strings, ANY/POINTER)
- tc300_cpu — CPU behavior (recursion, status word)
- tc999_projects — whole projects

Current pass rates: see [conformance.md](conformance.md#current-results-llvm-17-build-this-writing).
