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
| AWL | IEC 61131-3 STL block end (BE/BEA/BEU), Siemens AWL manual "Block end" | full | [lib/AwlGen/AwlGen.cpp:2781](../lib/AwlGen/AwlGen.cpp#L2781) |
| AWL | IEC 61131-3 STL conditional block end (BEB/BEC), Siemens AWL manual | full | [lib/AwlGen/AwlGen.cpp:2787](../lib/AwlGen/AwlGen.cpp#L2787) |
| AWL | IEC 61131-3 STL no-operation (NOP) | full | [lib/AwlGen/AwlGen.cpp:2798](../lib/AwlGen/AwlGen.cpp#L2798) |
| AWL | IEC 61131-3 STL load (L), Siemens AWL manual "Load and transfer" — see parseLoad | full | [lib/AwlGen/AwlGen.cpp:2806](../lib/AwlGen/AwlGen.cpp#L2806) |
| AWL | IEC 61131-3 STL transfer (T), Siemens AWL manual "Load and transfer" — see parseStore | full | [lib/AwlGen/AwlGen.cpp:2809](../lib/AwlGen/AwlGen.cpp#L2809) |
| AWL | IEC 61131-3 STL boolean logic ops (U/UN/A/AN/O/ON) + STW comparisons (U ==0 etc.), Siemens AWL manual "Logic operations" | full | [lib/AwlGen/AwlGen.cpp:2812](../lib/AwlGen/AwlGen.cpp#L2812) |
| AWL | IEC 61131-3 STL NOT (negate RLO) | full | [lib/AwlGen/AwlGen.cpp:2862](../lib/AwlGen/AwlGen.cpp#L2862) |
| AWL | IEC 61131-3 STL SET (RLO := 1) | full | [lib/AwlGen/AwlGen.cpp:2874](../lib/AwlGen/AwlGen.cpp#L2874) |
| AWL | IEC 61131-3 STL CLR (RLO := 0) | full | [lib/AwlGen/AwlGen.cpp:2884](../lib/AwlGen/AwlGen.cpp#L2884) |
| AWL | Siemens AWL manual "Opening a data block" (AUF/OPN DB/DI, incl. AUF DB [LW n]) | full | [lib/AwlGen/AwlGen.cpp:2894](../lib/AwlGen/AwlGen.cpp#L2894) |
| AWL | Siemens AWL manual "Address register instructions" (LAR1/LAR2/+ARn; LAR loads the operand content as pointer, awlsim insn_larN) | full | [lib/AwlGen/AwlGen.cpp:2930](../lib/AwlGen/AwlGen.cpp#L2930) |
| AWL | Siemens AWL manual "Address register instructions" (TAR1/TAR2/TAR; TAR stores the AR as a raw S7 pointer, awlsim insn_tarN) | full | [lib/AwlGen/AwlGen.cpp:3010](../lib/AwlGen/AwlGen.cpp#L3010) |
| AWL | IEC 61131-3 STL bit set/reset (S/R), Siemens AWL manual "Bit logic" | full | [lib/AwlGen/AwlGen.cpp:3119](../lib/AwlGen/AwlGen.cpp#L3119) |
| AWL | IEC 61131-3 STL word/dword logic (AW/OW/XOW/AD/OD/XOD), Siemens AWL manual "Word logic" | full | [lib/AwlGen/AwlGen.cpp:3124](../lib/AwlGen/AwlGen.cpp#L3124) |
| AWL | Siemens AWL manual "Timer instructions" (SP/SE/SS/SV/SI/SD, S5TIME), runtime scl_rt_timer_* | full | [lib/AwlGen/AwlGen.cpp:3128](../lib/AwlGen/AwlGen.cpp#L3128) |
| AWL | Siemens AWL manual "Counter instructions" (ZV/ZR/CU/CD), runtime scl_rt_counter_op | full | [lib/AwlGen/AwlGen.cpp:3132](../lib/AwlGen/AwlGen.cpp#L3132) |
| AWL | IEC 61131-3 STL unconditional/conditional jumps (SPA/SPB/SPN/SPBN/JU/JB/JNB/JBI/JC/LOOP), Siemens AWL manual "Jumps" | full | [lib/AwlGen/AwlGen.cpp:3135](../lib/AwlGen/AwlGen.cpp#L3135) |
| AWL | Siemens AWL manual "SAVE" (RLO -> BR status bit) | full | [lib/AwlGen/AwlGen.cpp:3138](../lib/AwlGen/AwlGen.cpp#L3138) |
| AWL | Siemens AWL manual S5-compat SF (english alias of SA, off-delay timer start — awlsim english2german "SF" -> "SA") | full | [lib/AwlGen/AwlGen.cpp:3148](../lib/AwlGen/AwlGen.cpp#L3148) |
| AWL | IEC 61131-3 STL TAK (exchange accumulators), Siemens AWL manual "Accumulator instructions" | full | [lib/AwlGen/AwlGen.cpp:3151](../lib/AwlGen/AwlGen.cpp#L3151) |
| AWL | IEC 61131-3 STL NEGI/NEGD (two's-complement ACCU1); awlsim | full | [lib/AwlGen/AwlGen.cpp:3157](../lib/AwlGen/AwlGen.cpp#L3157) |
| AWL | Siemens AWL manual "Shift and rotate" (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA), awl.shift | full | [lib/AwlGen/AwlGen.cpp:3211](../lib/AwlGen/AwlGen.cpp#L3211) |
| AWL | Siemens AWL manual "Real math" (ABS/NEGR/SIN/COS/TAN/...), awl.real_math (approx: float32 host math) | approx | [lib/AwlGen/AwlGen.cpp:3216](../lib/AwlGen/AwlGen.cpp#L3216) |
| AWL | Siemens AWL manual "Value conversions" TRUNC/RND (approx: truncation) | approx | [lib/AwlGen/AwlGen.cpp:3221](../lib/AwlGen/AwlGen.cpp#L3221) |
| AWL | Siemens AWL manual "Value conversions" DTR (dword->real), plus BTI/BTD/ITB/DTB BCD and ITD/DTI/INVI/INVD below | full | [lib/AwlGen/AwlGen.cpp:3248](../lib/AwlGen/AwlGen.cpp#L3248) |
| AWL | Siemens AWL manual "Enabling" FR (timer/counter), runtime scl_rt_timer_fr | full | [lib/AwlGen/AwlGen.cpp:3254](../lib/AwlGen/AwlGen.cpp#L3254) |
| AWL | Siemens AWL manual TDB (swap DB/DI registers) (approx: not modelled) | approx | [lib/AwlGen/AwlGen.cpp:3264](../lib/AwlGen/AwlGen.cpp#L3264) |
| AWL | Siemens AWL manual "Opening a data block" OPN (= AUF) | full | [lib/AwlGen/AwlGen.cpp:3275](../lib/AwlGen/AwlGen.cpp#L3275) |
| AWL | awlsim test directive __STWRST (reset status word) | full | [lib/AwlGen/AwlGen.cpp:3308](../lib/AwlGen/AwlGen.cpp#L3308) |
| AWL | awlsim test directive __FEATURE t [, v] (nrAccus / OB-temp presets), runtime scl_rt_feature | full | [lib/AwlGen/AwlGen.cpp:3320](../lib/AwlGen/AwlGen.cpp#L3320) |
| AWL | awlsim test directive __SLEEP ms (advance virtual clock), runtime scl_rt_sleep | full | [lib/AwlGen/AwlGen.cpp:3340](../lib/AwlGen/AwlGen.cpp#L3340) |
| AWL | Siemens AWL manual LC (load timer/counter current value, BCD-coded) | full | [lib/AwlGen/AwlGen.cpp:3350](../lib/AwlGen/AwlGen.cpp#L3350) |
| AWL | Siemens AWL manual FP/FN edge memory — the operand bit itself is the edge flag (awlsim insn_FP/FN): store flag := RLO, then new RLO = RLO & !oldFlag (FP) / !RLO & oldFlag (FN) | full | [lib/AwlGen/AwlGen.cpp:3369](../lib/AwlGen/AwlGen.cpp#L3369) |
| AWL | Siemens AWL manual MCR (master control relay) — MCRA/MCRD set the (compile-time tracked) mcrActive flag; SAVE (RLO -> BR) stays a no-op | full | [lib/AwlGen/AwlGen.cpp:3395](../lib/AwlGen/AwlGen.cpp#L3395) |
| AWL | Siemens AWL manual BLD <n> (display code — no-op) | full | [lib/AwlGen/AwlGen.cpp:3406](../lib/AwlGen/AwlGen.cpp#L3406) |
| AWL | Siemens AWL manual MCR( / MCR ) nesting — the pushed VKE (savedRlo of the isMcr bracket) gates stores via mcrOn()/gatedRlo() | full | [lib/AwlGen/AwlGen.cpp:3412](../lib/AwlGen/AwlGen.cpp#L3412) |
| AWL | IEC 61131-3 STL SPBB/SPB-branch save (save RLO into BR) | full | [lib/AwlGen/AwlGen.cpp:3438](../lib/AwlGen/AwlGen.cpp#L3438) |
| AWL | Siemens AWL manual ENT/PUSH (push accumulators), POP/LEAVE below | full | [lib/AwlGen/AwlGen.cpp:3441](../lib/AwlGen/AwlGen.cpp#L3441) |
| AWL | Siemens AWL manual POP/LEAVE (pop accumulators) | full | [lib/AwlGen/AwlGen.cpp:3451](../lib/AwlGen/AwlGen.cpp#L3451) |
| AWL | Siemens AWL manual INC/DEC (ACCU1 +/- n) | full | [lib/AwlGen/AwlGen.cpp:3462](../lib/AwlGen/AwlGen.cpp#L3462) |
| AWL | Siemens AWL manual INC/DEC n — byte op on ACCU1 (awlsim setByte) | full | [lib/AwlGen/AwlGen.cpp:3463](../lib/AwlGen/AwlGen.cpp#L3463) |
| AWL | Siemens AWL manual "Value conversions" ITD/DTI — ITD sign-extends the low word of ACCU1 to a dword (awlsim setDWord(getSignedWord)) | full | [lib/AwlGen/AwlGen.cpp:3484](../lib/AwlGen/AwlGen.cpp#L3484) |
| AWL | Siemens AWL manual INVI/INVD (one's complement) — INVI inverts the low word keeping the high word (awlsim setWord), INVD inverts the full dword | full | [lib/AwlGen/AwlGen.cpp:3494](../lib/AwlGen/AwlGen.cpp#L3494) |
| AWL | Siemens AWL manual MOD (remainder) | full | [lib/AwlGen/AwlGen.cpp:3513](../lib/AwlGen/AwlGen.cpp#L3513) |
| AWL | Siemens AWL manual TAD (dword accumulator exchange) | full | [lib/AwlGen/AwlGen.cpp:3538](../lib/AwlGen/AwlGen.cpp#L3538) |
| AWL | Siemens AWL manual TAD (swapEndianDWord(ACCU1): reverse all four bytes — awlsim insn_TAD) | full | [lib/AwlGen/AwlGen.cpp:3539](../lib/AwlGen/AwlGen.cpp#L3539) |
| AWL | IEC 61131-3 STL X/XN (XOR RLO with bit) | full | [lib/AwlGen/AwlGen.cpp:3581](../lib/AwlGen/AwlGen.cpp#L3581) |
| AWL | Siemens AWL manual SPL (jump to label via ACCU1) | full | [lib/AwlGen/AwlGen.cpp:3609](../lib/AwlGen/AwlGen.cpp#L3609) |
| AWL | IEC 61131-3 STL conditional jumps JNB/JBI/SPBNB/SPBI/SPBIN (with BR save) | full | [lib/AwlGen/AwlGen.cpp:3643](../lib/AwlGen/AwlGen.cpp#L3643) |
| AWL | IEC 61131-3 STL conditional jumps JB/JBI/JC (RLO/BR conditions) | full | [lib/AwlGen/AwlGen.cpp:3647](../lib/AwlGen/AwlGen.cpp#L3647) |
| AWL | IEC 61131-3 STL LOOP (decrement ACCU1, jump if != 0) | full | [lib/AwlGen/AwlGen.cpp:3650](../lib/AwlGen/AwlGen.cpp#L3650) |
| AWL | IEC 61131-3 STL zero/plus/minus jumps (SPZ/SPPZ/SPMZ/SPS/SPP/SPM) | full | [lib/AwlGen/AwlGen.cpp:3682](../lib/AwlGen/AwlGen.cpp#L3682) |
| AWL | Siemens AWL manual "Value conversions" BTI/BTD (BCD -> binary); | full | [lib/AwlGen/AwlGen.cpp:3717](../lib/AwlGen/AwlGen.cpp#L3717) |
| AWL | Siemens AWL manual "Value conversions" ITB/DTB (binary -> BCD); | full | [lib/AwlGen/AwlGen.cpp:3751](../lib/AwlGen/AwlGen.cpp#L3751) |
| AWL | Siemens AWL manual TAW (swap the two bytes of ACCU1's low word — awlsim insn_TAW: accu1 = (accu1 & 0xFFFF0000) | swapEndianWord(accu1 & 0xFFFF)) | full | [lib/AwlGen/AwlGen.cpp:3792](../lib/AwlGen/AwlGen.cpp#L3792) |
| AWL | awlsim test directive __ASSERT<cmp> (value oracles, runtime scl_rt_assert_cmp[_r]) | full | [lib/AwlGen/AwlGen.cpp:3815](../lib/AwlGen/AwlGen.cpp#L3815) |
| AWL | IEC 61131-3 STL block calls (CALL/UC/CC, incl. parameterized CALL and indirect UC FC [MW x]), see parseCall | full | [lib/AwlGen/AwlGen.cpp:3818](../lib/AwlGen/AwlGen.cpp#L3818) |
| AWL | Siemens AWL manual "Indirect addressing" ([AR1, P#x.y] area-relative, memory-indirect [LD n]/[#var] base = operand value, T/Z [..] = block number) | full | [lib/AwlGen/AwlGen.cpp:4242](../lib/AwlGen/AwlGen.cpp#L4242) |
| AWL | IEC 61131-3 STL / Siemens AWL manual "Addressing modes": absolute (M 10.0, DB1.DBW 4), indirect ([AR1, P#x.y], [LD n], [#var]), symbolic, P# pointer literals | full | [lib/AwlGen/AwlGen.cpp:4394](../lib/AwlGen/AwlGen.cpp#L4394) |
| AWL | Siemens AWL manual symbolic DB members (DBn.SYMBOL...) resolved through the DATA_BLOCK layout (approx: a member missing from a declared DB resolves to DBW 0/DBX 0.0 of that DB) | approx | [lib/AwlGen/AwlGen.cpp:6213](../lib/AwlGen/AwlGen.cpp#L6213) |
| AWL | Siemens AWL manual symbolic DB members of undeclared DBs, auto-assigned offsets in order of first use (awlsim symbol tables) -> autoDbCursor branch of resolveDbMember (approx: DB contents are runtime memory, so the synthetic offset only fixes the address, not the value) | approx | [lib/AwlGen/AwlGen.cpp:6232](../lib/AwlGen/AwlGen.cpp#L6232) |
| AWL | Siemens AWL manual "Calling blocks" (CALL/UC/CC with parameter lists := / =>, FB instance DB, multi-instance via AR2) | full | [lib/AwlGen/AwlGen.cpp:6883](../lib/AwlGen/AwlGen.cpp#L6883) |
| AWL | Siemens AWL manual "Addressing": absolute memory operand -> !scl.direct<area, db, byte, bit, width> (see docs/design/memory-areas.md) | full | [include/sclang/AwlDialect/AwlOps.td:51](../include/sclang/AwlDialect/AwlOps.td#L51) |
| AWL | IEC 61131-3 STL load (L) / status-word reads -> awl.load | full | [include/sclang/AwlDialect/AwlOps.td:94](../include/sclang/AwlDialect/AwlOps.td#L94) |
| AWL | IEC 61131-3 STL transfer (T) / bit set-reset (S/R) -> awl.store | full | [include/sclang/AwlDialect/AwlOps.td:107](../include/sclang/AwlDialect/AwlOps.td#L107) |
| AWL | Siemens AWL manual shift/rotate (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA) -> awl.shift | full | [include/sclang/AwlDialect/AwlOps.td:197](../include/sclang/AwlDialect/AwlOps.td#L197) |
| AWL | Siemens AWL manual real arithmetic (ABS/NEGR/SIN/...) -> awl.real_math | approx | [include/sclang/AwlDialect/AwlOps.td:207](../include/sclang/AwlDialect/AwlOps.td#L207) |
| AWL | IEC 61131-3 STL block call (CALL/UC/CC) -> awl.call (FC VL params, FB instance DB, SFC/SFB runtime) | full | [include/sclang/AwlDialect/AwlOps.td:254](../include/sclang/AwlDialect/AwlOps.td#L254) |
| AWL | Siemens AWL manual "Accumulator instructions" (ACCU1..4): runtime | full | [include/sclang/AwlDialect/AwlOps.td:301](../include/sclang/AwlDialect/AwlOps.td#L301) |
| AWL | awlsim test oracle __ASSERT<cmp> -> awl.assert/assert_r -> scl_rt_assert_cmp[_r] | full | [include/sclang/AwlDialect/AwlOps.td:393](../include/sclang/AwlDialect/AwlOps.td#L393) |
| AWL | awlsim __FEATURE t [, v] -> awl.feature -> scl_rt_feature | full | [include/sclang/AwlDialect/AwlOps.td:444](../include/sclang/AwlDialect/AwlOps.td#L444) |
| AWL | Siemens AWL manual AUF/OPN DB n -> awl.set_db / set_dbv / set_div (DB/DI registers, ABI v5) | full | [include/sclang/AwlDialect/AwlOps.td:463](../include/sclang/AwlDialect/AwlOps.td#L463) |
| AWL | Siemens AWL manual LARn/TARn -> awl.ar_set/ar_get/ar_raw (packed internal AR, raw S7 on TAR to memory) | full | [include/sclang/AwlDialect/AwlOps.td:503](../include/sclang/AwlDialect/AwlOps.td#L503) |
| AWL | Siemens AWL manual indirect addressing [ARn, P#x.y] -> awl.indirect (AR + bit offset) | full | [include/sclang/AwlDialect/AwlOps.td:540](../include/sclang/AwlDialect/AwlOps.td#L540) |
| AWL | Siemens AWL manual indirect load (L B/W/D [ARn, P#x.y]) -> awl.load_indirect | full | [include/sclang/AwlDialect/AwlOps.td:549](../include/sclang/AwlDialect/AwlOps.td#L549) |
| AWL | Siemens AWL manual indirect store (T B/W/D [ARn, P#x.y]) -> awl.store_indirect | full | [include/sclang/AwlDialect/AwlOps.td:558](../include/sclang/AwlDialect/AwlOps.td#L558) |
| SCL | IEC 61131-3 ST organization block (OB) -> void function with scan-cycle hooks (ADR 002) | full | [lib/SclGen/MLIRGen.cpp:464](../lib/SclGen/MLIRGen.cpp#L464) |
| SCL | IEC 61131-3 ST statement list: assignment, call, IF/ELSIF/ELSE, | full | [lib/SclGen/MLIRGen.cpp:786](../lib/SclGen/MLIRGen.cpp#L786) |
| SCL | IEC 61131-3 ST assignment (:=), incl. string/array copies -> see mlirGenStringAssignment and the array-copy path below | full | [lib/SclGen/MLIRGen.cpp:1114](../lib/SclGen/MLIRGen.cpp#L1114) |
| SCL | IEC 61131-3 ST type conversions (INT_TO_WORD, CHAR_TO_INT, ...) -> mlirGenBuiltinConversion; string functions (LEN/CONCAT/...) -> mlirGenStringBuiltin | full | [lib/SclGen/MLIRGen.cpp:2268](../lib/SclGen/MLIRGen.cpp#L2268) |
| SCL | IEC 61131-3 ST string functions LEN/LEFT/RIGHT/MID/FIND/CONCAT/DELETE/INSERT/REPLACE and <type>_TO_STRING/STRING_TO_<type> -> mlirGenStringBuiltin (approx: flat zero-padded buffers with C-string length; STRING_TO_REAL/REAL_TO_STRING/VAL_STRG remain external stubs) | approx | [lib/SclGen/MLIRGen.cpp:2370](../lib/SclGen/MLIRGen.cpp#L2370) |
| AWL | IEC 61131-3 STL memory areas (P/Q/M/DB/DI/L/T/C, big-endian S7 storage — docs/decisions/001) -> scl_rt_read/read_bit (C ABI, docs/decisions/002) | full | [lib/Runtime/SimRuntime.cpp:259](../lib/Runtime/SimRuntime.cpp#L259) |
| AWL | IEC 61131-3 STL memory areas, stores -> scl_rt_write/write_bit (C ABI) | full | [lib/Runtime/SimRuntime.cpp:309](../lib/Runtime/SimRuntime.cpp#L309) |
| AWL | Siemens AWL manual local data stack (L/VL frames per block call, incl. FC param area and 256-depth guard) -> scl_rt_l_push/l_pop/param_in/param_out | full | [lib/Runtime/SimRuntime.cpp:563](../lib/Runtime/SimRuntime.cpp#L563) |
| AWL | Siemens AWL manual address registers AR1/AR2 -> scl_rt_ar_get/ar_set/ar_add/ar_raw (packed internal form) | full | [lib/Runtime/SimRuntime.cpp:1054](../lib/Runtime/SimRuntime.cpp#L1054) |
| AWL | Siemens AWL manual indirect addressing -> scl_rt_read_indirect/write_indirect (raw S7 <-> packed decode) | full | [lib/Runtime/SimRuntime.cpp:1116](../lib/Runtime/SimRuntime.cpp#L1116) |
| AWL | Siemens AWL manual system functions SFC/SFB -> scl_rt_sfc/scl_rt_sfb (runtime-staged params in the marker area) | full | [lib/Runtime/SimRuntime.cpp:1156](../lib/Runtime/SimRuntime.cpp#L1156) |
| AWL | Siemens AWL manual status word (STW) cleared at OB entry (awlsim statusWord.reset) -> scl_rt_ob_start | full | [lib/Runtime/SimRuntime.cpp:1735](../lib/Runtime/SimRuntime.cpp#L1735) |
| AWL | awlsim __FEATURE t [, v] -> scl_rt_feature (nrAccus, OB-temp presets) | full | [lib/Runtime/SimRuntime.cpp:1768](../lib/Runtime/SimRuntime.cpp#L1768) |
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
