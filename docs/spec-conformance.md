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
| AWL | IEC 61131-3 STL block end (BE/BEA/BEU), Siemens AWL manual "Block end" | full | [lib/AwlGen/AwlGen.cpp:2799](../lib/AwlGen/AwlGen.cpp#L2799) |
| AWL | IEC 61131-3 STL conditional block end (BEB/BEC), Siemens AWL manual | full | [lib/AwlGen/AwlGen.cpp:2805](../lib/AwlGen/AwlGen.cpp#L2805) |
| AWL | IEC 61131-3 STL no-operation (NOP) | full | [lib/AwlGen/AwlGen.cpp:2816](../lib/AwlGen/AwlGen.cpp#L2816) |
| AWL | IEC 61131-3 STL load (L), Siemens AWL manual "Load and transfer" — see parseLoad | full | [lib/AwlGen/AwlGen.cpp:2824](../lib/AwlGen/AwlGen.cpp#L2824) |
| AWL | IEC 61131-3 STL transfer (T), Siemens AWL manual "Load and transfer" — see parseStore | full | [lib/AwlGen/AwlGen.cpp:2827](../lib/AwlGen/AwlGen.cpp#L2827) |
| AWL | IEC 61131-3 STL boolean logic ops (U/UN/A/AN/O/ON) + STW comparisons (U ==0 etc.), Siemens AWL manual "Logic operations" | full | [lib/AwlGen/AwlGen.cpp:2830](../lib/AwlGen/AwlGen.cpp#L2830) |
| AWL | IEC 61131-3 STL NOT (negate RLO) | full | [lib/AwlGen/AwlGen.cpp:2880](../lib/AwlGen/AwlGen.cpp#L2880) |
| AWL | IEC 61131-3 STL SET (RLO := 1) | full | [lib/AwlGen/AwlGen.cpp:2892](../lib/AwlGen/AwlGen.cpp#L2892) |
| AWL | IEC 61131-3 STL CLR (RLO := 0) | full | [lib/AwlGen/AwlGen.cpp:2902](../lib/AwlGen/AwlGen.cpp#L2902) |
| AWL | Siemens AWL manual "Opening a data block" (AUF/OPN DB/DI, incl. AUF DB [LW n]) | full | [lib/AwlGen/AwlGen.cpp:2912](../lib/AwlGen/AwlGen.cpp#L2912) |
| AWL | Siemens AWL manual "Address register instructions" (LAR1/LAR2/+ARn; LAR loads the operand content as pointer, awlsim insn_larN) | full | [lib/AwlGen/AwlGen.cpp:2948](../lib/AwlGen/AwlGen.cpp#L2948) |
| AWL | Siemens AWL manual "Address register instructions" (TAR1/TAR2/TAR; TAR stores the AR as a raw S7 pointer, awlsim insn_tarN) | full | [lib/AwlGen/AwlGen.cpp:3028](../lib/AwlGen/AwlGen.cpp#L3028) |
| AWL | IEC 61131-3 STL bit set/reset (S/R), Siemens AWL manual "Bit logic" | full | [lib/AwlGen/AwlGen.cpp:3137](../lib/AwlGen/AwlGen.cpp#L3137) |
| AWL | IEC 61131-3 STL word/dword logic (AW/OW/XOW/AD/OD/XOD), Siemens AWL manual "Word logic" | full | [lib/AwlGen/AwlGen.cpp:3142](../lib/AwlGen/AwlGen.cpp#L3142) |
| AWL | Siemens AWL manual "Timer instructions" (SP/SE/SS/SV/SI/SD, S5TIME), runtime scl_rt_timer_* | full | [lib/AwlGen/AwlGen.cpp:3146](../lib/AwlGen/AwlGen.cpp#L3146) |
| AWL | Siemens AWL manual "Counter instructions" (ZV/ZR/CU/CD), runtime scl_rt_counter_op | full | [lib/AwlGen/AwlGen.cpp:3150](../lib/AwlGen/AwlGen.cpp#L3150) |
| AWL | IEC 61131-3 STL unconditional/conditional jumps (SPA/SPB/SPN/SPBN/JU/JB/JNB/JBI/JC/LOOP), Siemens AWL manual "Jumps" | full | [lib/AwlGen/AwlGen.cpp:3153](../lib/AwlGen/AwlGen.cpp#L3153) |
| AWL | Siemens AWL manual "SAVE" (RLO -> BR status bit) | full | [lib/AwlGen/AwlGen.cpp:3156](../lib/AwlGen/AwlGen.cpp#L3156) |
| AWL | Siemens AWL manual S5-compat SF (english alias of SA, off-delay timer start — awlsim english2german "SF" -> "SA") | full | [lib/AwlGen/AwlGen.cpp:3166](../lib/AwlGen/AwlGen.cpp#L3166) |
| AWL | IEC 61131-3 STL TAK (exchange accumulators), Siemens AWL manual "Accumulator instructions" | full | [lib/AwlGen/AwlGen.cpp:3169](../lib/AwlGen/AwlGen.cpp#L3169) |
| AWL | IEC 61131-3 STL NEGI/NEGD (two's-complement ACCU1); awlsim | full | [lib/AwlGen/AwlGen.cpp:3175](../lib/AwlGen/AwlGen.cpp#L3175) |
| AWL | Siemens AWL manual "Shift and rotate" (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA), awl.shift | full | [lib/AwlGen/AwlGen.cpp:3229](../lib/AwlGen/AwlGen.cpp#L3229) |
| AWL | Siemens AWL manual "Real math" (ABS/NEGR/SIN/COS/TAN/...), awl.real_math (approx: float32 host math) | approx | [lib/AwlGen/AwlGen.cpp:3234](../lib/AwlGen/AwlGen.cpp#L3234) |
| AWL | Siemens AWL manual "Value conversions" TRUNC/RND (approx: truncation) | approx | [lib/AwlGen/AwlGen.cpp:3239](../lib/AwlGen/AwlGen.cpp#L3239) |
| AWL | Siemens AWL manual "Value conversions" DTR (dword->real), plus BTI/BTD/ITB/DTB BCD and ITD/DTI/INVI/INVD below | full | [lib/AwlGen/AwlGen.cpp:3271](../lib/AwlGen/AwlGen.cpp#L3271) |
| AWL | Siemens AWL manual "Enabling" FR (timer/counter), runtime scl_rt_timer_fr | full | [lib/AwlGen/AwlGen.cpp:3277](../lib/AwlGen/AwlGen.cpp#L3277) |
| AWL | Siemens AWL manual TDB (swap DB/DI registers) (approx: not modelled) | approx | [lib/AwlGen/AwlGen.cpp:3287](../lib/AwlGen/AwlGen.cpp#L3287) |
| AWL | Siemens AWL manual "Opening a data block" OPN (= AUF) | full | [lib/AwlGen/AwlGen.cpp:3298](../lib/AwlGen/AwlGen.cpp#L3298) |
| AWL | awlsim test directive __STWRST (reset status word) | full | [lib/AwlGen/AwlGen.cpp:3331](../lib/AwlGen/AwlGen.cpp#L3331) |
| AWL | awlsim test directive __FEATURE t [, v] (nrAccus / OB-temp presets), runtime scl_rt_feature | full | [lib/AwlGen/AwlGen.cpp:3343](../lib/AwlGen/AwlGen.cpp#L3343) |
| AWL | awlsim test directive __SLEEP ms (advance virtual clock), runtime scl_rt_sleep | full | [lib/AwlGen/AwlGen.cpp:3364](../lib/AwlGen/AwlGen.cpp#L3364) |
| AWL | Siemens AWL manual LC (load timer/counter current value, BCD-coded) | full | [lib/AwlGen/AwlGen.cpp:3374](../lib/AwlGen/AwlGen.cpp#L3374) |
| AWL | Siemens AWL manual FP/FN edge memory — the operand bit itself is the edge flag (awlsim insn_FP/FN): store flag := RLO, then new RLO = RLO & !oldFlag (FP) / !RLO & oldFlag (FN) | full | [lib/AwlGen/AwlGen.cpp:3393](../lib/AwlGen/AwlGen.cpp#L3393) |
| AWL | Siemens AWL manual MCR (master control relay) — MCRA/MCRD set the (compile-time tracked) mcrActive flag; SAVE (RLO -> BR) stays a no-op | full | [lib/AwlGen/AwlGen.cpp:3419](../lib/AwlGen/AwlGen.cpp#L3419) |
| AWL | Siemens AWL manual BLD <n> (display code — no-op) | full | [lib/AwlGen/AwlGen.cpp:3430](../lib/AwlGen/AwlGen.cpp#L3430) |
| AWL | Siemens AWL manual MCR( / MCR ) nesting — the pushed VKE (savedRlo of the isMcr bracket) gates stores via mcrOn()/gatedRlo() | full | [lib/AwlGen/AwlGen.cpp:3436](../lib/AwlGen/AwlGen.cpp#L3436) |
| AWL | IEC 61131-3 STL SPBB/SPB-branch save (save RLO into BR) | full | [lib/AwlGen/AwlGen.cpp:3462](../lib/AwlGen/AwlGen.cpp#L3462) |
| AWL | Siemens AWL manual ENT/PUSH (push accumulators), POP/LEAVE below | full | [lib/AwlGen/AwlGen.cpp:3465](../lib/AwlGen/AwlGen.cpp#L3465) |
| AWL | Siemens AWL manual POP/LEAVE (pop accumulators) | full | [lib/AwlGen/AwlGen.cpp:3475](../lib/AwlGen/AwlGen.cpp#L3475) |
| AWL | Siemens AWL manual INC/DEC (ACCU1 +/- n) | full | [lib/AwlGen/AwlGen.cpp:3486](../lib/AwlGen/AwlGen.cpp#L3486) |
| AWL | Siemens AWL manual INC/DEC n — byte op on ACCU1 (awlsim setByte) | full | [lib/AwlGen/AwlGen.cpp:3487](../lib/AwlGen/AwlGen.cpp#L3487) |
| AWL | Siemens AWL manual "Value conversions" ITD/DTI — ITD sign-extends the low word of ACCU1 to a dword (awlsim setDWord(getSignedWord)) | full | [lib/AwlGen/AwlGen.cpp:3508](../lib/AwlGen/AwlGen.cpp#L3508) |
| AWL | Siemens AWL manual INVI/INVD (one's complement) — INVI inverts the low word keeping the high word (awlsim setWord), INVD inverts the full dword | full | [lib/AwlGen/AwlGen.cpp:3518](../lib/AwlGen/AwlGen.cpp#L3518) |
| AWL | Siemens AWL manual MOD (remainder) | full | [lib/AwlGen/AwlGen.cpp:3537](../lib/AwlGen/AwlGen.cpp#L3537) |
| AWL | Siemens AWL manual TAD (dword accumulator exchange) | full | [lib/AwlGen/AwlGen.cpp:3562](../lib/AwlGen/AwlGen.cpp#L3562) |
| AWL | Siemens AWL manual TAD (swapEndianDWord(ACCU1): reverse all four bytes — awlsim insn_TAD) | full | [lib/AwlGen/AwlGen.cpp:3563](../lib/AwlGen/AwlGen.cpp#L3563) |
| AWL | IEC 61131-3 STL X/XN (XOR RLO with bit) | full | [lib/AwlGen/AwlGen.cpp:3605](../lib/AwlGen/AwlGen.cpp#L3605) |
| AWL | Siemens AWL manual SPL (jump to label via ACCU1) | full | [lib/AwlGen/AwlGen.cpp:3633](../lib/AwlGen/AwlGen.cpp#L3633) |
| AWL | IEC 61131-3 STL conditional jumps JNB/JBI/SPBNB/SPBI/SPBIN (with BR save) | full | [lib/AwlGen/AwlGen.cpp:3667](../lib/AwlGen/AwlGen.cpp#L3667) |
| AWL | IEC 61131-3 STL conditional jumps JB/JBI/JC (RLO/BR conditions) | full | [lib/AwlGen/AwlGen.cpp:3671](../lib/AwlGen/AwlGen.cpp#L3671) |
| AWL | IEC 61131-3 STL LOOP (decrement ACCU1, jump if != 0) | full | [lib/AwlGen/AwlGen.cpp:3674](../lib/AwlGen/AwlGen.cpp#L3674) |
| AWL | IEC 61131-3 STL zero/plus/minus jumps (SPZ/SPPZ/SPMZ/SPS/SPP/SPM) | full | [lib/AwlGen/AwlGen.cpp:3706](../lib/AwlGen/AwlGen.cpp#L3706) |
| AWL | Siemens AWL manual "Value conversions" BTI/BTD (BCD -> binary); | full | [lib/AwlGen/AwlGen.cpp:3741](../lib/AwlGen/AwlGen.cpp#L3741) |
| AWL | Siemens AWL manual "Value conversions" ITB/DTB (binary -> BCD); | full | [lib/AwlGen/AwlGen.cpp:3775](../lib/AwlGen/AwlGen.cpp#L3775) |
| AWL | Siemens AWL manual TAW (swap the two bytes of ACCU1's low word — awlsim insn_TAW: accu1 = (accu1 & 0xFFFF0000) | swapEndianWord(accu1 & 0xFFFF)) | full | [lib/AwlGen/AwlGen.cpp:3816](../lib/AwlGen/AwlGen.cpp#L3816) |
| AWL | awlsim test directive __ASSERT<cmp> (value oracles, runtime scl_rt_assert_cmp[_r]) | full | [lib/AwlGen/AwlGen.cpp:3839](../lib/AwlGen/AwlGen.cpp#L3839) |
| AWL | IEC 61131-3 STL block calls (CALL/UC/CC, incl. parameterized CALL and indirect UC FC [MW x]), see parseCall | full | [lib/AwlGen/AwlGen.cpp:3842](../lib/AwlGen/AwlGen.cpp#L3842) |
| AWL | Siemens AWL manual "Indirect addressing" ([AR1, P#x.y] area-relative, memory-indirect [LD n]/[#var] base = operand value, T/Z [..] = block number) | full | [lib/AwlGen/AwlGen.cpp:4272](../lib/AwlGen/AwlGen.cpp#L4272) |
| AWL | IEC 61131-3 STL / Siemens AWL manual "Addressing modes": absolute (M 10.0, DB1.DBW 4), indirect ([AR1, P#x.y], [LD n], [#var]), symbolic, P# pointer literals | full | [lib/AwlGen/AwlGen.cpp:4424](../lib/AwlGen/AwlGen.cpp#L4424) |
| AWL | Siemens AWL manual symbolic DB members (DBn.SYMBOL...) resolved through the DATA_BLOCK layout (approx: a member missing from a declared DB resolves to DBW 0/DBX 0.0 of that DB) | approx | [lib/AwlGen/AwlGen.cpp:6262](../lib/AwlGen/AwlGen.cpp#L6262) |
| AWL | Siemens AWL manual symbolic DB members of undeclared DBs, auto-assigned offsets in order of first use (awlsim symbol tables) -> autoDbCursor branch of resolveDbMember (approx: DB contents are runtime memory, so the synthetic offset only fixes the address, not the value) | approx | [lib/AwlGen/AwlGen.cpp:6281](../lib/AwlGen/AwlGen.cpp#L6281) |
| AWL | Siemens AWL manual "Calling blocks" (CALL/UC/CC with parameter lists := / =>, FB instance DB, multi-instance via AR2) | full | [lib/AwlGen/AwlGen.cpp:6932](../lib/AwlGen/AwlGen.cpp#L6932) |
| AWL | Siemens AWL manual "Addressing": absolute memory operand -> !scl.direct<area, db, byte, bit, width> (see docs/design/memory-areas.md) | full | [include/sclang/AwlDialect/AwlOps.td:51](../include/sclang/AwlDialect/AwlOps.td#L51) |
| AWL | IEC 61131-3 STL load (L) / status-word reads -> awl.load | full | [include/sclang/AwlDialect/AwlOps.td:94](../include/sclang/AwlDialect/AwlOps.td#L94) |
| AWL | IEC 61131-3 STL transfer (T) / bit set-reset (S/R) -> awl.store | full | [include/sclang/AwlDialect/AwlOps.td:107](../include/sclang/AwlDialect/AwlOps.td#L107) |
| AWL | Siemens AWL manual shift/rotate (SLW/SRW/SSI/SSD/SLD/SRD/RLD/RRD/RLDA/RRDA) -> awl.shift | full | [include/sclang/AwlDialect/AwlOps.td:197](../include/sclang/AwlDialect/AwlOps.td#L197) |
| AWL | Siemens AWL manual real arithmetic (ABS/NEGR/SIN/...) -> awl.real_math | approx | [include/sclang/AwlDialect/AwlOps.td:207](../include/sclang/AwlDialect/AwlOps.td#L207) |
| AWL | IEC 61131-3 STL block call (CALL/UC/CC) -> awl.call (FC VL params, FB instance DB, SFC/SFB runtime) | full | [include/sclang/AwlDialect/AwlOps.td:254](../include/sclang/AwlDialect/AwlOps.td#L254) |
| AWL | Siemens AWL manual "Accumulator instructions" (ACCU1..4): runtime | full | [include/sclang/AwlDialect/AwlOps.td:303](../include/sclang/AwlDialect/AwlOps.td#L303) |
| AWL | awlsim test oracle __ASSERT<cmp> -> awl.assert/assert_r -> scl_rt_assert_cmp[_r] | full | [include/sclang/AwlDialect/AwlOps.td:395](../include/sclang/AwlDialect/AwlOps.td#L395) |
| AWL | awlsim __FEATURE t [, v] -> awl.feature -> scl_rt_feature | full | [include/sclang/AwlDialect/AwlOps.td:446](../include/sclang/AwlDialect/AwlOps.td#L446) |
| AWL | Siemens AWL manual AUF/OPN DB n -> awl.set_db / set_dbv / set_div (DB/DI registers, ABI v5) | full | [include/sclang/AwlDialect/AwlOps.td:465](../include/sclang/AwlDialect/AwlOps.td#L465) |
| AWL | Siemens AWL manual LARn/TARn -> awl.ar_set/ar_get/ar_raw (packed internal AR, raw S7 on TAR to memory) | full | [include/sclang/AwlDialect/AwlOps.td:505](../include/sclang/AwlDialect/AwlOps.td#L505) |
| AWL | Siemens AWL manual indirect addressing [ARn, P#x.y] -> awl.indirect (AR + bit offset) | full | [include/sclang/AwlDialect/AwlOps.td:542](../include/sclang/AwlDialect/AwlOps.td#L542) |
| AWL | Siemens AWL manual indirect load (L B/W/D [ARn, P#x.y]) -> awl.load_indirect | full | [include/sclang/AwlDialect/AwlOps.td:551](../include/sclang/AwlDialect/AwlOps.td#L551) |
| AWL | Siemens AWL manual indirect store (T B/W/D [ARn, P#x.y]) -> awl.store_indirect | full | [include/sclang/AwlDialect/AwlOps.td:560](../include/sclang/AwlDialect/AwlOps.td#L560) |
| SCL | IEC 61131-3 ST VAR string initializer (VAR s : STRING := | full | [lib/SclGen/MLIRGen.cpp:881](../lib/SclGen/MLIRGen.cpp#L881) |
| SCL | IEC 61131-3 array/struct initializers | full | [lib/SclGen/MLIRGen.cpp:888](../lib/SclGen/MLIRGen.cpp#L888) |
| SCL | IEC 61131-3 ST organization block (OB) -> void function with scan-cycle hooks (ADR 002) | full | [lib/SclGen/MLIRGen.cpp:1067](../lib/SclGen/MLIRGen.cpp#L1067) |
| ST | IEC 61131-3 CONFIGURATION (ed.3 6.8): VAR_GLOBAL becomes a | full | [lib/SclGen/MLIRGen.cpp:1333](../lib/SclGen/MLIRGen.cpp#L1333) |
| SCL | IEC 61131-3 ST statement list: assignment, call, IF/ELSIF/ELSE, | full | [lib/SclGen/MLIRGen.cpp:1539](../lib/SclGen/MLIRGen.cpp#L1539) |
| SCL | IEC 61131-3 ST assignment (:=), incl. string/array copies -> see mlirGenStringAssignment and the array-copy path below | full | [lib/SclGen/MLIRGen.cpp:1873](../lib/SclGen/MLIRGen.cpp#L1873) |
| ST | IEC 61131-3 ed.3 6.4.4.1: a real literal that is not exactly | full | [lib/SclGen/MLIRGen.cpp:2266](../lib/SclGen/MLIRGen.cpp#L2266) |
| SCL | IEC 61131-3 ed.3 table 30 date/time arithmetic (DATE - DATE -> TIME, DT +/- TIME -> DT, DT - DT -> TIME), DT_TO_DATE / DT_TO_TOD conversions and DT_YEAR..DT_WEEKDAY field extractors -> mlirGenDateTimeArith / bcdDtToMs / msToBcdDt / mlirGenDtField (DATE = days since 1990-01-01 u16; DT = 8 big-endian BCD bytes in an i64, ADR-001; civil-calendar math via Hinnant's algorithms; BCD byte extraction is floor-div based because the DT sign bit is set for years 1990-1999) | full | [lib/SclGen/MLIRGen.cpp:3271](../lib/SclGen/MLIRGen.cpp#L3271) |
| SCL | IEC 61131-3 ST type conversions (INT_TO_WORD, CHAR_TO_INT, ...) -> mlirGenBuiltinConversion; string functions (LEN/CONCAT/...) -> mlirGenStringBuiltin | full | [lib/SclGen/MLIRGen.cpp:3696](../lib/SclGen/MLIRGen.cpp#L3696) |
| SCL | IEC 61131-3 ST string functions LEN/LEFT/RIGHT/MID/FIND/CONCAT/DELETE/INSERT/REPLACE and <type>_TO_STRING/STRING_TO_<type> -> mlirGenStringBuiltin (approx: flat zero-padded buffers with C-string length; STRING_TO_REAL/REAL_TO_STRING/VAL_STRG remain external stubs) | approx | [lib/SclGen/MLIRGen.cpp:3812](../lib/SclGen/MLIRGen.cpp#L3812) |
| SCL | IEC 61131-3 ST standard functions (ed.3 table 32/33): | full | [lib/SclGen/MLIRGen.cpp:4399](../lib/SclGen/MLIRGen.cpp#L4399) |
| SCL | Test-support builtins (awlsim-style oracles in SCL form, | full | [lib/SclGen/MLIRGen.cpp:4816](../lib/SclGen/MLIRGen.cpp#L4816) |
| AWL | IEC 61131-3 STL memory areas (P/Q/M/DB/DI/L/T/C, big-endian S7 storage — docs/decisions/001) -> scl_rt_read/read_bit (C ABI, docs/decisions/002) | full | [lib/Runtime/SimRuntime.cpp:262](../lib/Runtime/SimRuntime.cpp#L262) |
| AWL | IEC 61131-3 STL memory areas, stores -> scl_rt_write/write_bit (C ABI) | full | [lib/Runtime/SimRuntime.cpp:312](../lib/Runtime/SimRuntime.cpp#L312) |
| AWL | Siemens AWL manual local data stack (L/VL frames per block call, incl. FC param area and 256-depth guard) -> scl_rt_l_push/l_pop/param_in/param_out | full | [lib/Runtime/SimRuntime.cpp:566](../lib/Runtime/SimRuntime.cpp#L566) |
| AWL | Siemens AWL manual address registers AR1/AR2 -> scl_rt_ar_get/ar_set/ar_add/ar_raw (packed internal form) | full | [lib/Runtime/SimRuntime.cpp:1188](../lib/Runtime/SimRuntime.cpp#L1188) |
| AWL | Siemens AWL manual indirect addressing -> scl_rt_read_indirect/write_indirect (raw S7 <-> packed decode) | full | [lib/Runtime/SimRuntime.cpp:1250](../lib/Runtime/SimRuntime.cpp#L1250) |
| AWL | Siemens AWL manual system functions SFC/SFB -> scl_rt_sfc/scl_rt_sfb (runtime-staged params in the marker area) | full | [lib/Runtime/SimRuntime.cpp:1290](../lib/Runtime/SimRuntime.cpp#L1290) |
| AWL | Siemens AWL manual status word (STW) cleared at OB entry (awlsim statusWord.reset) -> scl_rt_ob_start | full | [lib/Runtime/SimRuntime.cpp:2031](../lib/Runtime/SimRuntime.cpp#L2031) |
| AWL | awlsim __FEATURE t [, v] -> scl_rt_feature (nrAccus, OB-temp presets) | full | [lib/Runtime/SimRuntime.cpp:2070](../lib/Runtime/SimRuntime.cpp#L2070) |
| ST | IEC 61131-3 ST STRING outputs observed by trace tests: the | full | [lib/Runtime/SimRuntime.cpp:2143](../lib/Runtime/SimRuntime.cpp#L2143) |
| ST | IEC 61131-3 LREAL (binary64) unary math functions (ed.3 table 33): | full | [lib/Runtime/SimRuntime.cpp:2173](../lib/Runtime/SimRuntime.cpp#L2173) |
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

Current pass rates: see [conformance.md](conformance.md#current-results-llvm-23-build-this-writing).
