//===- LangMode.h - SCL source-language mode ---------------------*- C++ -*-===//
//
// Source-language mode of the SCL frontend (ADR 004, docs/decisions/004):
// S7 = Siemens SCL dialect (default: % directs, TIMER/COUNTER, EN/ENO,
// "DB".member, STRING[n]); IEC = IEC 61131-3 ST surface (CONFIGURATION/
// RESOURCE/TASK, VAR_GLOBAL/VAR_EXTERNAL, LREAL, bare STRING). The mode is
// threaded through Lexer -> Parser -> ModuleAST -> MLIRGen; the scl dialect
// itself stays mode-agnostic.
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_LANGMODE_H_
#define SCLANG_LANGMODE_H_

namespace sclang {

enum class LangMode {
  S7,  // Siemens SCL (S7-300/400/1200/1500 TIA dialect) — default
  IEC, // IEC 61131-3 ST surface (`--std=iec`, `-x=st`)
};

} // namespace sclang

#endif // SCLANG_LANGMODE_H_
