//===- FrontendOptions.h - frontend-wide option globals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCLANG_FRONTEND_OPTIONS_H_
#define SCLANG_FRONTEND_OPTIONS_H_

namespace sclang {

/// Test-builtin gate (--test-builtins): when false, the special debug/test
/// builtins (__ASSERT, __DEBUG_PRINT, __BITS, __UNBITS, __STRTO_M,
/// __LOADM64, __STOREM64 in SCL/ST; __ASSERT in AWL) are diagnosed as
/// errors instead of being recognized, so normal source files cannot
/// accidentally depend on them. Set by the driver (sclang / sclang-gen)
/// from its command line; the LSP mode turns it on by default.
extern bool g_testBuiltins;

} // namespace sclang

#endif // SCLANG_FRONTEND_OPTIONS_H_
