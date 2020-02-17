#include "sclang/Lexer.h"

#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace sclang;

namespace {

TEST(Lexer, RealNumberLiteral) {
  llvm::StringRef source = "1.0";
  LexerBuffer lexer(source.begin(), source.end(), "unittest");
  EXPECT_EQ(lexer.getNextToken(), tok_real_number_literal);
  EXPECT_EQ(lexer.getRealNumberValue(), 1.0);
}

TEST(Lexer, IntegerLiteral) {
  llvm::StringRef source = "10";
  LexerBuffer lexer(source.begin(), source.end(), "unittest");
  EXPECT_EQ(lexer.getNextToken(), tok_integer_literal);
  EXPECT_EQ(lexer.getIntegerValue(), 10);
}

} // namespace
