add_llvm_library(SCLPARSE
  AST.cpp
  Lexer.cpp
)

include_directories(${SCLANG_MAIN_INCLUDE_DIR})
