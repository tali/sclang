add_llvm_library(SclGen
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclGen/AST.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclGen/Lexer.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclGen/Parser.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclGen/MLIRGen.h
  AST.cpp
  Lexer.cpp
  MLIRGen.cpp
  DEPENDS mlir-headers
)
set_target_properties(SclGen PROPERTIES FOLDER "Sclang")
