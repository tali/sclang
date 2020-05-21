add_llvm_library(SclDialect
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/Dialect.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/Ops.td
  Dialect.cpp
  DEPENDS mlir-headers
)
set_target_properties(SclDialect PROPERTIES FOLDER "Sclang")
