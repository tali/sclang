add_llvm_library(SclDialect
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/Dialect.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/SclBase.td
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/SclTypes.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclDialect/SclOps.td
  Dialect.cpp
  SclTypes.cpp
  SclOps.cpp
  DEPENDS mlir-headers
)
set_target_properties(SclDialect PROPERTIES FOLDER "Sclang")
