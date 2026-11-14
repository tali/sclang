add_llvm_library(AwlDialect
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlDialect/Dialect.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlDialect/AwlBase.td
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlDialect/AwlOps.td
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlDialect/AwlOps.h
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlDialect/Printer.h
  Dialect.cpp
  AwlOps.cpp
  Printer.cpp
  DEPENDS AwlOpsIncGen SclOpsIncGen mlir-headers
)
set_target_properties(AwlDialect PROPERTIES FOLDER "Sclang")
