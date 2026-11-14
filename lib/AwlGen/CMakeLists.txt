add_llvm_library(AwlGen
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/AwlGen/AwlGen.h
  AwlGen.cpp
  DEPENDS AwlOpsIncGen mlir-headers
)
set_target_properties(AwlGen PROPERTIES FOLDER "Sclang")
