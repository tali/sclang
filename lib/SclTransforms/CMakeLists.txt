add_llvm_library(SclTransforms
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclTransforms/Passes.h
  LowerToStd.cpp
  DEPENDS AwlOpsIncGen mlir-headers
)
set_target_properties(SclTransforms PROPERTIES FOLDER "Sclang")
