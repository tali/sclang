add_llvm_library(SclToStd
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/SclToStd/Passes.h
  LowerToSCF.cpp
  LowerToStd.cpp
  DEPENDS mlir-headers
)
set_target_properties(SclToStd PROPERTIES FOLDER "Sclang")
