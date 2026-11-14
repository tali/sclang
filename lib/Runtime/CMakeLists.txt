add_llvm_library(SclRuntime
  ${SCLANG_MAIN_INCLUDE_DIR}/sclang/Runtime/Runtime.h
  SimRuntime.cpp
  DEPENDS mlir-headers
)
set_target_properties(SclRuntime PROPERTIES FOLDER "Sclang")
