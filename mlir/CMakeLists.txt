add_llvm_library(SCLIR
  Dialect.cpp
  MLIRGen.cpp
)

add_dependencies(SCLIR SclangOpsIncGen)
include_directories(${SCLANG_MAIN_INCLUDE_DIR})
include_directories(${SCLANG_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})
target_link_libraries(SCLIR
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms
)
