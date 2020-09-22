# Build SCLANG within the LLVM source tree
# use `sclang` within LLVM_EXTERNAL_PROJECTS

add_llvm_executable(sclang-gen
  sclang-gen.cpp

  DEPENDS
  SclOpsIncGen
  SclDialect
  SclGen
  SclTransforms
)
set_target_properties(sclang-gen PROPERTIES FOLDER "Sclang")

include_directories(${SCLANG_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(sclang-gen
  PRIVATE
    ${dialect_libs}
    SclDialect
    SclGen
    SclTransforms
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
)
