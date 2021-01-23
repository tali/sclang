# Build SCLANG within the LLVM source tree
# use `sclang` within LLVM_EXTERNAL_PROJECTS

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

add_llvm_executable(sclang
  LowerToLLVM.cpp
  sclang.cpp

  DEPENDS
  SclOpsIncGen
  SclDialect
  SclGen
  SclTransforms
)
set_target_properties(sclang PROPERTIES FOLDER "Sclang")

include_directories(${SCLANG_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(sclang
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    SclDialect
    SclGen
    SclTransforms
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTransforms
)
