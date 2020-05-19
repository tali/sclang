# Build SCLANG within the LLVM source tree
# use `sclang` within LLVM_EXTERNAL_PROJECTS

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(SCLANG_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCLANG_MAIN_INCLUDE_DIR ${SCLAG_MAIN_SRC_DIR}/include)
set(SCLANG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include )
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../mlir/include )
set(MLIR_TABLEGEN_EXE mlir-tblgen)

add_llvm_executable(sclang
  include/sclang/AST.h
  include/sclang/Lexer.h
  include/sclang/Parser.h
  include/sclang/Passes.h
  include/sclang/Dialect.h
  include/sclang/MLIRGen.h
  include/sclang/Ops.td
  parser/AST.cpp
  parser/Lexer.cpp
  mlir/Dialect.cpp
  mlir/LowerToLLVM.cpp
  mlir/LowerToSCF.cpp
  mlir/LowerToStd.cpp
  mlir/MLIRGen.cpp
  sclang.cpp

  DEPENDS
  SclangOpsIncGen
)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(sclang
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
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

add_subdirectory(include)
add_subdirectory(test)

