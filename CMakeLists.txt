# Build SCLANG within the LLVM source tree
# use `sclang` within LLVM_EXTERNAL_PROJECTS

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
  include/sclang/Dialect.h
  include/sclang/MLIRGen.h
  include/sclang/Ops.td
  sclang.cpp
)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})
target_link_libraries(sclang
  PRIVATE
    SCLPARSE
    SCLIR
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffects
    MLIRTransforms
)

add_subdirectory(include)
add_subdirectory(parser)
add_subdirectory(mlir)
add_subdirectory(test)
add_subdirectory(unittests)

