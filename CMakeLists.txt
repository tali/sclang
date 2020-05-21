# Build SCLANG within the LLVM source tree
# use `sclang` within LLVM_EXTERNAL_PROJECTS

set(SCLANG_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCLANG_MAIN_INCLUDE_DIR ${SCLANG_MAIN_SRC_DIR}/include)
set(SCLANG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include )
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../mlir/include )
set(MLIR_TABLEGEN_EXE mlir-tblgen)

include_directories(${SCLANG_INCLUDE_DIR})
include_directories(${SCLANG_MAIN_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_MAIN_SRC_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(sclang)
add_subdirectory(sclang-gen)
add_subdirectory(test)
