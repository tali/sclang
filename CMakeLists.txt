add_custom_target(Sclang)
set_target_properties(Sclang PROPERTIES FOLDER Examples)

add_subdirectory(include)
add_subdirectory(test)


set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_example(sclang
  include/sclang/AST.h
  include/sclang/Dialect.h
  include/sclang/MLIRGen.h
  include/sclang/Lexer.h
  include/sclang/Parser.h
  include/sclang/Ops.td
  sclang.cpp
  mlir/Dialect.cpp
  mlir/MLIRGen.cpp
  parser/AST.cpp
  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
add_dependencies(sclang SclangOpsIncGen)
target_link_libraries(sclang
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms)
