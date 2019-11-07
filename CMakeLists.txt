add_custom_target(Sclang)
set_target_properties(Sclang PROPERTIES FOLDER Examples)


set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_example(sclang
  include/sclang/AST.h
  include/sclang/Lexer.h
  include/sclang/Parser.h
  sclang.cpp
  parser/AST.cpp
  )
include_directories(include/)
