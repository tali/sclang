# sclang-query: mlir-query driver with the scl/awl dialects registered.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(sclang-query
  sclang-query.cpp

  DEPENDS
  SclOpsIncGen
  AwlOpsIncGen
)
set_target_properties(sclang-query PROPERTIES FOLDER "Sclang")

target_link_libraries(sclang-query PRIVATE
  ${dialect_libs}
  SclDialect
  AwlDialect
  MLIRQueryLib
  MLIRRegisterAllDialects
  MLIRIR
  MLIRParser
  MLIRSupport
)

llvm_update_compile_flags(sclang-query)
