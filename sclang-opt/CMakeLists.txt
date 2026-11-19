# sclang-opt: mlir-opt-style driver with the scl/awl dialects and the sclang
# lowering passes registered (see sclang-opt.cpp).

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sclang-opt
  sclang-opt.cpp

  DEPENDS
  SclOpsIncGen
  AwlOpsIncGen
  SclTransformsPassesIncGen
)
set_target_properties(sclang-opt PROPERTIES FOLDER "Sclang")

target_link_libraries(sclang-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  SclDialect
  AwlDialect
  SclTransforms
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
)
