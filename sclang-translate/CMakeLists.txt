# sclang-translate: MLIR translation driver with the scl/awl dialects
# registered (--serialize/--deserialize bytecode, --mlir-to-llvmir).
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(sclang-translate
  sclang-translate.cpp

  DEPENDS
  SclOpsIncGen
  AwlOpsIncGen
)
set_target_properties(sclang-translate PROPERTIES FOLDER "Sclang")

target_link_libraries(sclang-translate PRIVATE
  ${dialect_libs}
  SclDialect
  AwlDialect
  MLIRTranslateLib
  MLIRBytecodeWriter
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRRegisterAllDialects
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMCore
  LLVMSupport
)

llvm_update_compile_flags(sclang-translate)
