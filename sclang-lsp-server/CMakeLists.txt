# sclang-lsp-server: the standard mlir-lsp-server with the scl/awl dialects
# registered (serves .mlir documents; for SCL/AWL source use `sclang --lsp`).
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(sclang-lsp-server
  sclang-lsp-server.cpp

  DEPENDS
  SclOpsIncGen
  AwlOpsIncGen
)
set_target_properties(sclang-lsp-server PROPERTIES FOLDER "Sclang")

target_link_libraries(sclang-lsp-server PRIVATE
  ${dialect_libs}
  SclDialect
  AwlDialect
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupportLSP
)
