set(LLVM_TARGET_DEFINITIONS SclOps.td)
mlir_tablegen(SclOps.h.inc -gen-op-decls)
mlir_tablegen(SclOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(SclOpsIncGen)
