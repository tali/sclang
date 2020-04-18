# SCLANG

`sclang` is an SCL compiler using LLVMs MLIR infrastructure.

# BUILDING

```
git clone https://github.com/tali/llvm-project
cd llvm-project
git clone https://github.com/tali/sclang
mkdir build
cd build
cmake -G Ninja ../llvm \
    -DLLVM_EXTERNAL_PROJECTS=sclang \
    -DLLVM_ENABLE_PROJECTS="sclang;mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --target check-sclang
```

