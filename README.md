# SCLANG

`sclang` is an SCL compiler using LLVMs MLIR infrastructure.

# BUILDING

In order to build Sclang, you need a recent MLIR.
Using the instructions below, you can first build a compatible version of LLVM and MLIR, then use that to build Sclang.

## Check out sources

```sh
git clone --recursive https://github.com/tali/sclang
cd sclang
```

## Build LLVM

```sh
cmake -G Ninja -B build.llvm -S third_party/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build.llvm
```

## Build Sclang

```sh
cmake -G Ninja -B build.sclang \
    -DMLIR_DIR=$PWD/build.llvm/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/build.llvm/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build.sclang --target check-sclang
```

