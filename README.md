# SCLANG

`sclang` is an work-in-progress SCL compiler using LLVMs [MLIR][MLIR] infrastructure.

Instead of generating AWL or machine code for S7, it directly targets the host computer.
Goal is to provide a fast compiler and runtime to run unit and integration tests for large SCL projects.

Current status:
parsing SCL files works quite well, code generation only works for a very limited subset of expressions.

[MLIR]: https://mlir.llvm.org/

# BUILDING

MLIR is still in active development and Sclang is regularly updated to follow that development.
A compatible version of _llvm-project/mlir_ is provided as submodule. 
You can choose to build it using either [CMake][CMake] or [Bazel][Bazel].
Both MacOS and Linux are supported.

[CMake]: https://cmake.org/cmake/help/latest/
[Bazel]: https://bazel.build/

## Check out sources

```sh
git clone --recursive https://github.com/tali/sclang
cd sclang
```

Note:
The repository is set up so that git submodule update performs a shallow clone,
meaning it download just enough of the LLVM repository to check out the currently specified commit.
If you wish to work with the full history of the LLVM repository, you can manually "unshallow" the submodule:

```sh
cd third_party/llvm-project
git fetch --unshallow
```

## Build using CMake

```sh
cmake -B build -S third_party/llvm-project/llvm -GNinja \
	-DBUILD_SHARED_LIBS=OFF \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DLLVM_ENABLE_PROJECTS="mlir" \
	-DLLVM_EXTERNAL_PROJECTS=sclang \
	-DLLVM_EXTERNAL_SCLANG_SOURCE_DIR=$PWD \
	-DLLVM_TARGETS_TO_BUILD=X86 \
	-DCMAKE_BUILD_TYPE=Release
# alternatively use CMAKE_BUILD_TYPE=DEBUG
cmake --build build --target check-sclang
```

## Build using Bazel

```
bazel test --config=generic_clang //test
# alternatively:
bazel test --config=generic_gcc //test
```
