# SCLANG

`sclang` is an work-in-progress SCL compiler using LLVMs [MLIR][MLIR] infrastructure.

Instead of generating AWL or machine code for S7, it directly targets the host computer.
Goal is to provide a fast compiler and runtime to run unit and integration tests for large SCL projects.

Current status:
parsing SCL files works quite well. Code generation covers functions, expressions,
control flow (if/elsif/else, for, while, repeat-until, continue/exit/return) and
array variables, and the compiler can JIT-execute the result on the host.
Function blocks, organization blocks, CASE/GOTO, strings and user-defined types
are parsed but not yet code-generated.

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

# REFERENCES

- SIEMENS, *Structured Control Language (SCL) for S7-300/S7-400 Programming*
  (the language this compiler implements):
  <https://cache.industry.siemens.com/dl/files/188/1137188/att_27471/v1/SCLV4_e.pdf>
- IEC 61131-3, *Programmable controllers – Programming languages*
  (the international standard SCL is based on): <https://webstore.iec.ch/publication/4552>
