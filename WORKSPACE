# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "sclang")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

SKYLIB_VERSION = "1.0.3"

http_archive(
    name = "bazel_skylib",
    sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version=SKYLIB_VERSION),
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version=SKYLIB_VERSION),
    ],
)

maybe(
    local_repository,
    name = "llvm_bazel",
    path = "third_party/llvm-project/utils/bazel",
)

load("@llvm_bazel//:zlib.bzl", "llvm_zlib_disable")

maybe(
    llvm_zlib_disable,
    name = "llvm_zlib",
)

load("@llvm_bazel//:terminfo.bzl", "llvm_terminfo_disable")

maybe(
    llvm_terminfo_disable,
    name = "llvm_terminfo",
)

load("@llvm_bazel//:configure.bzl", "llvm_configure")

maybe(
    llvm_configure,
    name = "llvm-project",
    src_path = "third_party/llvm-project",
    src_workspace = "@sclang//:WORKSPACE",
)

http_archive(
    name = "bazel_toolchains",
    sha256 = "1adf5db506a7e3c465a26988514cfc3971af6d5b3c2218925cd6e71ee443fc3f",
    strip_prefix = "bazel-toolchains-4.0.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/4.0.0/bazel-toolchains-4.0.0.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/4.0.0/bazel-toolchains-4.0.0.tar.gz",
    ],
)

load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(name = "rbe_default")
