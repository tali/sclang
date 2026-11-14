load("//test:lit_test.bzl", "sclang_lit_test_suite")

sclang_lit_test_suite(
    name = "lit",
    srcs = glob(["*.scl", "*.awl"]),
    sut = "//sclang",
    tags = ["hostonly"],
)
