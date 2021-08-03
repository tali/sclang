# Test suite for Sclang

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

test_suite(
    name = "test-gen",
    tests = [
        "//test/SclGen/MLIR:lit",
        "//test/SclGen/Parse:lit",
        "//test/SclTransforms:lit",
    ],
)

test_suite(
    name = "test",
    tests = [
        "//test/SclGen/MLIR:lit",
        "//test/SclGen/Parse:lit",
        "//test/SclTransforms:lit",
        "//test/sclang:lit",
    ],
)

exports_files(["run_lit.sh"])
