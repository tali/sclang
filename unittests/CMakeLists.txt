add_custom_target(SclangUnitTests)
set_target_properties(SclangUnitTests PROPERTIES FOLDER "Sclang Tests")

function(add_sclang_unittest test_dirname)
  add_unittest(SclangUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Lexer)
