add_sclang_unittest(LexerTests
  LexerTest.cpp
)
include_directories(${SCLANG_SOURCE_DIR})
target_link_libraries(LexerTests
  PRIVATE
    SCLPARSE
)
