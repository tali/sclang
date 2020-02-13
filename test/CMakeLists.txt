set(SCLANG_SOURCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCLANG_BINARY_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(SCLANG_TEST_DEPENDS
  FileCheck count not
  sclang
  )

add_lit_testsuite(check-sclang "Running the SCLANG regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SCLANG_TEST_DEPENDS}
  )
set_target_properties(check-sclang PROPERTIES FOLDER "Tests")

add_lit_testsuites(SCLANG ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_TEST_DEPS}
)
