configure_lit_site_cfg(
    ${CMAKE_CURRENT_LIST_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_LIST_DIR}/lit.cfg.py
)


set (HIFIVE_TEST_DEPENDS
    FileCheck count not
    mlir-opt
    mlir-cpu-runner
    # tutorial-opt
)

add_lit_testsuite(check-hifive "Running hifive regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${HIFIVE_TEST_DEPENDS}
)