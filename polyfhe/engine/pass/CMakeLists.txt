set(POLYFHE_SOURCE_FILES ${POLYFHE_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/analyze_intra_node_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calculate_memory_traffic_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calculate_smem_size_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/check_edge_overwrite_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/check_edge_same_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/check_subgraph_dependencies_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_reuse_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/extract_subgraph_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_launch_config_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lowering_ckks_to_poly_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rewrite_ntt_pass.cpp
    ${CMAKE_CURRENT_LIST_DIR}/set_block_phase_pass.cpp
)

set(POLYFHE_SOURCE_FILES ${POLYFHE_SOURCE_FILES} PARENT_SCOPE)
