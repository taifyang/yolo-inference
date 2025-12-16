aux_source_directory(. OPENVINO_SRCS)
add_library(openvino_backend STATIC ${OPENVINO_SRCS})
target_link_libraries(openvino_backend PRIVATE openvino::runtime)