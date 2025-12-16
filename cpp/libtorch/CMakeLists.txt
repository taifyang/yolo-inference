aux_source_directory(. LIBTORCH_SRCS)
add_library(torch_backend STATIC ${LIBTORCH_SRCS})
target_link_libraries(torch_backend PRIVATE ${TORCH_LIBRARIES})