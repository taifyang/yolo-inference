aux_source_directory(. ONNXRUNTIME_SRCS)
add_library(onnxruntime_backend STATIC ${ONNXRUNTIME_SRCS})
target_link_libraries(onnxruntime_backend PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS}/libonnxruntime.so)