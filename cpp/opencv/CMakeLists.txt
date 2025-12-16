aux_source_directory(. OPENCV_SRCS)
add_library(opencv_backend STATIC ${OPENCV_SRCS})