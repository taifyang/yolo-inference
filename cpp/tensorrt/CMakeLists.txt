file(GLOB TENSORRT_SRCS *.h *.cpp)
if(WITH_CUDA_PREPROCESS)
    list(APPEND TENSORRT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/preprocess.cuh ${CMAKE_CURRENT_SOURCE_DIR}/cuda/preprocess.cu)
    if(WITH_CUDA_POSTPROCESS)  
        list(APPEND TENSORRT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/decode.cuh ${CMAKE_CURRENT_SOURCE_DIR}/cuda/decode.cu)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(tensorrt_backend STATIC ${TENSORRT_SRCS})
target_link_libraries(tensorrt_backend PRIVATE nvinfer)