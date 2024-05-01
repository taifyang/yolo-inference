cmake_minimum_required(VERSION 3.23)
project(yolov5)
set(CMAKE_CXX_STANDARD 14)

aux_source_directory(. SRCS)

set(LIBTORCH_INCLUDE_DIRS D:/software/libtorch/include D:/software/libtorch/include/torch/csrc/api/include)
set(LIBTORCH_LIBRARY_DIRS D:/software/libtorch/lib)
include_directories(${LIBTORCH_INCLUDE_DIRS})
link_directories(${LIBTORCH_LIBRARY_DIRS})

set(ORT_INCLUDE_DIRS D:/software/onnxruntime-win-x64-gpu-1.12.1/include)
set(ORT_LIBRARY_DIRS D:/software/onnxruntime-win-x64-gpu-1.12.1/lib)
include_directories(${ORT_INCLUDE_DIRS})
link_directories(${ORT_LIBRARY_DIRS})

set(OPENCV_INCLUDE_DIRS D:/software/OpenCV4.5.5/opencv/build/include)
set(OPENCV_LIBRARY_DIRS D:/software/OpenCV4.5.5/opencv/build/x64/vc16/lib)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

set(OPENVINO_INCLUDE_DIRS D:/software/OpenVINO/openvino_2022.3.1.9227/runtime/include)
set(OPENVINO_LIBRARY_DIRS D:/software/OpenVINO/openvino_2022.3.1.9227/runtime/lib/intel64/Release)
include_directories(${OPENVINO_INCLUDE_DIRS})
link_directories(${OPENVINO_LIBRARY_DIRS})

set(TRT_INCLUDE_DIRS D:/software/TensorRT-8.2.1.8/include)
set(TRT_LIBRARY_DIRS D:/software/TensorRT-8.2.1.8/lib)
include_directories(${TRT_INCLUDE_DIRS})
link_directories(${TRT_LIBRARY_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_executable(yolov5 ${SRCS})
target_link_libraries(yolov5 ${CUDA_LIBRARIES} ${LIBTORCH_LIBRARY_DIRS}/*.lib ${ORT_LIBRARY_DIRS}/*.lib ${OPENCV_LIBRARY_DIRS}/*.lib ${OPENVINO_LIBRARY_DIRS}/*.lib ${TRT_LIBRARY_DIRS}/*.lib) 

