cmake_minimum_required(VERSION 3.16)

project(yolo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")

aux_source_directory(. SRCS)

option(SAVE_RESULT "save result" OFF)

option(SHOW_RESULT "show result" OFF)


IF (CMAKE_SYSTEM_NAME MATCHES "Windows")   
    set(LIBTORCH_INCLUDE_DIRS D:/software/libtorch/include D:/software/libtorch/include/torch/csrc/api/include)
    set(LIBTORCH_LIBRARY_DIRS D:/software/libtorch/lib)
    include_directories(${LIBTORCH_INCLUDE_DIRS})
    link_directories(${LIBTORCH_LIBRARY_DIRS})

    set(ORT_INCLUDE_DIRS D:/software/onnxruntime-win-x64-gpu-1.12.1/include)
    set(ORT_LIBRARY_DIRS D:/software/onnxruntime-win-x64-gpu-1.12.1/lib)
    include_directories(${ORT_INCLUDE_DIRS})
    link_directories(${ORT_LIBRARY_DIRS})

    set(OPENCV_INCLUDE_DIRS D:/software/OpenCV4.9.0/build/install/include)
    set(OPENCV_LIBRARY_DIRS D:/software/OpenCV4.9.0/build/install/x64/vc16/lib)
    include_directories(${OPENCV_INCLUDE_DIRS})
    link_directories(${OPENCV_LIBRARY_DIRS})

    set(OPENVINO_INCLUDE_DIRS D:/software/OpenVINO2024.1/runtime/include)
    set(OPENVINO_LIBRARY_DIRS D:/software/OpenVINO2024.1/runtime/lib/intel64/Release)
    include_directories(${OPENVINO_INCLUDE_DIRS})
    link_directories(${OPENVINO_LIBRARY_DIRS})

    set(TRT_INCLUDE_DIRS D:/software/TensorRT-8.2.1.8/include)
    set(TRT_LIBRARY_DIRS D:/software/TensorRT-8.2.1.8/lib)
    include_directories(${TRT_INCLUDE_DIRS})
    link_directories(${TRT_LIBRARY_DIRS})

    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIB_PATH})

    if(SAVE_RESULT)
        add_compile_options(-D_SAVE_RESULT)
    ENDIF()

    if(SHOW_RESULT)
        add_compile_options(-D_SHOW_RESULT)
    ENDIF()

    add_executable(yolo ${SRCS})
    target_link_libraries(yolo CUDA::cudart ${LIBTORCH_LIBRARY_DIRS}/*.lib ${ORT_LIBRARY_DIRS}/*.lib ${OPENCV_LIBRARY_DIRS}/*.lib ${OPENVINO_LIBRARY_DIRS}/*.lib ${TRT_LIBRARY_DIRS}/*.lib) 

ENDIF()


IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCV2_INCLUDE_DIRS})

    set(Torch_DIR /home/libtorch2.2.2_cuda11.8/share/cmake/Torch)
    find_package(Torch REQUIRED)
    include_directories(${TORCH_INCLUDE_DIRS})

    set(ORT_INCLUDE_DIRS /home/onnxruntime-linux-x64-gpu-1.12.1/include)
    set(ORT_LIBRARY_DIRS /home/onnxruntime-linux-x64-gpu-1.12.1/lib)
    include_directories(${ORT_INCLUDE_DIRS})

    set(OPENVINO_INCLUDE_DIRS /home/openvino2024.1.0/runtime/include)
    set(OPENVINO_LIBRARY_DIRS /home/openvino2024.1.0/runtime/lib/intel64)
    include_directories(${OPENVINO_INCLUDE_DIRS})

    set(TRT_INCLUDE_DIRS /home/TensorRT-8.2.1.8/include)
    set(TRT_LIBRARY_DIRS /home/TensorRT-8.2.1.8/lib)
    include_directories(${TRT_INCLUDE_DIRS})
    link_directories(${TRT_LIBRARY_DIRS})

    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    IF(SAVE_RESULT)
        add_compile_options(-D_SAVE_RESULT)
    ENDIF()

    if(SHOW_RESULT)
        add_compile_options(-D_SHOW_RESULT)
    ENDIF()
    
    cuda_add_executable(yolo ${SRCS})
    target_link_libraries(yolo ${OpenCV_LIBRARIES})  
    target_link_libraries(yolo ${TORCH_LIBRARIES})
    target_link_libraries(yolo ${ORT_LIBRARY_DIRS}/libonnxruntime.so)
    target_link_libraries(yolo ${OPENVINO_LIBRARY_DIRS}/libopenvino.so ${OPENVINO_LIBRARY_DIRS}/libopenvino_intel_cpu_plugin.so ${OPENVINO_LIBRARY_DIRS}/libopenvino_intel_gpu_plugin.so)
    target_link_libraries(yolo ${CUDA_LIBRARIES})
    target_link_libraries(yolo nvinfer)
  
ENDIF()