//https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/src/preprocess.h

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "yolo_tensorrt.h"

struct AffineMatrix
{
    float value[6];
};

void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, cudaStream_t stream);

