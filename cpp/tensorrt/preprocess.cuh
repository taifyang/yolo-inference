/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 21:26:07
 * @FilePath: \cpp\tensorrt\preprocess.cuh
 * @Description: cuda pre-processing decoding head file for YOLO algorithm
 * refer to https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/src/preprocess.h
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "yolo_tensorrt.h"

/**
 * @description: affine matrix
 */
struct AffineMatrix
{
    float value[6];
};

/**
 * @description: cuda pre-processing decoding kernel
 * @return {*}
 */
void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, cudaStream_t stream);

