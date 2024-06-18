/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 23:32:19
 * @FilePath: \cpp\tensorrt\preprocess.cuh
 * @Description: 前处理的cuda实现头文件，见https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/src/preprocess.h
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "yolo_tensorrt.h"

/**
 * @description: 仿射变换矩阵
 */
struct AffineMatrix
{
    float value[6];
};

/**
 * @description: 前处理的cuda实现
 * @return {*}
 */
void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, cudaStream_t stream);

