/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-06-30 20:05:54
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
 * @description:                            cuda pre-processing decoding kernel
 * @param {uint8_t*} src		            source image
 * @param {int} src_width                   width of source image
 * @param {int} src_height		            height of source image
 * @param {float*} dst			            destination image
 * @param {int} dst_width		            width of destination image
 * @param {int} dst_height		            height of destination image
 * @param {float*} affine_matrix			affine matrix
 * @param {float*} affine_matrix_inverse    inverse of affine_matrix
 * @param {cudaStream_t} stream		        cuda stream
 * @return {*}
 */
void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, float* affine_matrix_inverse, cudaStream_t stream);

