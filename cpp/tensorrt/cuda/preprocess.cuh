/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-19 18:26:01
 * @Description: head file for cuda pre-processing decoding
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

void cuda_centercrop(uint8_t* src, uint8_t* crop, uint8_t* dst, cv::Size src_size, cv::Size dst_size, int channel);

void cuda_normalize(uint8_t* src, float* dst, cv::Size src_size, Algo_Type algo_type);

/**
 * @brief  							cuda resize image use linear method
 * @param {uint8_t*} src      		input image
 * @param {uint8_t*} dst      		output image
 * @param {cv::Size} src_size		input image size
 * @param {cv::Size} dst_size 		output image size
 * @param {int} channel		        channel of image
 */
void cuda_resize_linear(uint8_t* src, uint8_t* dst, cv::Size src_size, cv::Size dst_size, int channel);

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
 * @return {*}
 */
void cuda_preprocess_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, float* affine_matrix_inverse);
