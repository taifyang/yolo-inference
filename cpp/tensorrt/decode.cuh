/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 21:21:01
 * @FilePath: \cpp\tensorrt\decode.cuh
 * @Description: cuda post-processing decoding head file for YOLO algorithm
 */

#pragma once

#include <cuda_runtime.h>
#include "yolo_tensorrt.h"

#define GPU_BLOCK_THREADS  1024
#define NUM_BOX_ELEMENT 7

/**
 * @description: cuda post-processing decoding kernel
 * @return {*}
 */
void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold,
	float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream, Algo_Type algo_type);

/**
 * @description: cuda NMS kernel
 * @return {*}
 */
void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
