/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 23:28:37
 * @FilePath: \cpp\tensorrt\decode.cuh
 * @Description: yolo检测器cuda后处理解码头文件
 */

#pragma once

#include <cuda_runtime.h>
#include "yolo_tensorrt.h"

#define GPU_BLOCK_THREADS  1024
#define NUM_BOX_ELEMENT 7

/**
 * @description: 后处理解码的cuda实现
 * @return {*}
 */
void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold,
	float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream, Algo_Type algo_type);

/**
 * @description: NMS的cuda实现
 * @return {*}
 */
void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
