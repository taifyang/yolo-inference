/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-06-30 20:08:12
 * @FilePath: \cpp\tensorrt\decode.cuh
 * @Description: cuda post-processing decoding head file for YOLO algorithm
 */

#pragma once

#include <cuda_runtime.h>
#include "yolo_tensorrt.h"

#define GPU_BLOCK_THREADS  1024

/**
 * @description: 							cuda post-processing decoding 
 * @param {float* } 						input predict array
 * @param {int} num_bboxes					number of bboxes
 * @param {int} num_classes					number of classes
 * @param {float} confidence_threshold		confidence threshold
 * @param {float* } inverse_affine_matrix	inverse of affine_matrix
 * @param {float* } parray					output array
 * @param {int} max_objects					max number of objects
 * @param {int} num_box_element				number of box element
 * @param {cudaStream_t} stream				cuda stream
 * @param {Algo_Type} algo_type				algorithm type
 * @param {Task_Type} task_type				task type
 * @return {*}
 */
void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* inverse_affine_matrix, 
							float* parray, int max_objects, int num_box_element, cudaStream_t stream, Algo_Type algo_type, Task_Type task_type);

/**
 * @description: 					cuda NMS kernel
 * @param {float* } 				input predict array
 * @param {float} nms_threshold		NMS threshold
 * @param {int} max_objects			max number of objects
 * @param {int} num_box_element		number of box element
 * @param {cudaStream_t} stream		cuda stream
 * @return {*}
 */
void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, int num_box_element, cudaStream_t stream);

/**
 * @description: 					decode single mask
 * @param {float} left				left of mask in image
 * @param {float} top				top of mask in image
 * @param {float*} mask_weights		weights of mask
 * @param {float*} mask_predict		predict of mask
 * @param {int} mask_width			width of mask
 * @param {int} mask_height			height of mask
 * @param {unsigned char} mask_out	result of decode mask
 * @param {int} mask_dim			dim of mask
 * @param {int} out_width			result of output width
 * @param {int} out_height			result of output height
 * @param {cudaStream_t} stream		cuda stream
 * @return {*}
 */
void decode_single_mask(float left, float top, float* mask_weights, float* mask_predict,
						int mask_width, int mask_height, unsigned char* mask_out,
                        int mask_dim, int out_width, int out_height, cudaStream_t stream);
