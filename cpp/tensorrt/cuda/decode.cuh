/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-19 18:15:08
 * @Description: head file for cuda post-processing decoding
 */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "yolo_tensorrt.h"

#define BLOCK_SIZE  1024
#define EPS  1e-8f

/**
 * @description: 							cuda post-processing decoding 
 * @param {float* } 						input predict array
 * @param {int} num_bboxes					number of bboxes
 * @param {int} num_classes					number of classes
 * @param {float} confidence_threshold		confidence threshold
 * @param {float} score_threshold			score threshold
 * @param {float* } inverse_affine_matrix	inverse of affine_matrix
 * @param {float* } parray					output array
 * @param {int} max_objects					max number of objects
 * @param {int} num_box_element				number of box element
 * @param {Algo_Type} algo_type				algorithm type
 * @param {Task_Type} task_type				task type
 * @return {*}
 */
void cuda_decode(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float score_threshold, float* inverse_affine_matrix, float* parray, 
	int max_objects, int num_box_element, cv::Size input_size, Algo_Type algo_type, Task_Type task_type);

/**
 * @description: 					cuda NMS
 * @param {float* } 				input predict array
 * @param {float} nms_threshold		NMS threshold
 * @param {int} max_objects			max number of objects
 * @param {int} num_box_element		number of box element
 * @return {*}
 */
void cuda_nms(float* parray, float nms_threshold, int max_objects, int num_box_element);

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
 * @return {*}
 */
void cuda_decode_mask(float left, float top, float* mask_weights, float* mask_predict, int mask_width, int mask_height, uint8_t* mask_out, int mask_dim, int out_width, int out_height);

/**
 * @brief  							cuda extract col 
 * @param {const float*} input_d   	input on device
 * @param {float*} output_d  		output col on device
 * @param {int} target_col 			target col
 * @param {int} rows      			input rows
 * @param {int} cols      			input cols
 */
void cuda_extract_col(const float* input_d, float* output_d, int target_col, int rows, int cols);

/**
 * @brief 						thrust argsort
 * @param {float*} scores_d 	scores on device
 * @param {int*} sorted_idx_d	sorted indices on device
 * @param {int} num_bbox 		number of input bbox
 */
void thrust_argsort(float* scores_d, int* sorted_idx_d, int num_bbox);

/**
 * @brief  							cuda pick bbox
 * @param {const float*} ious_d   	input on device
 * @param {int*} pick_d   			picked on device
 * @param {float} threshold  		IOU threshold
 * @param {int} num_bbox   			number of input bbox
 * @param {int&} pick_size  		picked bbox size
 */
void cuda_pick_bbox(const float* ious_d, int* pick_d, float threshold, int num_bbox, int& pick_size);

/** 
 * @brief  							cuda extract rows
 * @param {const float*} input_d   	input on device
 * @param {const int*} index_d   	index on device
 * @param {float*} output_d   		output on device
 * @param {int} num_bbox   			number of input bbox
 */
void cuda_extract_rows(const float* input_d, const int* index_d, float* output_d, int num_bbox);

/**
 * @brief 							cuda compute covariance matrix
 * @param  {const float*} boxes_d  	input boxes on device
 * @param  {float*} a_d      		output a on device
 * @param  {float*} b_d      		output b on device
 * @param  {float*} c_d      		output c on device
 * @param  {int} num_bbox   		number of input bbox
 */
void cuda_compute_covariance_matrix(const float* boxes_d, float* a_d, float* b_d, float* c_d, int num_bbox);

/**
 * @brief 							cuda compute hd
 * @param {const float*} obb1_d     input obb1 on device
 * @param {const float*} obb2_d     input obb2 on device
 * @param {const float*} a1_d       input a1 on device
 * @param {const float*} b1_d       input b1 on device
 * @param {const float*} c1_d       input c1 on device
 * @param {const float*} a2_d       input a2 on device
 * @param {const float*} b2_d       input b2 on device
 * @param {const float*} c2_h       input c2 on device
 * @param {float*} hd_d        		output hd on device
 * @param {int} num_bbox   			number of input bbox
 */
void cuda_compute_hd(const float* obb1_d, const float* obb2_d,
    const float* a1_d, const float* b1_d, const float* c1_d, const float* a2_d, const float* b2_d, const float* c2_d, float* hd_d, int box_num);

/**
 * @brief  							cuda triu k1
 * @param  {float*} mat_d     		matirx on device
 * @param  {int} rows      			input matirx rows
 * @param  {int} cols      			input matirx cols
 */
void cuda_triu_k1(float* mat_d, int rows, int cols);

/**
 * @brief 							cuda regularize bbox
 * @param {float*} boxes_d 			boxes on device
 * @param {int} num_bbox   			number of input bbox
 */
void cuda_regularize_bbox(float* boxes_d, int num_bbox);

/**
 * @brief 							cuda scale boxes
 * @param {float*} boxes_d 			boxes on device
 * @param {int} num_bbox   			number of input bbox
 * @param {float} output_h 			output height
 * @param {float} output_w 			output width
 * @param {float} gain 				scale of resize
 * @param {float} pad_w 			pad of witdh
 * @param {float} pad_h 			pad of height
 */
void cuda_scale_boxes(float* boxes_d, int num_bbox, float output_w, float output_h, float gain, float pad_w, float pad_h);
