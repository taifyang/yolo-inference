/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-10-15 23:16:37
 * @Description: header file for YOLO tensorrt inference
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "yolo_pose.h"
#include "yolo_obb.h"
#include "yolo_depth.h"
#include "yolo_semantic.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

/**
 * @description: class for YOLO tensorrt inference
 * @return {*}
 */
class YOLO_TensorRT : virtual public YOLO
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

	/**
	 * @description: release interface
	 * @return {*}
	 */
	void release();

protected:
	/**
	 * @description: nvinfer runtime
	 */
	nvinfer1::IRuntime* m_runtime;

	/**
	 * @description: nvinfer engine
	 */
	nvinfer1::ICudaEngine* m_engine;
	/**
	 * @description: nvinfer execution context
	 */
	nvinfer1::IExecutionContext* m_execution_context;

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: pointer to input
	 */
	float* m_input;
#else
	/**
	 * @description: pointer to uint8_t input 
	 */
	uint8_t* m_input;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: pointer to input device
	 */
	float* m_input_device;

	/**
	 * @description: max input height
	 */
	const int m_max_input_height = 4096;

	/**
	 * @description: max input width
	 */
	const int m_max_input_width = 4096;

	/**
	 * @description: max input size
	 */
	const int m_max_input_size = sizeof(float) * m_max_input_height * m_max_input_width * 3;

	/**
	 * @description: input and output tensor bindings
	 */
	std::vector<void*> m_bindings;

	/**
	 * @description: task type
	 */
	Task_Type m_task_type;

	/**
	 * @description: CUDA stream for async execution
	 */
	cudaStream_t m_stream = nullptr;
};

/**
 * @description: class for the yolo tensorrt classification inference
 */
class YOLO_TensorRT_Classify : public YOLO_TensorRT, public YOLO_Classify
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;

	/**
	 * @description: pointer to output on device
	 */
	float* m_output0_device;

#ifdef _CUDA_PREPROCESS
	uint8_t *m_image_crop, *m_image_centercrop_device, *m_image_resize_device;
#endif // _CUDA_PREPROCESS
};

/**
 * @description: class for the yolo tensorrt detection inference
 */
class YOLO_TensorRT_Detect : public YOLO_TensorRT, virtual public YOLO_Detect
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

	/**
	 * @description: pointer to output on device
	 */
	float* m_output0_device;

#ifdef _CUDA_PREPROCESS
	/**
	 * @description: d2s matrix on host
	 */
	float* m_d2s_host;

	/**
	 * @description: d2s matrix on device
	 */
	float* m_d2s_device;

		/**
	 * @description: s2d matrix on host
	 */
	float* m_s2d_host;

	/**
	 * @description: s2d matrix on device
	 */
	float* m_s2d_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	/**
	 * @description: output bounding box on host
	 */
	float* m_output_box_host;

	/**
	 * @description: output bounding box on device
	 */
	float* m_output_box_device;
#endif // _CUDA_POSTPROCESS

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 7;
	
	/**
	 * @description: max bounding box num
	 */
	const int m_max_box = 1024;
};

/**
 * @description: class for the yolo tensorrt segmentation inference
 */
class YOLO_TensorRT_Segment : public YOLO_TensorRT_Detect, public YOLO_Segment
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

	/**
	 * @description: pointer to output0 on device
	 */
	float* m_output0_device;

	/**
	 * @description: pointer to output1 on device
	 */
	float* m_output1_device;

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 8;

#ifdef _CUDA_POSTPROCESS
	uint8_t* m_mask_device, *m_mask_resized_device, *m_mask_host;
#endif // _CUDA_POSTPROCESS
};

/**
 * @description: class for the yolo tensorrt pose inference
 */
class YOLO_TensorRT_Pose : public YOLO_TensorRT_Detect, public YOLO_Pose
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 58;
};

/**
 * @description: tensorrt inference class for the yolo obb algorithm
 */
class YOLO_TensorRT_OBB : public YOLO_TensorRT_Detect, public YOLO_OBB
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;
#else
	/**
	 * @description: pointer to uint8_t input on host 
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: pointer to output on device
	 */
	float* m_output0_device;

#ifdef _CUDA_PREPROCESS
	/**
	 * @description: d2s matrix on host
	 */
	float* m_d2s_host;

	/**
	 * @description: d2s matrix on device
	 */
	float* m_d2s_device;

		/**
	 * @description: s2d matrix on host
	 */
	float* m_s2d_host;

	/**
	 * @description: s2d matrix on device
	 */
	float* m_s2d_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	/**
	 * @description: output bounding box on host
	 */
	float* m_output_box_host;

	/**
	 * @description: output bounding box on device
	 */
	float* m_output_box_device;

	/**
	 * @description: pre-allocated temp buffer: scores
	 */
	float* m_scores_device;

	/**
	 * @description: pre-allocated temp buffer: sorted indices
	 */
	int* m_sorted_idx_device;

	/**
	 * @description: pre-allocated temp buffer: sorted boxes
	 */
	float* m_output_box_sorted_device;

	/**
	 * @description: pre-allocated temp buffer: covariance a
	 */
	float* m_a_device;

	/**
	 * @description: pre-allocated temp buffer: covariance b
	 */
	float* m_b_device;

	/**
	 * @description: pre-allocated temp buffer: covariance c
	 */
	float* m_c_device;

	/**
	 * @description: pre-allocated temp buffer: hd matrix
	 */
	float* m_hd_device;

	/**
	 * @description: pre-allocated temp buffer: picked indices
	 */
	int* m_picked_device;

	/**
	 * @description: pre-allocated temp buffer: picked boxes
	 */
	float* m_picked_boxes_device;

	/**
	 * @description: pre-allocated host buffer for picked boxes
	 */
	float* m_picked_boxes_host;

	/**
	 * @description: pre-allocated temp buffer for cuda_pick_bbox: col_max
	 */
	float* m_col_max_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_pick_bbox: mask
	 */
	int* m_mask_pick_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_pick_bbox: prefix_sum
	 */
	int* m_prefix_sum_device;
#endif // _CUDA_POSTPROCESS

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 8;

	/**
	 * @description: max bounding box num
	 */
	const int m_max_box = 4096;
};

/**
 * @description: tensorrt inference class for the yolo depth estimation algorithm
 */
class YOLO_TensorRT_Depth : public YOLO_TensorRT_Detect, public YOLO_Depth
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;
#else
	/**
	 * @description: pointer to uint8_t input on host 
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: pointer to output on device
	 */
	float* m_output0_device;

#ifdef _CUDA_POSTPROCESS
	float* m_depth_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_scale_mask: crop
	 */
	float* m_scale_crop_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_scale_mask: tmp_out
	 */
	float* m_scale_tmp_out_device;

	/**
	 * @description: pitch of crop buffer
	 */
	size_t m_scale_crop_pitch;

	/**
	 * @description: pitch of tmp_out buffer
	 */
	size_t m_scale_tmp_out_pitch;
#endif // _CUDA_POSTPROCESS
};

/**
 * @description: tensorrt inference class for the yolo semantic segmentation algorithm
 */
class YOLO_TensorRT_Semantic : public YOLO_TensorRT_Detect, public YOLO_Semantic
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: resource release
	 * @return {*}
	 */
	void release();

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;
#else
	/**
	 * @description: pointer to uint8_t input on host 
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: pointer to output on device
	 */
	uint8_t* m_output0_device;

#ifdef _CUDA_POSTPROCESS
	/**
	 * @description: pre-allocated temp buffer for mask
	 */
	uint8_t* m_mask_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_scale_mask: crop
	 */
	uint8_t* m_scale_crop_device;

	/**
	 * @description: pre-allocated temp buffer for cuda_scale_mask: tmp_out
	 */
	uint8_t* m_scale_tmp_out_device;

	/**
	 * @description: pitch of crop buffer
	 */
	size_t m_scale_crop_pitch;

	/**
	 * @description: pitch of tmp_out buffer
	 */
	size_t m_scale_tmp_out_pitch;
#endif // _CUDA_POSTPROCESS
};