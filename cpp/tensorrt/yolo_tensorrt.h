/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-19 18:50:43
 * @Description: header file for YOLO tensorrt inference
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "yolo_pose.h"
#include "yolo_obb.h"
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

	/**
	 * @description: pointer to input device
	 */
	float* m_input_device;

	/**
	 * @description: max input size
	 */
	const int m_max_input_size = sizeof(float) * m_input_numel;

	/**
	 * @description: max bounding box num
	 */
	const int m_max_box = 1024;

	/**
	 * @description: task type
	 */
	Task_Type m_task_type;
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
	 * @description: input and output tensor bindings
	 */
	float* m_bindings[2];

	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;

	/**
	 * @description: pointer to output on device
	 */
	float* m_output0_device;

#ifdef _CUDA_PREPROCESS
	uint8_t *m_image_device, *m_image_crop, *m_image_centercrop_device, *m_image_resize_device;
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
	 * @description: input and output tensor bindings
	 */
	float* m_bindings[2];

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
#endif // _CUDA_POSTPROCESS

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 7;
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
	 * @description: input and output tensor bindings
	 */
	float* m_bindings[3];

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

	/**
	 * @description: input and output tensor bindings
	 */
	float* m_bindings[2];

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
#endif // _CUDA_POSTPROCESS

	/**
	 * @description: box element num
	 */
	const int m_num_box_element = 8;

		/**
	 * @description: max input size
	 */
	const int m_max_input_size = sizeof(float) * 3 * 4096 * 4096;

	/**
	 * @description: max bounding box num
	 */
	const int m_max_box = 4096;
};

/**
 * @description: 			compute affine transformation
 * @param {float*} matrix	input matrix
 * @param {float} x			input x
 * @param {float} y			input y
 * @param {float*} ox		output x
 * @param {float*} oy		output y
 * @return {*}
 */	
static void affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}
