/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-06-29 22:00:02
 * @FilePath: \cpp\tensorrt\yolo_tensorrt.h
 * @Description: tensorrt inference header file for YOLO algorithm
 */

#pragma once

#include <cuda_runtime.h>
#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

/**
 * @description: tensorrt inference class for YOLO algorithm
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
	 * @description: cuda stream
	 */
	cudaStream_t m_stream;

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
 * @description: tensorrt inference class for the yolo classification algorithm
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
	float* m_output_device;
};

/**
 * @description: tensorrt inference class for the yolo detection algorithm
 */
class YOLO_TensorRT_Detect : public YOLO_TensorRT, public YOLO_Detect
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
	float* m_output_device;

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
 * @description: tensorrt inference class for the yolo segmentation algorithm
 */
class YOLO_TensorRT_Segment : public YOLO_TensorRT, public YOLO_Segment
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

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: pointer to input on host
	 */
	float* m_input_host;
#else
	/**
	 * @description:  pointer to uint8_t input on host
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: pointer to output0 on device
	 */
	float* m_output0_device;

	/**
	 * @description: pointer to output1 on device
	 */
	float* m_output1_device;

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
};