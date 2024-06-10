#pragma once

#include <cuda_runtime.h>
#include "yolo_classification.h"
#include "yolo_detection.h"
#include "yolo_segmentation.h"
#include "utils.h"

#ifdef _YOLO_TENSORRT

#include <NvInfer.h>
#include <NvInferRuntime.h>
	
//#define _CUDA_PREPROCESS	
//#define _CUDA_POSTPROCESS
//
//#ifdef _CUDA_POSTPROCESS
//	#define _CUDA_PREPROCESS
//#endif // CUDA_POSTPROCESS

class YOLO_TensorRT : virtual public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	Algo_Type m_algo;

	Model_Type m_model;

	nvinfer1::IExecutionContext* m_execution_context;

	cudaStream_t m_stream;

	float* m_input_device;

	const int m_max_input_size = sizeof(float) * m_input_numel;

	const int m_max_image_bbox = 1024;
};

class YOLO_TensorRT_Classification : public YOLO_TensorRT, public YOLO_Classification
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	void release();

	float* m_bindings[2];

	float* m_input_host;

	float* m_output_host;

	float* m_output_device;
};

class YOLO_TensorRT_Detection : public YOLO_TensorRT, public YOLO_Detection
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	void release();

	float* m_bindings[2];

#ifndef _CUDA_PREPROCESS
	float* m_input_host;
#else
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	float* m_output_host;

	float* m_output_device;

#ifdef _CUDA_PREPROCESS
	float* m_affine_matrix_host;

	float* m_affine_matrix_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	float* m_output_box_host;

	float* m_output_box_device;
#endif // _CUDA_POSTPROCESS
};

class YOLO_TensorRT_Segmentation : public YOLO_TensorRT, public YOLO_Segmentation
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	void release();

	float* m_bindings[3];

#ifndef _CUDA_PREPROCESS
	float* m_input_host;
#else
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	float* m_output0_host;

	float* m_output1_host;

	float* m_output0_device;

	float* m_output1_device;

#ifdef _CUDA_PREPROCESS
	float* m_affine_matrix_host;

	float* m_affine_matrix_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	float* m_output_box_host;

	float* m_output_box_device;
#endif // _CUDA_POSTPROCESS
};

#endif // _YOLO_TensorRT