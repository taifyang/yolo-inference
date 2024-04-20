#pragma once


#include "yolov5.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
	

#define MAX_IMAGE_INPUT_SIZE_THRESH sizeof(float) * input_numel
#define MAX_IMAGE_BBOX 1024
#define CUDA_PREPROCESS
#define CUDA_POSTPROCESS

#ifdef CUDA_POSTPROCESS
	#define CUDA_PREPROCESS
#endif // CUDA_POSTPROCESS


class YOLOv5_TensorRT : public YOLOv5
{
public:
	void init(const std::string model_path, const Device_Type device_type, Model_Type model_type);

	void release();

private:
	void pre_process();

	void process();

#ifdef CUDA_POSTPROCESS
	void post_process();
#endif // CUDA_POSTPROCESS

	Model_Type m_model;

	nvinfer1::IExecutionContext* m_execution_context;

	cudaStream_t m_stream;

	float* m_bindings[2];

	float* m_inputs_device;

	float* m_outputs_device;

#ifndef CUDA_PREPROCESS
	float* m_inputs_host;
#else
	uint8_t* m_inputs_host;
#endif // !CUDA_PREPROCESS

#ifdef CUDA_PREPROCESS
	float* m_affine_matrix_host;

	float* m_affine_matrix_device;
#endif // CUDA_PREPROCESS

#ifdef CUDA_POSTPROCESS
	float* m_outputs_box_host;

	float* m_outputs_box_device;
#endif // CUDA_POSTPROCESS
};
