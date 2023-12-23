#pragma once


#include "yolov5.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>


class YOLOv5_TensorRT : public YOLOv5
{
public:
	YOLOv5_TensorRT(std::string model_path, Device_Type device_type, Model_Type model_type);

	~YOLOv5_TensorRT();

private:
	void pre_process();

	void process();

	void post_process();

	nvinfer1::IExecutionContext* m_execution_context;

	cudaStream_t m_stream;

	float* m_bindings[2];

	float* m_inputs_device;

	float* m_outputs_device;

	float* m_inputs_host;

	uint16_t* m_inputs_host_fp16;

	uint16_t* m_outputs_host_fp16;

	Model_Type m_model;
};