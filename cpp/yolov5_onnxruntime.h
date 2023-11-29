#pragma once


#include "yolov5.h"
#include <onnxruntime_c_api.h>


class YOLOv5_ONNXRuntime : public YOLOv5
{
public:
	YOLOv5_ONNXRuntime(std::string model_path, Device_Type device_type);

	~YOLOv5_ONNXRuntime();

private:
	void pre_process();

	void process();

	void post_process();

	const OrtApi* m_ort;

	OrtSession* m_session;

	OrtMemoryInfo* m_memory_info;

	float* m_inputs;

	float* m_outputs;

	std::vector<const char*> m_input_names;

	std::vector<const char*> m_output_names;
};