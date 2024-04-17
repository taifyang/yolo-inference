#pragma once


#include "yolov5.h"
#include <onnxruntime_cxx_api.h>


class YOLOv5_ONNXRuntime : public YOLOv5
{
public:
	YOLOv5_ONNXRuntime(std::string model_path, Device_Type device_type, Model_Type model_type);

	~YOLOv5_ONNXRuntime();

private:
	void pre_process();

	void process();

	Model_Type m_model;

	Ort::Env m_env;

	Ort::Session* m_session;

	std::vector<float> m_inputs;

	std::vector<uint16_t> m_inputs_fp16;

	std::vector<uint16_t> m_outputs_fp16;

	std::vector<const char*> m_input_names;

	std::vector<const char*> m_output_names;
};