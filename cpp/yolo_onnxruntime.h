#pragma once


#include "yolo.h"
#include <onnxruntime_cxx_api.h>


class YOLO_ONNXRuntime : public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

	void release();

private:
	void pre_process();

	void process();

	void post_process();

	Algo_Type m_algo;

	Model_Type m_model;

	Ort::Env m_env;

	Ort::Session* m_session;

	std::vector<float> m_inputs;

	std::vector<uint16_t> m_inputs_fp16;

	std::vector<uint16_t> m_outputs_fp16;

	std::vector<const char*> m_input_names;

	std::vector<const char*> m_output_names;
};