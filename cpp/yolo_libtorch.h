#pragma once


#include "yolo.h"
#include <torch/script.h>
#include <torch/torch.h>


class YOLO_Libtorch : public YOLO
{	
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();
		
	void process();

	void post_process();

	Algo_Type m_algo;

	torch::DeviceType m_device;

	Model_Type m_model;

	torch::jit::script::Module module;

	std::vector<torch::jit::IValue> m_inputs;

	torch::jit::IValue m_outputs;
};