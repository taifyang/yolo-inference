#pragma once


#include "yolov5.h"
#include <torch/script.h>
#include <torch/torch.h>


class YOLOv5_Libtorch : public YOLOv5
{	
public:
	YOLOv5_Libtorch(std::string model_path, Device_Type device_type, Model_Type model_type);

	~YOLOv5_Libtorch();

private:
	void pre_process();
		
	void process();

	void post_process();

	torch::DeviceType m_device;

	Model_Type m_model;

	torch::jit::script::Module module;

	std::vector<torch::jit::IValue> m_inputs;

	torch::jit::IValue m_outputs;
};