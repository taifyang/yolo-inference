#pragma once

#include "yolo_classification.h"
#include "yolo_detection.h"
#include "yolo_segmentation.h"
#include "utils.h"

#ifdef _YOLO_LIBTORCH

#include <torch/script.h>
#include <torch/torch.h>

class YOLO_Libtorch : virtual public YOLO
{	
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	Algo_Type m_algo;

	torch::DeviceType m_device;

	Model_Type m_model;

	torch::jit::script::Module module;

	std::vector<torch::jit::IValue> m_input;

	torch::jit::IValue m_output;
};

class YOLO_Libtorch_Classification : public YOLO_Libtorch, public YOLO_Classification
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_Libtorch_Detection : public YOLO_Libtorch, public YOLO_Detection
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_Libtorch_Segmentation : public YOLO_Libtorch, public YOLO_Segmentation
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output0_host;

	float* m_output1_host;
};

#endif // _YOLO_Libtorch