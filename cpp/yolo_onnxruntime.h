#pragma once

#include "yolo_classification.h"
#include "yolo_detection.h"
#include "yolo_segmentation.h"
#include "utils.h"

#ifdef _YOLO_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

class YOLO_ONNXRuntime : virtual public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

	void release();

protected:
	Algo_Type m_algo;

	Model_Type m_model;

	Ort::Env m_env;

	Ort::Session* m_session;

	std::vector<float> m_input;

	std::vector<uint16_t> m_input_fp16;

	std::vector<uint16_t> m_output_fp16;

	std::vector<const char*> m_input_names;

	std::vector<const char*> m_output_names;
};

class YOLO_ONNXRuntime_Classification : public YOLO_ONNXRuntime, public YOLO_Classification
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_ONNXRuntime_Detection : public YOLO_ONNXRuntime, public YOLO_Detection
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_ONNXRuntime_Segmentation : public YOLO_ONNXRuntime, public YOLO_Segmentation
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output0_host;

	float* m_output1_host;

	std::vector<uint16_t> m_output0_fp16;

	std::vector<uint16_t> m_output1_fp16;
};

#endif // _YOLO_ONNXRuntime