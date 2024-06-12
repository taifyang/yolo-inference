#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
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

class YOLO_ONNXRuntime_Classify : public YOLO_ONNXRuntime, public YOLO_Classify
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_ONNXRuntime_Detect : public YOLO_ONNXRuntime, public YOLO_Detect
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_ONNXRuntime_Segment : public YOLO_ONNXRuntime, public YOLO_Segment
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