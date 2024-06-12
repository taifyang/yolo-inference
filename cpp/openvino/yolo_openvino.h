#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <openvino/openvino.hpp> 

class YOLO_OpenVINO : virtual public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	Algo_Type m_algo;

	ov::InferRequest m_infer_request;

	ov::Output<const ov::Node> m_input_port;

	cv::Mat m_input;
};

class YOLO_OpenVINO_Classify : public YOLO_OpenVINO, public YOLO_Classify
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_OpenVINO_Detect : public YOLO_OpenVINO, public YOLO_Detect
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_OpenVINO_Segment : public YOLO_OpenVINO, public YOLO_Segment
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