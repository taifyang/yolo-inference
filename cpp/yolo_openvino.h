#pragma once


#include "yolo.h"
#include <openvino/openvino.hpp> 


class YOLO_OpenVINO : public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	Algo_Type m_algo;

	ov::InferRequest m_infer_request;

	ov::Output<const ov::Node> m_input_port;

	cv::Mat m_inputs;

	ov::Tensor m_outputs;
};