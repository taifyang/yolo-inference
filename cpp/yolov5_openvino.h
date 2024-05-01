#pragma once


#include "yolov5.h"
#include <openvino/openvino.hpp> 


class YOLOv5_OpenVINO : public YOLOv5
{
public:
	void init(const std::string model_path, const Device_Type device_type, const Model_Type model_type);

private:
	void pre_process();

	void process();

	ov::InferRequest m_infer_request;

	ov::Output<const ov::Node> m_input_port;

	cv::Mat m_inputs;

	ov::Tensor m_outputs;
};