#pragma once


#include "yolov5.h"
#include <openvino/openvino.hpp> 


class YOLOv5_OpenVINO : public YOLOv5
{
public:
	YOLOv5_OpenVINO(std::string model_path, Device_Type device_type);

private:
	void pre_process();

	void process();

	void post_process();

	void release();

	ov::InferRequest m_infer_request;

	ov::Output<const ov::Node> m_input_port;

	cv::Mat m_inputs;

	ov::Tensor m_outputs;
};