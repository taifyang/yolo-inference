#pragma once


#include "yolov5.h"
#include <opencv2/opencv.hpp>


class YOLOv5_OpenCV : public YOLOv5
{
public:
	void init(const std::string model_path, const Device_Type device_type, const Model_Type model_type);

private:
	void pre_process();

	void process();

	cv::dnn::Net m_net;

	cv::Mat m_inputs;

	std::vector<cv::Mat> m_outputs;
};