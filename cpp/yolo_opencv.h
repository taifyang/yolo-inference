#pragma once


#include "yolo.h"
#include <opencv2/opencv.hpp>


class YOLO_OpenCV : public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	Algo_Type m_algo;

	cv::dnn::Net m_net;

	cv::Mat m_inputs;

	std::vector<cv::Mat> m_outputs;
};