#pragma once


#include "yolov5.h"
#include <opencv2/opencv.hpp>


class YOLOv5_OpenCV : public YOLOv5
{
public:
	YOLOv5_OpenCV(std::string model_path, Device_Type device_type);

	~YOLOv5_OpenCV();

private:
	void pre_process();

	void process();

	void post_process();

	cv::dnn::Net m_net;

	cv::Mat m_inputs;

	std::vector<cv::Mat> m_outputs;
};