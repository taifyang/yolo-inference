#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

class YOLO_OpenCV : virtual public YOLO
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	void process();

	Algo_Type m_algo;

	cv::dnn::Net m_net;

	cv::Mat m_input;

	std::vector<cv::Mat> m_output;

	float* m_output_host;
};

class YOLO_OpenCV_Classify : public YOLO_OpenCV, public YOLO_Classify
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};

class YOLO_OpenCV_Detect : public YOLO_OpenCV, public YOLO_Detect
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};

class YOLO_OpenCV_Segment : public YOLO_OpenCV, public YOLO_Segment
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};