#pragma once

#include "yolo_classification.h"
#include "yolo_detection.h"
#include "yolo_segmentation.h"
#include "utils.h"

#ifdef _YOLO_OPENCV

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

class YOLO_OpenCV_Classification : public YOLO_OpenCV, public YOLO_Classification
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};

class YOLO_OpenCV_Detection : public YOLO_OpenCV, public YOLO_Detection
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};

class YOLO_OpenCV_Segmentation : public YOLO_OpenCV, public YOLO_Segmentation
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void post_process();
};

#endif // _YOLO_OpenCV