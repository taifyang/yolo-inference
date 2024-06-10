#pragma once

#include "yolo_classification.h"
#include "yolo_detection.h"
#include "yolo_segmentation.h"
#include "utils.h"

#ifdef _YOLO_OPENVINO

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

class YOLO_OpenVINO_Classification : public YOLO_OpenVINO, public YOLO_Classification
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_OpenVINO_Detection : public YOLO_OpenVINO, public YOLO_Detection
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	void pre_process();

	void process();

	void post_process();

	float* m_output_host;
};

class YOLO_OpenVINO_Segmentation : public YOLO_OpenVINO, public YOLO_Segmentation
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

#endif // _YOLO_OpenVINO