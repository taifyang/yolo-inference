/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2024-10-30 22:30:23
 * @Description: classification algorithm class
 */

#pragma once

#include "yolo.h"
#include "utils.h"

/**
 * @description: classification network output related parameters
 */
struct OutputCls
{
	int id;             //class id
	float score;   		//score
};

/**
 * @description: classification class for YOLO algorithm
 */
class YOLO_Classify : virtual public YOLO
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
	{
		if(m_algo_type == YOLOv5)
		{
			m_input_size.width = 640;
			m_input_size.height = 640;
		}
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
		{
			m_input_size.width = 224;
			m_input_size.height = 224;
		}
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}

protected:
	void CenterCrop(cv::Mat& input_image, cv::Mat& output_image)
	{
		int crop_size = std::min(input_image.cols, input_image.rows);
		int left = (input_image.cols - crop_size) / 2, top = (input_image.rows - crop_size) / 2;
		output_image = input_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(output_image, output_image, cv::Size(m_input_size.width, m_input_size.height));
	}

	void Normalize(cv::Mat& input_image, cv::Mat& output_image, Algo_Type algo_type)
	{
		output_image.convertTo(input_image, CV_32FC3, 1. / 255.);
		if (m_algo_type == YOLOv5)
		{
			cv::subtract(output_image, cv::Scalar(0.406, 0.456, 0.485), output_image);
			cv::divide(output_image, cv::Scalar(0.225, 0.224, 0.229), output_image);
		}
	}

	/**	
	 * @description: 								draw result
	 * @param {OutputCls} output_cls				classification model output
	 * @return {*}
	 */
	void draw_result(OutputCls output_cls)
	{
		int baseLine;
    	m_result = m_image.clone();
		std::string label = "class" + std::to_string(output_cls.id) + ":" + cv::format("%.2f", output_cls.score);
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
		cv::putText(m_result, label, cv::Point(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
	}

	/**
	 * @description: class num 
	 */	
	int m_class_num = 1000;

	/**
	 * @description: model output on host
	 */
	float* m_output0_host;

	/**
	 * @description: classification network output
	 */
	OutputCls m_output_cls;
};
