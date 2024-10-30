/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 22:30:23
 * @FilePath: \cpp\yolo_classify.h
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
protected:
	/**
	 * @description: 								draw result
	 * @param {OutputCls} output_cls				classification model output
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
	float* m_output_host;

	/**
	 * @description: classification network output
	 */
	OutputCls m_output_cls;
};
