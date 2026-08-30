/* 
 * @Author: taifyang
 * @Date: 2026-08-05 20:53:59
 * @LastEditTime: 2026-08-14 23:29:31
 * @Description: depth estimation algorithm class
 */

#pragma once

#include "yolo_detect.h"

/**
 * @description: segmentation class for YOLO algorithm
 */
class YOLO_Depth : virtual public YOLO_Detect
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
	{
		m_input_size = cv::Size(768, 768);
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
		m_output_numdet = 1 * 1 * m_input_size.width * m_input_size.height;
	}

protected:
	/**
	 * @description: draw result
	 * @return {*}
	 */	
	void draw_result()
	{
		m_depth.convertTo(m_result, CV_16UC1, 1000);
		cv::normalize(m_result, m_result, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}

	cv::Mat m_depth;
};