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
	void scale_mask(cv::Mat& mask, const cv::Size input_shape, const cv::Size output_shape)
	{
		double gain = std::min(static_cast<double>(input_shape.height) / output_shape.height,
							static_cast<double>(input_shape.width) / output_shape.width);

		int pad_w = static_cast<int>((input_shape.width - output_shape.width * gain) / 2);
		int pad_h = static_cast<int>((input_shape.height - output_shape.height * gain) / 2);

		cv::Rect roi(pad_w, pad_h, mask.cols - 2 * pad_w, mask.rows - 2 * pad_h);
		roi &= cv::Rect(0, 0, mask.cols, mask.rows);
		mask = mask(roi).clone();
		cv::resize(mask, mask, cv::Size(output_shape.width, output_shape.height), 0, 0, cv::INTER_LINEAR);
	}

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