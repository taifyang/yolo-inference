/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 21:34:52
 * @FilePath: \cpp\yolo_classify.h
 * @Description: 分类算法类
 */

#pragma once

#include "yolo.h"
#include "utils.h"

/**
 * @description: 分类算法抽象类
 */
class YOLO_Classify : virtual public YOLO
{
protected:
	/**
	 * @description: 			画出结果
	 * @param {string} label	类别标签
	 */
	void draw_result(std::string label)
	{
		int baseLine;
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
		cv::putText(m_result, label, cv::Point(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
	}

	/**
	 * @description: 分类类别数
	 */	
	int class_num = 1000;

	/**
	 * @description: 模型输出
	 */
	float* m_output_host;
};
