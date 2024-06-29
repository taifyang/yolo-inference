/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-29 16:33:27
 * @FilePath: \cpp\yolo_classify.h
 * @Description: 分类算法类
 */

#pragma once

#include "yolo.h"
#include "utils.h"

/**
 * @description: 分类网络输出相关参数
 */
struct OutputCls
{
	int id;             //结果类别id
	float score;   		//结果得分
};

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
	void draw_result(OutputCls output_cls)
	{
		int baseLine;
		std::string label = "class" + std::to_string(output_cls.id) + ":" + cv::format("%.2f", output_cls.score);
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

	/**
	 * @description: 分类模型输出
	 */
	OutputCls m_output_cls;
};
