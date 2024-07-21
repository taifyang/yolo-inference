/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 
 * @LastEditTime: 2024-06-29 16:28:24
 * @FilePath: \cpp\yolo_segment.h
 * @Description: 分割算法类
 */

#pragma once

#include "yolo_detect.h"

/**
 * @description: 分割网络输出相关参数
 */
struct OutputSeg
{
	int id;             //结果类别id
	float score;   		//结果得分
	cv::Rect box;       //矩形框
	cv::Mat boxMask;    //矩形框内mask，节省内存空间和加快速度
};

/**
 * @description: 掩膜相关参数
 */
struct MaskParams
{
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

/**
 * @description: 分割算法抽象类
 */
class YOLO_Segment : public YOLO_Detect
{
protected:
	/**
	 * @description: 					获取掩码
	 * @param {Mat&} maskProposals		掩码候选
	 * @param {Mat&} mask_protos		输入掩码
	 * @param {OutputSeg&} output		输出掩码
	 * @param {MaskParams&} maskParams	掩码参数
	 * @return {*}
	 */
	void GetMask(const cv::Mat& maskProposals, const cv::Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams)
	{
		int seg_channels = maskParams.segChannels;
		int net_width = maskParams.netWidth;
		int seg_width = maskParams.segWidth;
		int net_height = maskParams.netHeight;
		int seg_height = maskParams.segHeight;
		float mask_threshold = maskParams.maskThreshold;
		cv::Vec4f params = maskParams.params;
		cv::Size src_img_shape = maskParams.srcImgShape;
		cv::Rect temp_rect = output.box;

		//crop from mask_protos
		int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
		int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
		int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
		int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

		rang_w = MAX(rang_w, 1);
		rang_h = MAX(rang_h, 1);
		if (rang_x + rang_w > seg_width)
		{
			if (seg_width - rang_x > 0)
				rang_w = seg_width - rang_x;
			else
				rang_x -= 1;
		}
		if (rang_y + rang_h > seg_height)
		{
			if (seg_height - rang_y > 0)
				rang_h = seg_height - rang_y;
			else
				rang_y -= 1;
		}

		std::vector<cv::Range> roi_rangs;
		roi_rangs.push_back(cv::Range(0, 1));
		roi_rangs.push_back(cv::Range::all());
		roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
		roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

		//crop
		cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
		cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
		cv::Mat matmul_res = (maskProposals * protos).t();
		cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
		cv::Mat dest, mask;

		//sigmoid
		cv::exp(-masks_feature, dest);
		dest = 1.0 / (1.0 + dest);

		int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
		int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
		int width = ceil(net_width / seg_width * rang_w / params[0]);
		int height = ceil(net_height / seg_height * rang_h / params[1]);

		cv::resize(dest, mask, cv::Size(width, height), cv::INTER_LINEAR);
		mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
		output.boxMask = mask;
	}

	/**
	 * @description: 						画出结果
	 * @param {vector<OutputSeg>} result	分割结果
	 */	
	void draw_result(std::vector<OutputSeg> output_seg)
	{
		srand(time(0));
		std::vector<cv::Scalar> color;
		for (int i = 0; i < class_num; i++)
		{
			color.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
		}

		cv::Mat mask = m_image.clone();
    	m_result = m_image.clone();
		for (int i = 0; i < output_seg.size(); i++)
		{
			cv::rectangle(m_result, output_seg[i].box, cv::Scalar(255, 0, 0), 1);
			mask(output_seg[i].box).setTo(color[output_seg[i].id], output_seg[i].boxMask);
			std::string label = "class" + std::to_string(output_seg[i].id) + ":" + cv::format("%.2f", output_seg[i].score);
			cv::putText(m_result, label, cv::Point(output_seg[i].box.x, output_seg[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}
		
		addWeighted(m_result, 0.5, mask, 0.5, 0, m_result);
	}

	/**
	 * @description: 掩码参数
	 */
	MaskParams m_mask_params;

	/**
	 * @description: 输出分割尺寸
	 */	
	int m_output_numseg;

	 /**
	 * @description: 模型输出
	 * @return {*}
	 */
	float* m_output0_host;

	/**
	 * @description: 模型输出
	 * @return {*}
	 */
	float* m_output1_host;

	/**
	 * @description: 分割模型输出
	 */
	std::vector<OutputSeg> m_output_seg;
};
