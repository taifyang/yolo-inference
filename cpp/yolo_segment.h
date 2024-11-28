/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-11-22 23:20:41
 * @FilePath: \cpp\yolo_segment.h
 * @Description: segmentation algorithm class
 */

#pragma once

#include "yolo_detect.h"

/**
 * @description: segmentation network output related parameters
 */
struct OutputSeg
{
	int id;             //class id
	float score;   		//score
	cv::Rect box;       //bounding box
	cv::Mat mask;    	//mask
};

/**
 * @description: mask parameters
 */
struct MaskParams
{
	int seg_channels = 32;
	int seg_width = 160;
	int seg_height = 160;
	int net_width = 640;
	int net_height = 640;
	float mask_threshold = 0.5;
	cv::Size input_shape;
	cv::Vec4d params;
	Algo_Type algo_type;
};

/**
 * @description: segmentation class for YOLO algorithm
 */
class YOLO_Segment : public YOLO_Detect
{
protected:
	/**
	 * @description: 					get mask
	 * @param {Mat&} mask_proposals		mask proposals
	 * @param {Mat&} mask_protos		mask protos
	 * @param {OutputSeg&} output		mask output
	 * @param {MaskParams&} mask_params	mask parameters
	 * @return {*}
	 */
	void GetMask(const cv::Mat& mask_proposals, const cv::Mat& mask_protos, OutputSeg& output, const MaskParams& mask_params)
	{
		int seg_channels = mask_params.seg_channels;
		int net_width = mask_params.net_width;
		int seg_width = mask_params.seg_width;
		int net_height = mask_params.net_height;
		int seg_height = mask_params.seg_height;
		float mask_threshold = mask_params.mask_threshold;
		cv::Vec4f params = mask_params.params;
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
		cv::Mat matmul_res = (mask_proposals * protos).t();
		cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
		cv::Mat dest, mask;

		//sigmoid
		if(mask_params.algo_type == YOLOv5)
		{
			cv::exp(-masks_feature, dest);
			dest = 1.0 / (1.0 + dest);
		}
		else
		{
			dest = masks_feature;
		}

		int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
		int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
		int width = ceil(net_width / seg_width * rang_w / params[0]);
		int height = ceil(net_height / seg_height * rang_h / params[1]);

		cv::resize(dest, mask, cv::Size(width, height), cv::INTER_LINEAR);
		mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
		output.mask = mask;
	}

	/**
	 * @description: 						draw result
	 * @param {vector<OutputSeg>} result	segmentation model output
	 */	
	void draw_result(std::vector<OutputSeg> output_seg)
	{
		cv::Mat mask = m_image.clone();
    	m_result = m_image.clone();
		srand(time(0));

		for (int i = 0; i < output_seg.size(); i++)
		{
			cv::rectangle(m_result, output_seg[i].box, cv::Scalar(255, 0, 0), 1);
			mask(output_seg[i].box).setTo(cv::Scalar(rand() % 256, rand() % 256, rand() % 256), output_seg[i].mask);
			std::string label = "class" + std::to_string(output_seg[i].id) + ":" + cv::format("%.2f", output_seg[i].score);
			cv::putText(m_result, label, cv::Point(output_seg[i].box.x, output_seg[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}
		
		addWeighted(m_result, 0.5, mask, 0.5, 0, m_result);
	}

	/**
	 * @description: mask parameters
	 */
	MaskParams m_mask_params;

	/**
	 * @description: output segmentation size
	 */	
	int m_output_numseg;

	 /**
	 * @description: model output0 on host
	 * @return {*}
	 */
	float* m_output0_host;

	/**
	 * @description: model output1 on host
	 * @return {*}
	 */
	float* m_output1_host;

	/**
	 * @description: segmentation model output
	 */
	std::vector<OutputSeg> m_output_seg;
};

