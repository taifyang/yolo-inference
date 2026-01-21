/* 
 * @Author: taifyang
 * @Date: 2026-01-03 19:31:25
 * @LastEditTime: 2026-01-17 20:19:33
 * @Description: pose algorithm class
 */

#pragma once

#include "yolo_detect.h"

/**
 * @description: pose network output related parameters
 */
struct OutputPose
{
	int id;             			//class id
	float score;   					//score
	cv::Rect box;       			//bounding box
	std::vector<float> keypoint;	//keypoint
};

/**
 * @description: pose class for YOLO algorithm
 */
class YOLO_Pose : virtual public YOLO_Detect
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
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			m_output_numprob = 56;
			m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
		}
		else if(m_algo_type == YOLO26)
		{
			m_output_numprob = 57;
			m_output_numbox = 300;
		}
	
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	}

protected:
	/**
	 * @description: 										scale boxes
	 * @param {std::vector<cv::Rect>&} boxes				detect boxes
	 * @param {std::vector<std::vector<float>>&} keypoints	pose keypoints
	 * @param {Size} size									output image shape
	 * @return {*}
	 */
	void scale_boxes(std::vector<cv::Rect>& boxes, std::vector<std::vector<float>>& keypoints, cv::Size size)
	{
		float gain = std::min(m_input_size.width * 1.0 / size.width, m_input_size.height * 1.0 / size.height);
		int pad_w = (m_input_size.width - size.width * gain) / 2;
		int pad_h = (m_input_size.height - size.height * gain) / 2;
		
		for (auto& box : boxes)
		{
			box.x -= pad_w;
			box.y -= pad_h;
			box.x /= gain;
			box.y /= gain;
			box.width /= gain;
			box.height /= gain;
		}

		for(auto& keypoint : keypoints)
		{
			for (size_t i = 0; i < keypoint.size() / 3; i++)
			{
				keypoint[3 * i] = (keypoint[3 * i] - pad_w) / gain;
				keypoint[3 * i + 1] = (keypoint[3 * i + 1] - pad_h) / gain;
			}
		}
	}

	/**
	 * @description: 								draw result
	 * @param {std::vector<OutputPose>} output_det	pose model output
	 * @return {*}
	 */
	void draw_result(std::vector<OutputPose> output_pose)
	{
    	m_result = m_image.clone();
		for (int i = 0; i < output_pose.size(); i++)
		{
			OutputPose output = output_pose[i];
			int idx = output.id;
			float score = output.score;
			cv::Rect box = output.box;
			std::string label = "class" + std::to_string(idx) + ":" + cv::format("%.2f", score);
			cv::rectangle(m_result, box, cv::Scalar(0, 255, 0), 2);
			cv::putText(m_result, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			std::vector<float> keypoint = output.keypoint;
			for (size_t i = 0; i < keypoint.size() / 3; i++)
			{
				if (keypoint[3 * i + 2] < 0.5)
					continue;
				cv::circle(m_result, cv::Point(keypoint[3 * i], keypoint[3 * i + 1]), 4, cv::Scalar(255, 0, 0), -1);
			}

			for (auto skeleton : skeletons)
			{
				cv::Point pos1(keypoint[3 * (skeleton.first - 1)], keypoint[3 * (skeleton.first - 1) + 1]);
				cv::Point pos2(keypoint[3 * (skeleton.second - 1)], keypoint[3 * (skeleton.second - 1) + 1]);
				float conf1 = keypoint[3 * (skeleton.first - 1) + 2];
				float conf2 = keypoint[3 * (skeleton.second - 1) + 2];
				if (conf1 > 0.5 && conf2 > 0.5)
				{
					cv::line(m_result, pos1, pos2, cv::Scalar(255, 0, 0), 2);
				}
			}
		}
	}

	/**
	 * @description: class num
	 */	
	int m_class_num = 1;

	/**
	 * @description: detection model output
	 */
	std::vector<OutputPose> m_output_pose;

	const std::unordered_multimap<int, int> skeletons = { {16, 14}, { 14, 12 }, {17, 15}, {15, 13}, {12, 13}, {6, 12},
	{7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7} };
};
