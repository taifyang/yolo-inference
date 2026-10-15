/* 
 * @Author: taifyang
 * @Date: 2026-01-08 22:18:50
 * @LastEditTime: 2026-10-15 22:58:00
 * @Description: obb algorithm class
 */

#pragma once

#include "yolo_detect.h"
#include <Eigen/Dense>

/**
 * @description: detection network output related parameters
 */
struct OutputOBB
{
	int id;             				//class id
	float score;   						//score
	cv::RotatedRect  box_rotate;       	//rotate box
};

/**
 * @description: detection class for YOLO algorithm
 */
class YOLO_OBB : virtual public YOLO_Detect
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
			m_output_numprob = 5 + m_class_num;
			m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
		}
		else if(m_algo_type == YOLO26)
		{
			m_output_numprob = 7;
			m_output_numbox = 300;
		}

		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}

protected:
	/**
	 * @description: 								rotated Non-Maximum Suppression
	 * @param {const vector<vector<float>>&} boxes	obb rotated boxes
	 * @param {const vector<float>&} scores			obb scores		
	 * @param {float} score_threshold				score threshold
	 * @param {float} nms_threshold					IOU threshold
	 * @param {vector<int>} & indices				output indices
	 * @return {*}
	 */
	void nms_rotated(const std::vector<std::vector<float>> & rboxes, const std::vector<float> & scores, float score_threshold, float nms_threshold, std::vector<int> & indices)
	{
		assert(rboxes.size() == scores.size());
		int N = rboxes.size();
		if (N == 0) { indices.clear(); return; }

		std::vector<int> sorted_idx = argsort_desc(scores);

		std::vector<bool> suppressed(N, false);
		indices.clear();
		indices.reserve(N);

		std::vector<float> a1(N), b1(N), c1(N);
		for (int i = 0; i < N; i++)
		{
			int idx = sorted_idx[i];
			float w = rboxes[idx][2], h = rboxes[idx][3], theta = rboxes[idx][6];
			float a = (w * w) / 12.0f, b = (h * h) / 12.0f;
			float cos_t = std::cos(theta), sin_t = std::sin(theta);
			float cos2 = cos_t * cos_t, sin2 = sin_t * sin_t;
			a1[i] = a * cos2 + b * sin2;
			b1[i] = a * sin2 + b * cos2;
			c1[i] = (a - b) * cos_t * sin_t;
		}

		for (int i = 0; i < N; i++)
		{
			if (suppressed[i]) continue;
			int idx_i = sorted_idx[i];
			indices.push_back(idx_i);

			float x1 = rboxes[idx_i][0], y1 = rboxes[idx_i][1];
			float ai = a1[i], bi = b1[i], ci = c1[i];
			float term1_i = std::max(ai * bi - ci * ci, 0.0f);

			for (int j = i + 1; j < N; j++)
			{
				if (suppressed[j]) continue;
				int idx_j = sorted_idx[j];

				float x2 = rboxes[idx_j][0], y2 = rboxes[idx_j][1];
				float aj = a1[j], bj = b1[j], cj = c1[j];

				float dx = x1 - x2, dy = y1 - y2;
				float a_sum = ai + aj, b_sum = bi + bj, c_sum = ci + cj;
				float den = a_sum * b_sum - c_sum * c_sum + 1e-7f;

				float t1 = (a_sum * dy * dy + b_sum * dx * dx) / den * 0.25f;
				float t2 = (c_sum * (-dx) * dy) / den * 0.5f;

				float term1_j = std::max(aj * bj - cj * cj, 0.0f);
				float t3_numer = a_sum * b_sum - c_sum * c_sum;
				float t3_denom = 4.0f * std::sqrt(term1_i * term1_j) + 1e-7f;
				float t3 = std::log(std::max(t3_numer / t3_denom + 1e-7f, 1e-7f)) * 0.5f;

				float bd = std::max(std::min(t1 + t2 + t3, 100.0f), 1e-7f);
				float hd = std::sqrt(std::max(1.0f - std::exp(-bd) + 1e-7f, 1e-7f));
				float iou = 1.0f - hd;

				if (iou > nms_threshold)
					suppressed[j] = true;
			}
		}
	}

	/**
	 * @description: 								arg sort by descent
	 * @param {const std::vector<float>&} vec		input vector
	 * @return {std::vector<int>}					sorted idx
	 */
	std::vector<int> argsort_desc(const std::vector<float>& scores) 
	{
		std::vector<int> sorted_idx(scores.size());
		for (int i = 0; i < scores.size(); ++i)		
		{
			sorted_idx[i] = i;
		}
		std::sort(sorted_idx.begin(), sorted_idx.end(), [&scores](int i, int j) {return scores[i] > scores[j]; });
		return sorted_idx;
	}
	
	/**
	 * @description: 									scale rotated boxes
	 * @param {std::vector<std::vector<float>>&} rboxes	obb rotated boxes
	 * @return {*}
	 */
	void regularize_rboxes(std::vector<std::vector<float>> & rboxes)
	 {
		for (auto& rbox : rboxes) 
		{
			float x = rbox[0], y = rbox[1], w = rbox[2], h = rbox[3], score = rbox[4], cls = rbox[5], t = rbox[6];
			float w_ = std::max(w, h);
			float h_ = std::min(w, h);
			float t_temp = (w > h) ? t : (t + M_PI/2);
			float t_ = fmod(t_temp, M_PI);
			if (t_ < 0) 
			{
				t_ += M_PI;
			}
			rbox = {x, y, w_, h_, score, cls, t_};
		}
	}

	/**
	 * @description: 									scale rotated boxes
	 * @param {std::vector<std::vector<float>>&} rboxes	obb rotated boxes
	 * @param {Size} size								output image shape
	 * @return {*}
	 */
	void scale_rboxes(std::vector<std::vector<float>> & rboxes, cv::Size size)
	{
		float gain = std::min(m_input_size.width * 1.0 / size.width, m_input_size.height * 1.0 / size.height);
		int pad_w = (m_input_size.width - size.width * gain) / 2;
		int pad_h = (m_input_size.height - size.height * gain) / 2;
		for (auto& rbox : rboxes)
		{
			rbox[0] -= pad_w;
			rbox[1] -= pad_h;
			rbox[0] /= gain;
			rbox[1] /= gain;
			rbox[2] /= gain;
			rbox[3] /= gain;
		}
	}
	/**
	 * @description: 								draw result
	 * @return {*}
	 */
	void draw_result()
	{
    	m_result = m_image.clone();
		for (int i = 0; i < m_output_obb.size(); i++)
		{
			OutputOBB output = m_output_obb[i];
			int idx = output.id;
			float score = output.score;
			cv::RotatedRect box_rotate = output.box_rotate;
			cv::Point2f vertices[4];
			box_rotate.points(vertices);
			for (int i = 0; i < 4; i++) 
			{
				cv::line(m_result, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
			}
			std::string label = "class" + std::to_string(idx) + ":" + cv::format("%.2f", score);
			cv::putText(m_result, label, cv::Point(vertices[0]), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		}
	}

	/**
	 * @description: class num
	 */	
	int m_class_num = 15;

	/**
	 * @description: model input image size
	 */
	cv::Size m_input_size = cv::Size(1024, 1024);

	/**
	 * @description: score threshold
	 */
	float m_score_threshold = 0.25;

	/**
	 * @description: IOU threshold
	 */
	float m_nms_threshold = 0.7;

	/**
	 * @description: letterbox related parameters
	 */
	cv::Vec4d m_params;

	/**
	 * @description:output detection size
	 */
	int m_output_numprob;

	/**
	 * @description: output bounding box num
	 */
	int m_output_numbox;

	/**
	 * @description: output feature map size
	 */
	int m_output_numdet;

	/**
	 * @description: model output on host
	 */
	float* m_output_host;

	/**
	 * @description: obb model output
	 */
	std::vector<OutputOBB> m_output_obb;
};
