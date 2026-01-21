/* 
 * @Author: taifyang
 * @Date: 2026-01-08 22:18:50
 * @LastEditTime: 2026-01-17 20:33:31
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
		std::vector<int> sorted_idx = argsort_desc(scores);
		std::vector<std::vector<float>> rboxes_sorted(rboxes.size());
		for(int i=0; i<rboxes.size(); i++)	
		{
			rboxes_sorted[i] = rboxes[sorted_idx[i]];
		}
		Eigen::MatrixXf boxes = vector2Eigen(rboxes_sorted);
		Eigen::MatrixXf ious = triu_k1(probiou(boxes, boxes, 1e-7));
		std::vector<int> pick = find_indices_max_below_threshold(ious, nms_threshold);
		indices.resize(pick.size());
		for(int i=0; i<pick.size(); i++)
		{
			indices[i] = sorted_idx[pick[i]];
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
	 * @description: 											transform vector to eigen matrix
	 * @param {const std::vector<std::vector<float>> &} vec		input vector
	 * @return {Eigen::MatrixXf}								output matrix
	 */
	Eigen::MatrixXf vector2Eigen(const std::vector<std::vector<float>> & vec) 
	{
		Eigen::MatrixXf mat(vec.size(), vec[0].size());
		for (int i = 0; i < vec.size(); ++i) 
			for (int j = 0; j < vec[0].size(); ++j) 
				mat(i, j) = vec[i][j];
		return mat;
	}

	/**
	 * @description: 															get covariance matrix
	 * @param {Eigen::MatrixXf} rboxes											rotated rboxes
	 * @return {std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>}	output tuple of matrix
	 */
	std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> get_covariance_matrix(Eigen::MatrixXf rboxes)
	{
		Eigen::MatrixXf xy_square = rboxes.block(0, 2, rboxes.rows(), 2).array().square() / 12.0;
		Eigen::MatrixXf c_col = rboxes.col(rboxes.cols() - 1);

		Eigen::MatrixXf gbbs(rboxes.rows(), 3);
		gbbs << xy_square, c_col;

		Eigen::MatrixXf a = gbbs.col(0);
		Eigen::MatrixXf b = gbbs.col(1);
		Eigen::MatrixXf c = gbbs.col(2);

		Eigen::MatrixXf cos = c.array().cos();
		Eigen::MatrixXf sin = c.array().sin();
		Eigen::MatrixXf cos2 = cos.array().square();
		Eigen::MatrixXf sin2 = sin.array().square();

		return {a.array() * cos2.array() + b.array() * sin2.array(), a.array() * sin2.array() + b.array() * cos2.array(), (a.array() - b.array()) * cos.array() * sin.array()};
	}

	/**
	 * @description: 					probiou
	 * @param {Eigen::MatrixXf&} obb1	obb1
	 * @param {Eigen::MatrixXf&} obb2	obb2
	 * @param {float} eps				eps
	 * @return {Eigen::MatrixXf}		output matrix
	 */
	Eigen::MatrixXf probiou(Eigen::MatrixXf& obb1, Eigen::MatrixXf& obb2, float eps)
	{
		int N = obb1.rows();
		Eigen::MatrixXf x1 = obb1.block(0, 0, N, 1), y1 = obb1.block(0, 1, N, 1); 
		Eigen::MatrixXf x2 = obb2.col(0).transpose(), y2 = obb2.col(1).transpose();

		auto [a1, b1, c1] = get_covariance_matrix(obb1);
		auto [a2_raw, b2_raw, c2_raw] = get_covariance_matrix(obb1);
		Eigen::MatrixXf a2 = a2_raw.transpose(), b2 = b2_raw.transpose(), c2 = c2_raw.transpose();
			
		Eigen::ArrayXXf a_sum = a1.array().replicate(1, N) + a2.array().replicate(N, 1);
		Eigen::ArrayXXf b_sum = b1.array().replicate(1, N) + b2.array().replicate(N, 1);
		Eigen::ArrayXXf c_sum = c1.array().replicate(1, N) + c2.array().replicate(N, 1);
		Eigen::ArrayXXf x_diff = x1.array().replicate(1, N) - x2.array().replicate(N, 1);
		Eigen::ArrayXXf y_diff = y1.array().replicate(1, N) - y2.array().replicate(N, 1);
		Eigen::ArrayXXf denominator = a_sum * b_sum - c_sum.square() + eps;
		Eigen::ArrayXXf t1 = (a_sum * y_diff.square() + b_sum * x_diff.square()) / denominator * 0.25f;
		Eigen::ArrayXXf t2 = (c_sum * (-x_diff) * y_diff) / denominator * 0.5f; 

		Eigen::ArrayXXf term1 = (a1.array() * b1.array() - c1.array().square()).cwiseMax(0.0f);
		Eigen::ArrayXXf term2 = (a2.array() * b2.array() - c2.array().square()).cwiseMax(0.0f);
		Eigen::ArrayXXf numerator_t3 = a_sum * b_sum - c_sum.square();
		Eigen::ArrayXXf term1_term2 = term1.replicate(1, N) * term2.replicate(N, 1);
		Eigen::ArrayXXf denominator_t3 = 4.0f * term1_term2.sqrt() + eps;
		Eigen::ArrayXXf log_input = numerator_t3 / denominator_t3 + eps;
		Eigen::ArrayXXf t3 = log_input.log() * 0.5f;
		Eigen::ArrayXXf bd = (t1.array() + t2.array() + t3).cwiseMax(eps).cwiseMin(100.0f);
		Eigen::ArrayXXf hd = (1.0f - (-bd).exp() + eps).cwiseMax(eps).sqrt();

		return (1 - hd).matrix();
	}

	/**
	 * @description: 							triu k1
	 * @param {const Eigen::MatrixXf&} mat		input matrix
	 * @return {Eigen::MatrixXf}				output matrix
	 */
	Eigen::MatrixXf triu_k1(const Eigen::MatrixXf& mat) 
	{
		Eigen::MatrixXf upper = mat.triangularView<Eigen::Upper>();  
		Eigen::MatrixXf result = upper;
		int min_dim = std::min(mat.rows(), mat.cols());
		for(int i = 0; i < min_dim; i++) 
		{
			result(i, i) = 0.0; 
		}
		return result;
	}

	/**
	 * @description: 							find indices max below threshold
	 * @param {const Eigen::MatrixXf&} ious		ious
	 * @param {float} threshold					threshold
	 * @return {std::vector<int>}				indices
	 */
	std::vector<int> find_indices_max_below_threshold(const Eigen::MatrixXf& ious, float threshold) 
	{	
		std::vector<int> indices;
		Eigen::VectorXf col_max = ious.colwise().maxCoeff();	
		for (int i = 0; i < ious.cols(); ++i) 
			if (col_max(i) < threshold) 
				indices.push_back(i);	
		return indices;
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
	 * @param {std::vector<OutputOBB>} output_obb	obb model output
	 * @return {*}
	 */
	void draw_result(std::vector<OutputOBB> output_obb)
	{
    	m_result = m_image.clone();
		for (int i = 0; i < output_obb.size(); i++)
		{
			OutputOBB output = output_obb[i];
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
