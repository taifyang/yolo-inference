/* 
 * @Author: taifyang
 * @Date: 2025-12-21 21:51:23
 * @LastEditTime: 2026-01-06 11:12:15
 * @Description: source file for YOLO libtorch segmentation
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
	YOLO_Segment::init(algo_type, device_type, model_type, model_path);
}

void YOLO_Libtorch_Segment::pre_process()
{
	YOLO_Libtorch_Detect::pre_process();
}

void YOLO_Libtorch_Segment::process()
{	
	m_output = m_module.forward(m_input);
	torch::Tensor pred0, pred1;
	pred0 = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
	pred1 = m_output.toTuple()->elements()[1].toTensor().to(torch::kFloat).to(at::kCPU);
	m_output0.assign(pred0.data_ptr<float>(), pred0.data_ptr<float>() + m_output_numdet);
	m_output1.assign(pred1.data_ptr<float>(), pred1.data_ptr<float>() + m_output_numseg);
}

void YOLO_Libtorch_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0.data() + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		else if (m_algo_type == YOLO26)
		{
			score = ptr[4];
			class_id = int(ptr[5]);
		}

		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if (m_algo_type == YOLOv5 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float x = ptr[0];
			float y = ptr[1];
			float w = ptr[2];
			float h = ptr[3];
			int left = int(x - 0.5 * w) > 0 ? int(x - 0.5 * w) : 0;
			int top = int(y - 0.5 * h) > 0 ? int(y - 0.5 * h) : 0;
			int width = int(w) > 0 ? int(w) : 0;
			int height = int(h)> 0 ? int(h) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
		}
		else if (m_algo_type == YOLO26)
		{
			int left = int(ptr[0]) > 0 ? int(ptr[0]) : 0;
			int top = int(ptr[1]) > 0 ? int(ptr[1]) : 0;
			int width = int(ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			int height = int(ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
		}

		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
		
		if (m_algo_type == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 5, ptr + m_class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 4, ptr + m_class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
		else if(m_algo_type == YOLO26)
		{
			std::vector<float> temp_proto(ptr + 6, ptr + 38);
			picked_proposals.push_back(temp_proto);
		}
	}

	scale_boxes(boxes, m_image.size());

	std::vector<std::vector<float>> temp_mask_proposals;
	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		std::vector<int> indices;
		nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);
		m_output_seg.clear();
		m_output_seg.resize(indices.size());
		cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
		for (int i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			OutputSeg output;
			output.id = class_ids[idx];
			output.score = scores[idx];
			output.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			m_output_seg[i] = output;
		}
	}
	else if(m_algo_type == YOLO26)
	{
		m_output_seg.clear();
		m_output_seg.resize(boxes.size());
		cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
		for (int i = 0; i < boxes.size(); ++i)
		{
			OutputSeg output;
			output.id = class_ids[i];
			output.score = scores[i];
			output.box = boxes[i] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[i]);
			m_output_seg[i] = output;
		}
	} 

	m_mask_params.params = m_params;
	m_mask_params.input_shape = m_image.size();
	int shape[4] = { 1, m_mask_params.seg_channels, m_mask_params.seg_width, m_mask_params.seg_height, };
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1.begin(), m_output1.end(), (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, m_output_seg[i], m_mask_params, m_algo_type);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}