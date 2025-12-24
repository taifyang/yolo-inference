/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2025-12-21 23:05:17
 * @Description: opencv detect source file for YOLO algorithm
 */

#include "yolo_opencv.h"

void YOLO_OpenCV_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv3 && algo_type != YOLOv4 && algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLOv13)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv3 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		m_output_numprob = 4 + m_class_num;
		m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
	}
	else if (m_algo_type == YOLOv4)
	{
		m_output_numprob = 4 + m_class_num;
		m_output_numbox = 3 * (m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32);
	}
	else if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		m_output_numprob = 5 + m_class_num;
		m_output_numbox = 3 * (m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32);
	}

	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
}

void YOLO_OpenCV_Detect::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_size.width, m_input_size.height), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Detect::post_process()
{
	m_output_host = (float*)m_output[0].data;

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv3 || m_algo_type == YOLOv4 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		else if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		
		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if(m_algo_type == YOLOv3 || m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
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
		else if (m_algo_type == YOLOv4)
		{
			float x1 = ptr[0] * m_input_size.width;
			float y1 = ptr[1] * m_input_size.height;
			float x2 = ptr[2] * m_input_size.width;
			float y2 = ptr[3] * m_input_size.height;
			int left = int(x1) > 0 ? int(x1) : 0;
			int top = int(y1) > 0 ? int(y1) : 0;
			int width = int(x2 - x1) > 0 ? int(x2 - x1) : 0;
			int height = int(y2 - y1)> 0 ? int(y2 - y1) : 0;
			box = cv::Rect(left, top, width, height);
		}
		else if (m_algo_type == YOLOv10)
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
	}

	scale_boxes(boxes, m_image.size());

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	m_output_det.clear();
	m_output_det.resize(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		OutputDet output;
		output.id = class_ids[idx];
		output.score = scores[idx];
		output.box = boxes[idx];
		m_output_det[i] = output;
	}

	if(m_draw_result)
		draw_result(m_output_det);
}
