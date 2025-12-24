/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2025-12-21 23:15:11
 * @Description: openvino segment source file for YOLO algorithm
 */

#include "yolo_openvino.h"

void YOLO_OpenVINO_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenVINO::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv5)
	{
		m_output_numprob = 37 + m_class_num;
		m_output_numbox = 3 * (m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_output_numprob = 36 + m_class_num;
		m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;
}

void YOLO_OpenVINO_Segment::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));
	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_size.width, m_input_size.height), cv::Scalar(), true, false);
}

void YOLO_OpenVINO_Segment::process()
{
	ov::Tensor input_tensor(m_input_port.get_element_type(), m_input_port.get_shape(), m_input.ptr(0)); //Create tensor from external memory
	m_infer_request.set_input_tensor(input_tensor); // Set input tensor for model with one input
	m_infer_request.infer(); //Start inference
	m_output0_host = (float*)m_infer_request.get_output_tensor(0).data();  //Get the inference result 
	m_output1_host = (float*)m_infer_request.get_output_tensor(1).data();
}

void YOLO_OpenVINO_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;
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

		if (score < m_score_threshold)
			continue;

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
		cv::Rect box = cv::Rect(left, top, width, height);

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
	}

	scale_boxes(boxes, m_image.size());

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	m_output_seg.clear();
	m_output_seg.resize(indices.size());
	std::vector<std::vector<float>> temp_mask_proposals;
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

	m_mask_params.params = m_params;
	m_mask_params.input_shape = m_image.size();
	int shape[4] = { 1, m_mask_params.seg_channels, m_mask_params.seg_width, m_mask_params.seg_height, };
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1_host, m_output1_host + m_output_numseg, (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, m_output_seg[i], m_mask_params, m_algo_type);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}
