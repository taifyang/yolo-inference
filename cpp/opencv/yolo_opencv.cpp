/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-07-07 16:27:40
 * @FilePath: \cpp\opencv\yolo_opencv.cpp
 * @Description: yolo算法的opencv推理框架实现
 */

#include "yolo_opencv.h"

void YOLO_OpenCV::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo = algo_type;

	if (algo_type == YOLOv8 && device_type == GPU)
	{
		std::cout << "OpenCV YOLOv8 not supported GPU yet!" << std::endl;
		std::exit(-1);
	}

	if(!std::filesystem::exists(model_path))
	{
		std::cout << "model not exists!" << std::endl;
		std::exit(-1);
	}
	m_net = cv::dnn::readNet(model_path);
	
	if (model_type != FP32 && model_type != FP16)
	{
		std::cout << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	
	if (device_type == CPU)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	if (device_type == GPU)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		if (model_type == FP32)
		{
			m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		}
		if (model_type == FP16)
		{
			m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		}
	}
}

void YOLO_OpenCV_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv8)
	{
		m_input_width = 224;
		m_input_height = 224;
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}
}

void YOLO_OpenCV_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv5 || m_algo == YOLOv7)
	{
		m_output_numprob = 5 + class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
	}
	if (m_algo == YOLOv6)
	{
		m_output_numprob = 5 + class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	if (m_algo == YOLOv8 || m_algo == YOLOv9)
	{
		m_output_numprob = 4 + class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	if(m_algo == YOLOv10)
	{
		m_output_numprob = 6;
		m_output_numbox = 300;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
}

void YOLO_OpenCV_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv5)
	{
		m_output_numprob = 37 + class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.segChannels * m_mask_params.segWidth * m_mask_params.segHeight;
	}
	if (m_algo == YOLOv8)
	{
		m_output_numprob = 36 + class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.segChannels * m_mask_params.segWidth * m_mask_params.segHeight;
	}
}

void YOLO_OpenCV_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo == YOLOv5)
	{
		//CenterCrop
		int crop_size = std::min(m_image.cols, m_image.rows);
		int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
		crop_image = m_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_height));

		//Normalize
		crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);
		cv::subtract(crop_image, cv::Scalar(0.406, 0.456, 0.485), crop_image);
		cv::divide(crop_image, cv::Scalar(0.225, 0.224, 0.229), crop_image);
	}
	if (m_algo == YOLOv8)
	{
		cv::cvtColor(m_image, crop_image, cv::COLOR_BGR2RGB);

		if (m_image.cols > m_image.rows)
			cv::resize(crop_image, crop_image, cv::Size(m_input_height * m_image.cols / m_image.rows, m_input_height));
		else
			cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_width * m_image.rows / m_image.cols));

		//CenterCrop
		int crop_size = std::min(crop_image.cols, crop_image.rows);
		int  left = (crop_image.cols - crop_size) / 2, top = (crop_image.rows - crop_size) / 2;
		crop_image = crop_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_height));

		//Normalize
		crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);
	}

	cv::dnn::blobFromImage(crop_image, m_input, 1, cv::Size(crop_image.cols, crop_image.rows), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Detect::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_width, m_input_height), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Segment::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_width, m_input_height), cv::Scalar(), true, false);
}	

void YOLO_OpenCV::process()
{
	m_net.setInput(m_input);
	m_net.forward(m_output, m_net.getUnconnectedOutLayersNames());
}

void YOLO_OpenCV_Classify::post_process()
{
	m_output_host = (float*)m_output[0].data;

	std::vector<float> scores;
	float sum = 0.0f;
	for (size_t i = 0; i < class_num; i++)
	{
		scores.push_back(m_output_host[i]);
		sum += exp(m_output_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	if (m_algo == YOLOv8)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
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
		if (m_algo == YOLOv5 || m_algo == YOLOv6 || m_algo == YOLOv7)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8 || m_algo == YOLOv9)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		if (m_algo == YOLOv10)
		{
			score = ptr[4];
			class_id = int(ptr[5]);
		}
		if (score < score_threshold)
			continue;

		cv::Rect box;
		if(m_algo == YOLOv5 || m_algo == YOLOv6 || m_algo == YOLOv7 || m_algo == YOLOv8 || m_algo == YOLOv9)
		{
			float x = ptr[0];
			float y = ptr[1];
			float w = ptr[2];
			float h = ptr[3];
			int left = int(x - 0.5 * w);
			int top = int(y - 0.5 * h);
			int width = int(w);
			int height = int(h);
			box = cv::Rect(left, top, width, height);
		}
		if (m_algo == YOLOv10)
		{
			box = cv::Rect(ptr[0], ptr[1], ptr[2] - ptr[0], ptr[3] - ptr[1]);
		}

		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	if(m_algo == YOLOv5 || m_algo == YOLOv6 || m_algo == YOLOv7 || m_algo == YOLOv8 || m_algo == YOLOv9)
	{
		std::vector<int> indices;
		nms(boxes, scores, score_threshold, nms_threshold, indices);
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
	}
	if (m_algo == YOLOv10)
	{
		m_output_det.clear();
		m_output_det.resize(boxes.size());
		for (int i = 0; i < boxes.size(); i++)
		{
			OutputDet output;
			output.id = class_ids[i];
			output.score = scores[i];
			output.box = boxes[i];
			m_output_det[i] = output;
		}
	}

	if(m_draw_result)
		draw_result(m_output_det);
}

void YOLO_OpenCV_Segment::post_process()
{
	m_output_host = (float*)m_output[0].data;

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id];
		}

		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);

		if (m_algo == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + class_num + 5, ptr + class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		if (m_algo == YOLOv8)
		{
			std::vector<float> temp_proto(ptr + class_num + 4, ptr + class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);

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
	m_mask_params.srcImgShape = m_image.size();
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), m_output[1], m_output_seg[i], m_mask_params);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}
