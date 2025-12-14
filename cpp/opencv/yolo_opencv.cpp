/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-12-02 19:51:26
 * @FilePath: \cpp\opencv\yolo_opencv.cpp
 * @Description: opencv inference source file for YOLO algorithm
 */

#include "yolo_opencv.h"

void YOLO_OpenCV::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}

	m_net = cv::dnn::readNet(model_path);
	if(m_net.empty())
	{
		std::cerr << "opencv read net failed!" << std::endl;
		std::exit(-1);
	}
	
	if (model_type != FP32 && model_type != FP16)
	{
		std::cerr << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	
	if (device_type == CPU)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	else if (device_type == GPU)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		if (model_type == FP32)
		{
			m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		}
		else if (model_type == FP16)
		{
			m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		}
	}
}

void YOLO_OpenCV_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_input_size.width = 224;
		m_input_size.height = 224;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}
}

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

void YOLO_OpenCV_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv3 && algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

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

void YOLO_OpenCV_Classify::pre_process()
{
	cv::Mat crop_image;

	if (m_algo_type == YOLOv5)
	{
#ifndef OPENCV_WITH_CUDA
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
#else
		cv::cuda::GpuMat gpu_image, gpu_crop_image, gpu_cvt_image;
    	                gpu_image.upload(m_image);
		int crop_size = std::min(m_image.cols, m_image.rows);
		int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
		gpu_crop_image = gpu_image(cv::Rect(left, top, crop_size, crop_size));
		cv::cuda::resize(gpu_crop_image, gpu_crop_image, cv::Size(m_input_size.width, m_input_size.height));
		gpu_crop_image.convertTo(gpu_cvt_image, CV_32FC3, 1. / 255.);
		cv::cuda::subtract(gpu_cvt_image, cv::Scalar(0.406, 0.456, 0.485), gpu_cvt_image);
		cv::cuda::divide(gpu_cvt_image, cv::Scalar(0.225, 0.224, 0.229), gpu_cvt_image);
		gpu_cvt_image.download(crop_image);
#endif
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{	
#ifndef OPENCV_WITH_CUDA	
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
#else
		cv::cuda::GpuMat gpu_image, gpu_crop_image, gpu_cvt_image;
		gpu_image.upload(m_image);
		if (m_image.cols > m_image.rows)
			cv::cuda::resize(gpu_image, gpu_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::cuda::resize(gpu_image, gpu_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));
		int crop_size = std::min(crop_image.cols, crop_image.rows);
		int left = (crop_image.cols - crop_size) / 2, top = (crop_image.rows - crop_size) / 2;
		gpu_crop_image = gpu_image(cv::Rect(left, top, crop_size, crop_size));
		cv::cuda::resize(gpu_crop_image, gpu_crop_image, cv::Size(m_input_size.width, m_input_size.height));
		gpu_crop_image.convertTo(gpu_cvt_image, CV_32FC3, 1. / 255.);
		gpu_cvt_image.download(crop_image);
#endif
	}

	cv::dnn::blobFromImage(crop_image, m_input, 1, cv::Size(crop_image.cols, crop_image.rows), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Detect::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_size.width, m_input_size.height), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Segment::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::dnn::blobFromImage(letterbox, m_input, 1. / 255., cv::Size(m_input_size.width, m_input_size.height), cv::Scalar(), true, false);
}	

void YOLO_OpenCV::process()
{
	m_net.setInput(m_input);
	m_net.forward(m_output, m_net.getUnconnectedOutLayersNames());
}

void YOLO_OpenCV_Classify::post_process()
{
	m_output_host = (float*)m_output[0].data;

	std::vector<float> scores(m_class_num); 
	float sum = 0.0f;
	for (size_t i = 0; i < m_class_num; i++)
	{
		scores[i] = m_output_host[i];
		sum += exp(m_output_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo_type == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
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
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), m_output[1], m_output_seg[i], m_mask_params, m_algo_type);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}
