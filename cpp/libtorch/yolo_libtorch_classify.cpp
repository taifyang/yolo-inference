/* 
 * @Author: taifyang
 * @Date: 2025-12-21 21:47:22
 * @LastEditTime: 2026-01-19 19:11:00
 * @Description: source file for YOLO libtorch classification
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
	YOLO_Classify::init(algo_type, device_type, model_type, model_path);
}

void YOLO_Libtorch_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo_type == YOLOv5)
	{
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));

		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);		
	}

	cv::cvtColor(crop_image, crop_image, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		input = torch::from_blob(crop_image.data, { 1, crop_image.rows, crop_image.cols, crop_image.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
	{
		crop_image.convertTo(crop_image, CV_16FC3);
		input = torch::from_blob(crop_image.data, { 1, crop_image.rows, crop_image.cols, crop_image.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Classify::process()
{
	m_output = m_module.forward(m_input);

	torch::Tensor pred;
	if (m_algo_type == YOLOv5)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		else if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		else if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}

	m_output0.assign(pred.data_ptr<float>(), pred.data_ptr<float>() + m_class_num);
}

void YOLO_Libtorch_Classify::post_process()
{
	std::vector<float> scores(m_class_num); 
	float sum = 0.0f;
	for (size_t i = 0; i < m_class_num; i++)
	{
		scores[i] = m_output0[i];
		sum += exp(m_output0[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo_type == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
}
