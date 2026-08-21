/* 
 * @Author: taifyang
 * @Date: 2025-12-21 21:51:23
 * @LastEditTime: 2026-08-09 23:06:37
 * @Description: source file for YOLO libtorch depth estimation
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch_Depth::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
	YOLO_Depth::init(algo_type, device_type, model_type, model_path);
}

void YOLO_Libtorch_Depth::pre_process()
{
	YOLO_Libtorch_Detect::pre_process();
}

void YOLO_Libtorch_Depth::process()
{	
	m_output = m_module.forward(m_input);
	torch::Tensor pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
	m_output0.assign(pred.data_ptr<float>(), pred.data_ptr<float>() + m_output_numdet);
}

void YOLO_Libtorch_Depth::post_process()
{
	m_depth = cv::Mat::zeros(m_input_size, CV_32FC1);	
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	std::copy(m_output0.begin(), m_output0.end(), (float*)m_depth.data);
	scale_mask(m_depth, m_input_size, m_image.size());
	
	if(m_draw_result)
		draw_result();
}