/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-03 20:37:29
 * @Description: source file for YOLO opencv inference 
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

void YOLO_OpenCV::process()
{
	m_net.setInput(m_input);
	m_net.forward(m_output, m_net.getUnconnectedOutLayersNames());
}
