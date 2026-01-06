/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-03 20:34:03
 * @Description: source file for YOLO libtorch inference
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}
  
	try
	{
		m_module = torch::jit::load(model_path);
	}
	catch (const c10::Error& e) 
	{
		std::cerr << "libtorch load model failed!" << std::endl;
		std::exit(-1);
	}

	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	m_module.to(m_device);

	if (model_type != FP32 && model_type != FP16)
	{
		std::cerr << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	if (model_type == FP16 && device_type != GPU)
	{
		std::cerr << "FP16 only support GPU!" << std::endl;
		std::exit(-1);
	}
	m_model_type = model_type;
	if (model_type == FP16)
	{
		m_module.to(torch::kHalf);
	}
}
