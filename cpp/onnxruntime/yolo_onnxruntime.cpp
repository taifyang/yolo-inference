/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-05 09:29:19
 * @Description: source file for YOLO onnxruntime inference
 */

#include <thread>
#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency() / 2);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (device_type == GPU)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_option);
	}
	
	m_model_type = model_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}

#ifdef _WIN32
	m_session = new Ort::Session(m_env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
#endif

#ifdef __linux__
	m_session = new Ort::Session(m_env, model_path.c_str(), session_options);
#endif
	if(m_session == nullptr)
	{
		std::cerr << "onnxruntime session create failed!" << std::endl;
		std::exit(-1);
	}

	m_input_names.push_back("images");
	m_output_names.push_back("output0");
}

void YOLO_ONNXRuntime::release()
{
	m_session->release();
	m_env.release();
}
