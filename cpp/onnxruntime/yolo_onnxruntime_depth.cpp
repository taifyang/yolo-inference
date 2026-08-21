/* 
 * @Author: taifyang
 * @Date: 2026-01-08 22:26:25
 * @LastEditTime: 2026-08-09 22:39:37
 * @Description: source file for YOLO onnxruntime depth estimation
 */

#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_Depth::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_Depth::init(algo_type, device_type, model_type, model_path);
	
	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0.resize(m_output_numdet);
	}
}

void YOLO_ONNXRuntime_Depth::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, m_input_size);

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
	
	std::vector<cv::Mat> split_images;
	cv::split(letterbox, split_images);
	m_input.clear();
	for (size_t i = 0; i < letterbox.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_input.insert(m_input.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model_type == FP16)
	{
		for (size_t i = 0; i < m_input_numel; i++)
		{
			m_input_fp16[i] = float32_to_float16(m_input[i]);
		}
	}
}

void YOLO_ONNXRuntime_Depth::process()
{
	Ort::Value input_tensor{ nullptr };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_size.width, m_input_size.height };
	
	if(m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);		
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor)); 
	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output0_host = const_cast<float*>(outputs[0].GetTensorData<float>());
		m_output0.assign(m_output0_host, m_output0_host + m_output_numdet);
	}
	else if (m_model_type == FP16)
	{
		uint16_t* output0_fp16 = const_cast<uint16_t*>(outputs[0].GetTensorData<uint16_t>());
		m_output0_fp16.assign(output0_fp16, output0_fp16 + m_output_numdet);
		for (size_t i = 0; i < m_output_numdet; i++)
		{
			m_output0[i] = float16_to_float32(m_output0_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Depth::post_process()
{
	m_depth = cv::Mat::zeros(m_input_size.width, m_input_size.height, CV_32FC1);	
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	std::copy(m_output0.begin(), m_output0.end(), (float*)m_depth.data);
	scale_mask(m_depth, cv::Size(m_input_size.width , m_input_size.height), m_image.size());
	
	if(m_draw_result)
		draw_result();
}
