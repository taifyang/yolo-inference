/* 
 * @Author: taifyang
 * @Date: 2026-01-08 22:26:25
 * @LastEditTime: 2026-08-25 22:11:56
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
	
	m_input.resize(m_input_numel);
	m_input_node_dims = { 1, 3, m_input_size.height, m_input_size.width };
	m_split_images.resize(3);
	m_output0.resize(m_output_numdet);

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
	}
}

void YOLO_ONNXRuntime_Depth::pre_process()
{
	LetterBox(m_image, m_letterbox, m_params, m_input_size);

	cv::cvtColor(m_letterbox, m_letterbox, cv::COLOR_BGR2RGB);
	m_letterbox.convertTo(m_letterbox, CV_32FC3, 1.0f / 255.0f);
	
	cv::split(m_letterbox, m_split_images);
	float* dst = m_input.data();
	for (int i = 0; i < 3; ++i)
	{
		memcpy(dst, m_split_images[i].ptr<float>(0), m_split_images[i].total() * sizeof(float));
		dst += m_split_images[i].total();
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
	
	if(m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(*m_memory_info, m_input.data(), sizeof(float) * m_input_numel, m_input_node_dims.data(), m_input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);		
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(*m_memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, m_input_node_dims.data(), m_input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor)); 
	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output0_host = const_cast<float*>(outputs[0].GetTensorData<float>());
		memcpy(m_output0.data(), m_output0_host, m_output_numdet * sizeof(float));
	}
	else if (m_model_type == FP16)
	{
		uint16_t* output0_fp16 = const_cast<uint16_t*>(outputs[0].GetTensorData<uint16_t>());
		for (size_t i = 0; i < m_output_numdet; i++)
		{
			m_output0[i] = float16_to_float32(output0_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Depth::post_process()
{
	cv::Mat depth(m_input_size, CV_32FC1, m_output0.data());
	m_depth = cv::Mat::zeros(m_image.size(), CV_32FC1);		
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	scale_mask(depth, m_depth, m_input_size, m_image.size());
	
	if(m_draw_result)
		draw_result();
}
