/* 
 * @Author: taifyang
 * @Date: 2025-12-21 22:19:47
 * @LastEditTime: 2026-01-17 19:07:03
 * @Description: 
 */
/* 
 * @Author: taifyang
 * @Date: 2025-12-21 22:19:47
 * @LastEditTime: 2026-01-06 12:58:31
 * @Description: source file for YOLO onnxruntime classification
 */

#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_Classify::init(algo_type, device_type, model_type, model_path);
	
	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0.resize(m_class_num);
	}
}

void YOLO_ONNXRuntime_Classify::pre_process()
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

	std::vector<cv::Mat> split_images;
	cv::split(crop_image, split_images);
	m_input.clear();
	for (size_t i = 0; i < crop_image.channels(); ++i)
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

void YOLO_ONNXRuntime_Classify::process()
{
	Ort::Value input_tensor{ nullptr };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_size.width, m_input_size.height };

	if (m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor)); 
	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output0_host = const_cast<float*>(outputs[0].GetTensorData<float>());
		m_output0.assign(m_output0_host, m_output0_host + m_class_num);
	}
	else if (m_model_type == FP16)
	{
		uint16_t* output0_fp16 = const_cast<uint16_t*>(outputs[0].GetTensorData<uint16_t>());
		m_output0_fp16.assign(output0_fp16, output0_fp16 + m_class_num);
		for (size_t i = 0; i < m_class_num; i++)
		{
			m_output0[i] = float16_to_float32(m_output0_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Classify::post_process()
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