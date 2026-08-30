/* 
 * @Author: taifyang
 * @Date: 2026-01-03 19:35:16
 * @LastEditTime: 2026-08-29 16:24:10
 * @Description: source file for YOLO onnxruntime semantic segmentation
 */

 #include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_Semantic::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_Semantic::init(algo_type, device_type, model_type, model_path);

	m_output0.resize(m_output_numdet);
	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0_fp16.resize(m_output_numdet);
	}
}

void YOLO_ONNXRuntime_Semantic::pre_process()
{
	YOLO_ONNXRuntime_Detect::pre_process();
}

void YOLO_ONNXRuntime_Semantic::process()
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

	m_output0_host = const_cast<uint8_t*>(outputs[0].GetTensorData<uint8_t>());
}

void YOLO_ONNXRuntime_Semantic::post_process()
{
	m_mask = cv::Mat::zeros(m_image.size(), CV_8UC1);
	cv::Mat output = cv::Mat::zeros(m_input_size, CV_8UC1);
	std::copy(m_output0_host, m_output0_host + m_output_numdet, output.data);
	scale_mask(output, m_mask, m_input_size, m_image.size());

	if (m_draw_result)
	{
		draw_result();
	}
}
