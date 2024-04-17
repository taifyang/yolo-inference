#include "yolov5_onnxruntime.h"
#include "utils.h"


YOLOv5_ONNXRuntime::YOLOv5_ONNXRuntime(std::string model_path, Device_Type device_type, Model_Type model_type)
{
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(12);//设置线程数
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);//启用模型优化策略

	if (device_type == GPU)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_options.AppendExecutionProvider_CUDA(cuda_option);
	}
	
	m_model = model_type;
	m_session = new Ort::Session(m_env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);

	Ort::AllocatorWithDefaultOptions allocator;

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output");
	}

	if (m_model == FP16)
	{
		m_inputs_fp16.resize(input_numel);
		m_outputs_fp16.resize(output_numel);
	}
	m_outputs_host = (float*)malloc(sizeof(float) * output_numel);
}


void YOLOv5_ONNXRuntime::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
	std::vector<cv::Mat> split_images;
	cv::split(letterbox, split_images);
	m_inputs.clear();
	for (size_t i = 0; i < letterbox.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_inputs.insert(m_inputs.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model == FP16)
	{
		for (size_t i = 0; i < input_numel; i++)
		{
			m_inputs_fp16[i] = float32_to_float16(m_inputs[i]);
		}
	}
}


void YOLOv5_ONNXRuntime::process()	
{
	//input_tensor
	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), input_width, input_height };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor{ nullptr };
	if(m_model == FP32 || m_model == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_inputs.data(), sizeof(float) * input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);		
	else if (m_model == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_inputs_fp16.data(), sizeof(uint16_t) * input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor)); 

	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), ort_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	//取output数据	
	if (m_model == FP32 || m_model == INT8)
	{
		m_outputs_host = const_cast<float*> (outputs[0].GetTensorData<float>());
	}
	else if (m_model == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + output_numel, m_outputs_fp16.begin());
		for (size_t i = 0; i < output_numel; i++)
		{
			m_outputs_host[i] = float16_to_float32(m_outputs_fp16[i]);
		}
	}
}


YOLOv5_ONNXRuntime::~YOLOv5_ONNXRuntime()
{
	m_session->release();
	m_env.release();
}

