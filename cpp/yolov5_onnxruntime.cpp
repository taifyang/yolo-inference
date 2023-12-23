#include "yolov5_onnxruntime.h"
#include "utils.h"


YOLOv5_ONNXRuntime::YOLOv5_ONNXRuntime(std::string model_path, Device_Type device_type, Model_Type model_type)
{
	//初始化OrtApi
	m_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION); 	

	//初始化OrtEnv
	OrtEnv* env;
	m_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yolov5", &env);

	//初始化session_options
	OrtSessionOptions* session_options;
	m_ort->CreateSessionOptions(&session_options);
	if (device_type == GPU)
	{
		OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	}

	assert(("unsupported model type!", model_type == FP32 || model_type == FP16));
	m_model = model_type;

	//初始化session
#ifdef _WIN32
	m_ort->CreateSession(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options, &m_session);
#else
	g_ort->CreateSession(env, model_path.c_str(), session_options, &m_session);
#endif

	OrtAllocator* allocator;
	m_ort->GetAllocatorWithDefaultOptions(&allocator);
	size_t input_count, output_count;
	m_ort->SessionGetInputCount(m_session, &input_count);
	m_ort->SessionGetOutputCount(m_session, &output_count);

	char* input_name;
	for (size_t i = 0; i < input_count; i++)
	{
		m_ort->SessionGetInputName(m_session, i, allocator, &input_name);
		m_input_names.push_back(input_name);
	}
	char* output_name;
	for (size_t i = 0; i < output_count; i++)
	{
		m_ort->SessionGetOutputName(m_session, i, allocator, &output_name);
		m_output_names.push_back(output_name);
	}

	m_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_memory_info);

	if (m_model == FP16)
		m_inputs_fp16 = (uint16_t*)malloc(sizeof(uint16_t) * input_numel);
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
	m_inputs = (float*)malloc(sizeof(float) * input_numel);
	for (size_t i = 0; i < letterbox.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		std::copy(split_image_data.begin(), split_image_data.end(), m_inputs + i * split_image_data.size());
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
	OrtValue* input_tensor = NULL;
	int64_t input_shape[4] = { 1, m_image.channels(), input_width, input_height }; 
	if(m_model == FP32)
		m_ort->CreateTensorWithDataAsOrtValue(m_memory_info, m_inputs, sizeof(float) * input_numel, input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
	else if (m_model == FP16)
		m_ort->CreateTensorWithDataAsOrtValue(m_memory_info, m_inputs_fp16, sizeof(uint16_t) * input_numel, input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &input_tensor);
	
	//推理
	OrtValue* output_tensor = NULL;
	m_ort->Run(m_session, NULL, m_input_names.data(), (const OrtValue* const*)& input_tensor, m_input_names.size(), m_output_names.data(), m_output_names.size(), &output_tensor);

	//取output数据	
	if (m_model == FP32)
	{
		m_ort->GetTensorMutableData(output_tensor, (void**)& m_outputs);
		m_outputs_host = m_outputs;
	}
	else if (m_model == FP16)
	{
		m_ort->GetTensorMutableData(output_tensor, (void**)& m_outputs_fp16);
		for (size_t i = 0; i < output_numel; i++)
		{
			m_outputs_host[i] = float16_to_float32(m_outputs_fp16[i]);
		}
	}
}

