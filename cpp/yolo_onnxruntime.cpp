#include "yolo_onnxruntime.h"
#include "utils.h"


void YOLO_ONNXRuntime::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	m_algo = algo_type;
	if (m_algo == YOLOv5)
	{
		m_output_numprob = 5 + num_classes;
		m_output_numbox = 3 * (input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32);
		m_output_numel = 1 * m_output_numprob * m_output_numbox;
	}
	if (m_algo == YOLOv8)
	{
		m_output_numprob = 4 + num_classes;
		m_output_numbox = input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32;
		m_output_numel = 1 * m_output_numprob * m_output_numbox;
	}

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(12);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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

#ifdef _WIN32
	m_session = new Ort::Session(m_env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
#endif

#ifdef __linux__
	m_session = new Ort::Session(m_env, model_path.c_str(), session_options);
#endif

	Ort::AllocatorWithDefaultOptions allocator;

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output0");
	}

	if (m_model == FP16)
	{
		m_inputs_fp16.resize(input_numel);
		m_outputs_fp16.resize(m_output_numel);
	}
	m_outputs_host = (float*)malloc(sizeof(float) * m_output_numel);
}


void YOLO_ONNXRuntime::pre_process()
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


void YOLO_ONNXRuntime::process()	
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

	//output_tensor
	if (m_model == FP32 || m_model == INT8)
	{
		m_outputs_host = const_cast<float*> (outputs[0].GetTensorData<float>());
	}
	else if (m_model == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + m_output_numel, m_outputs_fp16.begin());
		for (size_t i = 0; i < m_output_numel; i++)
		{
			m_outputs_host[i] = float16_to_float32(m_outputs_fp16[i]);
		}
	}
}


void YOLO_ONNXRuntime::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_outputs_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
			score = classes_scores[class_id];
		}
		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		std::string label = class_names[class_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);
		draw_result(m_result, label, box);
	}
}


void YOLO_ONNXRuntime::release()
{
	m_session->release();
	m_env.release();
}

