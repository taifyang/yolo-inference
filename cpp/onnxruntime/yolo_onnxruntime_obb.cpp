/* 
 * @Author: taifyang
 * @Date: 2026-01-08 22:26:25
 * @LastEditTime: 2026-01-17 22:20:19
 * @Description: source file for YOLO onnxruntime obb
 */

#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_OBB::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_OBB::init(algo_type, device_type, model_type, model_path);
	
	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0.resize(m_output_numdet);
	}
}

void YOLO_ONNXRuntime_OBB::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

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

void YOLO_ONNXRuntime_OBB::process()
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

void YOLO_ONNXRuntime_OBB::post_process()
{
	std::vector<std::vector<float>> rboxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0.data() + i * m_output_numprob;

		int class_id;
		float score;
		std::vector<float> rbox(7);
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
			if (score < m_score_threshold)
				continue;
			float angle = ptr[m_class_num + 4];
			rbox = { ptr[0], ptr[1], ptr[2], ptr[3], score, class_id, angle};
		}
		else if(m_algo_type == YOLO26)
		{
			if (ptr[4] < m_score_threshold)
				continue;
			std::copy(ptr, ptr + 7, rbox.begin());
		}

		rboxes.push_back(rbox);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}
	
	std::vector<int> indices;
	nms_rotated(rboxes, scores, m_score_threshold, m_nms_threshold, indices);
	std::vector<std::vector<float>> boxes_nms(indices.size());
	for(int i=0; i<indices.size(); i++)	
	{
		boxes_nms[i] = rboxes[indices[i]];
	}

	regularize_rboxes(boxes_nms);
	
	scale_rboxes(boxes_nms, m_image.size());

	m_output_obb.clear();
	m_output_obb.resize(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		OutputOBB output;
		output.score = boxes_nms[i][4];
		output.id = boxes_nms[i][5];
		float angle = boxes_nms[i][6] * 180 / CV_PI;
		output.box_rotate = cv::RotatedRect(cv::Point2f(boxes_nms[i][0], boxes_nms[i][1]), cv::Size2f(boxes_nms[i][2], boxes_nms[i][3]), angle);
		m_output_obb[i] = output;
	}

	if(m_draw_result)
		draw_result(m_output_obb);
}
