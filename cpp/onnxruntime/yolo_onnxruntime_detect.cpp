/* 
 * @Author: taifyang
 * @Date: 2025-12-21 22:21:42
 * @LastEditTime: 2026-10-17 23:57:05
 * @Description: source file for YOLO onnxruntime detection
 */

#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv3 && algo_type != YOLOv4 && algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLOv13 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_Detect::init(algo_type, device_type, model_type, model_path);

	m_input.resize(m_input_numel);
	m_input_node_dims = { 1, 3, m_input_size.height, m_input_size.width };
	m_split_images.resize(3);
	m_output0.resize(m_output_numdet);

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
	}
}

void YOLO_ONNXRuntime_Detect::pre_process()
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

void YOLO_ONNXRuntime_Detect::process()
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

void YOLO_ONNXRuntime_Detect::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0.data() + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv3 || m_algo_type == YOLOv4 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		else if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		else if (m_algo_type == YOLO26)
		{
			score = ptr[4];
			class_id = int(ptr[5]);
		}

		if (score < m_score_threshold)
			continue;

		int left, top, width, height;
		if(m_algo_type == YOLOv3 || m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
		{
			left = (ptr[0] - 0.5 * ptr[2]) > 0 ? int(ptr[0] - 0.5 * ptr[2]) : 0;
			top = (ptr[1] - 0.5 * ptr[3]) > 0 ? int(ptr[1] - 0.5 * ptr[3]) : 0;
			width = ptr[2] > 0 ? int(ptr[2]) : 0;
			height = ptr[3] > 0 ? int(ptr[3]) : 0;
		}
		else if (m_algo_type == YOLOv4)
		{
			left = ptr[0] > 0 ? int(ptr[0] * m_input_size.width) : 0;
			top = ptr[1] > 0 ? int(ptr[1] * m_input_size.height) : 0;
			width = (ptr[2] - ptr[0]) > 0 ? int((ptr[2] - ptr[0]) * m_input_size.width) : 0;
			height = (ptr[3] - ptr[1])> 0 ? int((ptr[3] - ptr[1]) * m_input_size.height) : 0;
		}
		else if (m_algo_type == YOLOv10 || m_algo_type == YOLO26)
		{
			left = ptr[0] > 0 ? int(ptr[0]) : 0;
			top = ptr[1] > 0 ? int(ptr[1]) : 0;
			width = (ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			height = (ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
		}
		width = (left + width) < m_image.cols ? width : (m_image.cols - left);
		height = (top + height) < m_image.rows ? height : (m_image.rows - top);
		cv::Rect box = cv::Rect(left, top, width, height);

		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	scale_boxes(boxes, m_image.size());

	if(m_algo_type == YOLO26)
	{
		m_output_det.clear();
		m_output_det.resize(boxes.size());
		for (int i = 0; i < boxes.size(); i++)
		{
			OutputDet output;
			output.id = class_ids[i];
			output.score = scores[i];
			output.box = boxes[i];
			m_output_det[i] = output;
		}
	}
	else
	{
		std::vector<int> indices;
		nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);
		m_output_det.clear();
		m_output_det.resize(indices.size());
		for (int i = 0; i < indices.size(); i++)
		{
			int idx = indices[i];
			OutputDet output;
			output.id = class_ids[idx];
			output.score = scores[idx];
			output.box = boxes[idx];
			m_output_det[i] = output;
		}
	}

	if(m_draw_result)
		draw_result();
}