/* 
 * @Author: taifyang
 * @Date: 2025-12-21 22:22:59
 * @LastEditTime: 2026-01-17 19:54:22
 * @Description: source file for YOLO onnxruntime segmentation
 */

#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	YOLO_Segment::init(algo_type, device_type, model_type, model_path);

	m_output_names.push_back("output1");

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0.resize(m_output_numdet);
		m_output1.resize(m_output_numseg);
	}
}

void YOLO_ONNXRuntime_Segment::pre_process()
{
	YOLO_ONNXRuntime_Detect::pre_process();
}

void YOLO_ONNXRuntime_Segment::process()
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
		m_output1_host = const_cast<float*>(outputs[1].GetTensorData<float>());
		m_output0.assign(m_output0_host, m_output0_host + m_output_numdet);
		m_output1.assign(m_output1_host, m_output1_host + m_output_numseg);
	}
	else if (m_model_type == FP16)
	{
		uint16_t* output0_fp16 = const_cast<uint16_t*>(outputs[0].GetTensorData<uint16_t>());
		uint16_t* output1_fp16 = const_cast<uint16_t*>(outputs[1].GetTensorData<uint16_t>());
		m_output0_fp16.assign(output0_fp16, output0_fp16 + m_output_numdet);
		m_output1_fp16.assign(output1_fp16, output1_fp16 + m_output_numseg);
		for (size_t i = 0; i < m_output_numdet; i++)
		{
			m_output0[i] = float16_to_float32(m_output0_fp16[i]);
		}
		for (size_t i = 0; i < m_output_numseg; i++)
		{
			m_output1[i] = float16_to_float32(m_output1_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0.data() + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		else if (m_algo_type == YOLO26)
		{
			score = ptr[4];
			class_id = int(ptr[5]);
		}

		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if (m_algo_type == YOLOv5 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float x = ptr[0];
			float y = ptr[1];
			float w = ptr[2];
			float h = ptr[3];
			int left = int(x - 0.5 * w) > 0 ? int(x - 0.5 * w) : 0;
			int top = int(y - 0.5 * h) > 0 ? int(y - 0.5 * h) : 0;
			int width = int(w) > 0 ? int(w) : 0;
			int height = int(h)> 0 ? int(h) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
		}
		else if (m_algo_type == YOLO26)
		{
			int left = int(ptr[0]) > 0 ? int(ptr[0]) : 0;
			int top = int(ptr[1]) > 0 ? int(ptr[1]) : 0;
			int width = int(ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			int height = int(ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
		}

		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);

		if (m_algo_type == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 5, ptr + m_class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 4, ptr + m_class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
		else if(m_algo_type == YOLO26)
		{
			std::vector<float> temp_proto(ptr + 6, ptr + 38);
			picked_proposals.push_back(temp_proto);
		}
	}

	scale_boxes(boxes, m_image.size());

	std::vector<std::vector<float>> temp_mask_proposals;
	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		std::vector<int> indices;
		nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);
		m_output_seg.clear();
		m_output_seg.resize(indices.size());
		cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
		for (int i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			OutputSeg output;
			output.id = class_ids[idx];
			output.score = scores[idx];
			output.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			m_output_seg[i] = output;
		}
	}
	else if(m_algo_type == YOLO26)
	{
		m_output_seg.clear();
		m_output_seg.resize(boxes.size());
		cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
		for (int i = 0; i < boxes.size(); ++i)
		{
			OutputSeg output;
			output.id = class_ids[i];
			output.score = scores[i];
			output.box = boxes[i] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[i]);
			m_output_seg[i] = output;
		}
	} 

	m_mask_params.params = m_params;
	m_mask_params.input_shape = m_image.size();
	int shape[4] = { 1, m_mask_params.seg_channels, m_mask_params.seg_width, m_mask_params.seg_height, };
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1.begin(), m_output1.end(), (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, m_output_seg[i], m_mask_params, m_algo_type);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}
