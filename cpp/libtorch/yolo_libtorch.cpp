/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 22:19:04
 * @FilePath: \cpp\libtorch\yolo_libtorch.cpp
 * @Description: yolo算法的libtorch推理框架实现
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	m_algo = algo_type;

	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	module = torch::jit::load(model_path);
	module.to(m_device);

	if (model_type != FP32 && model_type != FP16)
	{
		std::cout << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	if (model_type == FP16 && device_type != GPU)
	{
		std::cout << "FP16 only support GPU!" << std::endl;
		std::exit(-1);

	}
	m_model = model_type;
	if (model_type == FP16)
	{
		module.to(torch::kHalf);
	}
}

void YOLO_Libtorch_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv8)
	{
		m_input_width = 224;
		m_input_height = 224;
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}

	m_output_host = new float[class_num];
}

void YOLO_Libtorch_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv5)
	{
		m_output_numprob = 5 + class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	}
	if (m_algo == YOLOv8)
	{
		m_output_numprob = 4 + class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	}

	m_output_host = new float[m_output_numdet];
}

void YOLO_Libtorch_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo == YOLOv5)
	{
		m_output_numprob = 37 + class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.segChannels * m_mask_params.segWidth * m_mask_params.segHeight;
	}
	if (m_algo == YOLOv8)
	{
		m_output_numprob = 36 + class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.segChannels * m_mask_params.segWidth * m_mask_params.segHeight;
	}

	m_output0_host = new float[m_output_numdet];
	m_output1_host = new float[m_output_numseg];
}

void YOLO_Libtorch_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo == YOLOv5)
	{
		//CenterCrop
		int crop_size = std::min(m_image.cols, m_image.rows);
		int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
		crop_image = m_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_height));

		//Normalize
		crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);
		cv::subtract(crop_image, cv::Scalar(0.406, 0.456, 0.485), crop_image);
		cv::divide(crop_image, cv::Scalar(0.225, 0.224, 0.229), crop_image);

		cv::cvtColor(crop_image, crop_image, cv::COLOR_BGR2RGB);
	}

	if (m_algo == YOLOv8)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, m_image, cv::Size(m_input_height * m_image.cols / m_image.rows, m_input_height)); 
		else
			cv::resize(m_image, m_image, cv::Size(m_input_width, m_input_width * m_image.rows / m_image.cols));

		//CenterCrop
		int crop_size = std::min(m_image.cols, m_image.rows);
		int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
		crop_image = m_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_height));

		//Normalize
		crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);

		cv::cvtColor(m_image, m_image, cv::COLOR_BGR2RGB);
	}

	torch::Tensor input;
	if (m_model == FP32)
	{
		input = torch::from_blob(crop_image.data, { 1, crop_image.rows, crop_image.cols, crop_image.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model == FP16)
	{
		crop_image.convertTo(crop_image, CV_16FC3);
		input = torch::from_blob(crop_image.data, { 1, crop_image.rows, crop_image.cols, crop_image.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Detect::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model == FP16)
	{
		letterbox.convertTo(letterbox, CV_16FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Segment::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model == FP16)
	{
		letterbox.convertTo(letterbox, CV_16FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Classify::process()
{
	m_output = module.forward(m_input);

	torch::Tensor pred;
	if (m_algo == YOLOv5)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}
	if (m_algo == YOLOv8)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}

	std::copy(pred.data_ptr<float>(), pred.data_ptr<float>() + class_num, m_output_host);
}

void YOLO_Libtorch_Detect::process()
{
	m_output = module.forward(m_input);

	torch::Tensor pred;
	if (m_algo == YOLOv5)
	{
		pred = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
	}
	if (m_algo == YOLOv8)
	{
		pred = m_output.toTensor().to(at::kCPU);
	}

	std::copy(pred.data_ptr<float>(), pred.data_ptr<float>() + m_output_numdet, m_output_host);
}

void YOLO_Libtorch_Segment::process()
{
	m_output = module.forward(m_input);
	torch::Tensor pred0, pred1;
	if (m_algo == YOLOv5)
	{
		pred0 = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
		pred1 = m_output.toTuple()->elements()[1].toTensor().to(torch::kFloat).to(at::kCPU);
	}
	if (m_algo == YOLOv8)
	{
		pred0 = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
		pred1 = m_output.toTuple()->elements()[1].toTensor().to(torch::kFloat).to(at::kCPU);
	}

	std::copy(pred0.data_ptr<float>(), pred0.data_ptr<float>() + m_output_numdet, m_output0_host);
	std::copy(pred1.data_ptr<float>(), pred1.data_ptr<float>() + m_output_numseg, m_output1_host);
}

void YOLO_Libtorch_Classify::post_process()
{
	std::vector<float> scores;
	float sum = 0.0f;
	for (size_t i = 0; i < class_num; i++)
	{
		scores.push_back(m_output_host[i]);
		sum += exp(m_output_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	std::string label;
	if (m_algo == YOLOv5)
		label = "class" + std::to_string(id) + ":" + cv::format("%.2f", exp(scores[id]) / sum);
	if (m_algo == YOLOv8)
		label = "class" + std::to_string(id) + ":" + cv::format("%.2f", scores[id]);

	draw_result(label);
}

void YOLO_Libtorch_Detect::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)	
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
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
		std::string label = "class" + std::to_string(class_ids[idx]) + ":" + cv::format("%.2f", scores[idx]);
		draw_result(label, box);
	}
}

void YOLO_Libtorch_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
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
		
		if (m_algo == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + class_num + 5, ptr + class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		if (m_algo == YOLOv8)
		{
			std::vector<float> temp_proto(ptr + class_num + 4, ptr + class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);

	std::vector<OutputSeg> output;
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
	for (int i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		OutputSeg result;
		result.id = class_ids[idx];
		result.confidence = scores[idx];
		result.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		output.push_back(result);
	}

	m_mask_params.params = m_params;
	m_mask_params.srcImgShape = m_image.size();
	int shape[4] = { 1, m_mask_params.segChannels, m_mask_params.segWidth, m_mask_params.segHeight, };
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1_host, m_output1_host + m_output_numseg, (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, output[i], m_mask_params);
	}

	draw_result(output);
}