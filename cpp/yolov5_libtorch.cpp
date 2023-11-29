#include "yolov5_libtorch.h"
#include "utils.h"


YOLOv5_Libtorch::YOLOv5_Libtorch(std::string model_path, Device_Type device_type)
{
	module = torch::jit::load(model_path);
	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	module.to(m_device);
}


void YOLOv5_Libtorch::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
	torch::Tensor input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }).to(m_device);
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_inputs.emplace_back(input);
}


void YOLOv5_Libtorch::process()
{
	m_outputs = module.forward(m_inputs);
}


void YOLOv5_Libtorch::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	torch::Tensor preds = m_outputs.toTuple()->elements()[0].toTensor().to(at::kCPU);
	float* data = new float[preds[0].numel()];
	std::copy(preds[0].data_ptr<float>(), preds[0].data_ptr<float>() + preds[0].numel(), data);

	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = data + i * output_numprob;
		float objness = ptr[4];
		if (objness < confidence_threshold)
			continue;

		float* classes_scores = ptr + 5;
		int class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
		float max_class_score = classes_scores[class_id];
		float score = max_class_score * objness;
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
		scale_boxes(box, m_image.size());
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

