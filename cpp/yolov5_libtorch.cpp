#include "yolov5_libtorch.h"
#include "utils.h"


YOLOv5_Libtorch::YOLOv5_Libtorch(std::string model_path, Device_Type device_type, Model_Type model_type)
{
	module = torch::jit::load(model_path);
	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	module.to(m_device);

	m_model = model_type;
	assert(("unsupported model type!", model_type == FP32 || model_type == FP16));
	if (model_type == FP16)
	{
		assert(("FP16 only support CPU!", device_type == GPU));
		module.to(torch::kHalf);
	}

	m_outputs_host = new float[output_numel];
}


void YOLOv5_Libtorch::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
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
	m_inputs.clear();
	m_inputs.emplace_back(input);
}


void YOLOv5_Libtorch::process()
{
	m_outputs = module.forward(m_inputs);

	torch::Tensor preds = m_outputs.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);

	std::copy(preds[0].data_ptr<float>(), preds[0].data_ptr<float>() + output_numel, m_outputs_host);
}

