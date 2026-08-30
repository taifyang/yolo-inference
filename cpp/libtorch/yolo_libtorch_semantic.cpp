/* 
 * @Author: taifyang
 * @Date: 2025-12-21 21:51:23
 * @LastEditTime: 2026-08-30 11:33:41
 * @Description: source file for YOLO libtorch semantic segmentation
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch_Semantic::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
	YOLO_Semantic::init(algo_type, device_type, model_type, model_path);
}

void YOLO_Libtorch_Semantic::pre_process()
{
	YOLO_Libtorch_Detect::pre_process();
}

void YOLO_Libtorch_Semantic::process()
{	
	m_output = m_module.forward(m_input);	
}

void scale_masks(const torch::Tensor& input_mask, torch::Tensor& output_mask, const cv::Size input_shape,  const cv::Size output_shape)
{
    int64_t H_in  = input_shape.height;
    int64_t W_in  = input_shape.width;
    int64_t H_out = output_shape.height;
    int64_t W_out = output_shape.width;

    double gain = std::min(static_cast<double>(H_in) / H_out, static_cast<double>(W_in) / W_out);

    double pad_w = (W_in - W_out * gain) / 2.0;
    double pad_h = (H_in - H_out * gain) / 2.0;

    int64_t left  = static_cast<int64_t>(std::round(pad_w - 0.1));
    int64_t right = static_cast<int64_t>(W_in - std::round(pad_w + 0.1));

    int64_t top    = static_cast<int64_t>(std::round(pad_h - 0.1));
    int64_t bottom = static_cast<int64_t>(H_in - std::round(pad_h + 0.1));

	auto cropped = input_mask.index({at::indexing::Ellipsis, at::indexing::Slice(top, bottom), at::indexing::Slice(left, right)});

    output_mask = torch::nn::functional::interpolate(cropped,
        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{H_out, W_out}).mode(torch::kBilinear).align_corners(false));
}

void YOLO_Libtorch_Semantic::post_process()
{
	torch::Tensor pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU), class_map;
	scale_masks(pred, class_map, m_input_size, m_image.size());
	torch::Tensor mask_tensor = class_map[0].argmax(0).to(torch::kCPU).to(torch::kUInt8);
	m_mask = cv::Mat((int)mask_tensor.size(0), (int)mask_tensor.size(1), CV_8UC1, mask_tensor.data_ptr<uint8_t>()).clone();

	if (m_draw_result)
	{
		draw_result();
	}
}