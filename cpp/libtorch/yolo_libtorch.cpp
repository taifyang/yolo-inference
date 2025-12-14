/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-12-14 10:56:37
 * @FilePath: \cpp\libtorch\yolo_libtorch.cpp
 * @Description: libtorch inference source file for YOLO algorithm
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}
 
	try
	{
		m_module = torch::jit::load(model_path);
	}
	catch (const c10::Error& e) 
	{
		std::cerr << "libtorch load model failed!" << std::endl;
		std::exit(-1);
	}

	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	m_module.to(m_device);

	if (model_type != FP32 && model_type != FP16)
	{
		std::cerr << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	if (model_type == FP16 && device_type != GPU)
	{
		std::cerr << "FP16 only support GPU!" << std::endl;
		std::exit(-1);
	}
	m_model_type = model_type;
	if (model_type == FP16)
	{
		m_module.to(torch::kHalf);
	}
}

void YOLO_Libtorch_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_input_size.width = 224;
		m_input_size.height = 224;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}
}

void YOLO_Libtorch_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv3 && algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11  && algo_type != YOLOv12  && algo_type != YOLOv13)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		m_output_numprob = 5 + m_class_num;
		m_output_numbox = 3 * (m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32);
	}
	else if (m_algo_type == YOLOv3 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10|| m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		m_output_numprob = 4 + m_class_num;
		m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
	}

	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
}

void YOLO_Libtorch_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv5)
	{
		m_output_numprob = 37 + m_class_num;
		m_output_numbox = 3 * (m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_output_numprob = 36 + m_class_num;
		m_output_numbox = m_input_size.width / 8 * m_input_size.height / 8 + m_input_size.width / 16 * m_input_size.height / 16 + m_input_size.width / 32 * m_input_size.height / 32;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;
}

void YOLO_Libtorch_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo_type == YOLOv5)
	{
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
		cv::cvtColor(crop_image, crop_image, cv::COLOR_BGR2RGB);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));

		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
		cv::cvtColor(crop_image, crop_image, cv::COLOR_BGR2RGB);
	}

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		input = torch::from_blob(crop_image.data, { 1, crop_image.rows, crop_image.cols, crop_image.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
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
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
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
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
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
	m_output = m_module.forward(m_input).toTensor();
}

void YOLO_Libtorch_Detect::process()
{
	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
		m_output = m_module.forward(m_input).toTuple()->elements()[0].toTensor();
	else if (m_algo_type == YOLOv3 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11|| m_algo_type == YOLOv12|| m_algo_type == YOLOv13)
		m_output = m_module.forward(m_input).toTensor();
}

void YOLO_Libtorch_Segment::process()
{	
	m_output0 = m_module.forward(m_input).toTuple()->elements()[0].toTensor();
	m_output1 = m_module.forward(m_input).toTuple()->elements()[1].toTensor();
}

void YOLO_Libtorch_Classify::post_process()
{
	m_output_cls.id = torch::argmax(m_output).item<int64_t>();
	if (m_algo_type == YOLOv5)
		m_output_cls.score = torch::softmax(m_output.flatten(), 0)[m_output_cls.id].to(torch::kCPU).item<float>();
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		m_output_cls.score = torch::max(m_output).to(torch::kCPU).item<float>();

	if(m_draw_result)
		draw_result(m_output_cls);
}

torch::Tensor xywh2xyxy(const torch::Tensor& x)
{
    torch::Tensor y = x.clone();  
    y.index({torch::indexing::Ellipsis, 0}) = x.index({torch::indexing::Ellipsis, 0}) - x.index({torch::indexing::Ellipsis, 2}) / 2;  
    y.index({torch::indexing::Ellipsis, 1}) = x.index({torch::indexing::Ellipsis, 1}) - x.index({torch::indexing::Ellipsis, 3}) / 2;  
    y.index({torch::indexing::Ellipsis, 2}) = x.index({torch::indexing::Ellipsis, 0}) + x.index({torch::indexing::Ellipsis, 2}) / 2; 
    y.index({torch::indexing::Ellipsis, 3}) = x.index({torch::indexing::Ellipsis, 1}) + x.index({torch::indexing::Ellipsis, 3}) / 2;  
    return y;
}

std::vector<cv::Rect> tensor2rects(const torch::Tensor& tensor) 
{
    auto coords = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto ptr = coords.data_ptr<float>(); 
	std::vector<cv::Rect> rects(coords.size(0)); 
    for (int i = 0; i < coords.size(0); ++i) 
	{
        int x1 = static_cast<int>(ptr[i*4]), y1 = static_cast<int>(ptr[i*4+1]);
        int x2 = static_cast<int>(ptr[i*4+2]), y2 = static_cast<int>(ptr[i*4+3]);
        rects[i] = cv::Rect(x1, y1, x2-x1, y2-y1); 
    }
    return rects;
}

template <typename T>
std::vector<T> tensor2vec(const torch::Tensor& tensor) 
{
	torch::Dtype dtype;
	if constexpr (std::is_same_v<T, float>) 
        dtype = torch::kFloat32;
	else if constexpr (std::is_same_v<T, int>) 
        dtype = torch::kInt32;
    auto tensor_cpu = tensor.to(torch::kCPU).to(dtype);
    return {tensor_cpu.data_ptr<T>(), tensor_cpu.data_ptr<T>() + tensor_cpu.numel()};
}

void YOLO_Libtorch_Detect::post_process()
{
	torch::Tensor output = torch::squeeze(m_output);
	torch::Tensor cls_scores;
	if (m_algo_type == YOLOv3 || m_algo_type == YOLOv4 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		cls_scores = output.index({torch::indexing::Ellipsis, torch::indexing::Slice(4, 4 + m_class_num)});

	}	
	else if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		torch::Tensor obj_conf = output.index({torch::indexing::Ellipsis, 4});
        torch::Tensor conf_mask = obj_conf > m_confidence_threshold;
		output = output.index({conf_mask});
		cls_scores = output.index({torch::indexing::Ellipsis, torch::indexing::Slice(5, 5 + m_class_num)});
	}  

	if (m_algo_type == YOLOv3 || m_algo_type == YOLOv4 || m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		output.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}) = xywh2xyxy(output.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}));
	}	

	torch::Tensor xc = torch::amax(cls_scores, 1) > m_score_threshold;
	torch::Tensor output_filtered = output.index({xc});

	torch::Tensor box, obj, cls;
	if (m_algo_type == YOLOv3 || m_algo_type == YOLOv4 || m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		std::vector<torch::Tensor> split_result = torch::split(output_filtered, {4, m_class_num}, 1);
		box = split_result[0];
		cls = split_result[1]; 
	}
	else if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		std::vector<torch::Tensor> split_result = torch::split(output_filtered, {4, 1, m_class_num}, 1);
		box = split_result[0];
		obj = split_result[1];
		cls = split_result[2]; 
	}

  	auto [scores_tensor, j] = torch::max(cls, 1, true);

	std::vector<cv::Rect> boxes = tensor2rects(box);
	scale_boxes(boxes, m_image.size());

	std::vector<float> scores;
	if(obj.defined())
		scores = tensor2vec<float>(scores_tensor * obj);
	else
		scores = tensor2vec<float>(scores_tensor);
		
	std::vector<int> class_ids = tensor2vec<int>(j.to(torch::kInt32));

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

	if(m_draw_result)
		draw_result(m_output_det);
}

torch::Tensor crop_mask(torch::Tensor masks, torch::Tensor boxes) 
{
    int64_t n = masks.size(0);
    int64_t h = masks.size(1);
    int64_t w = masks.size(2);
    torch::Tensor boxes_xyxy = boxes.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}).unsqueeze(2);
    std::vector<torch::Tensor> xyxy_chunks = torch::chunk(boxes_xyxy, 4, 1);
    torch::Tensor x1 = xyxy_chunks[0];
    torch::Tensor y1 = xyxy_chunks[1];
    torch::Tensor x2 = xyxy_chunks[2];
    torch::Tensor y2 = xyxy_chunks[3];
    torch::Tensor r = torch::arange(w, torch::TensorOptions().device(masks.device()).dtype(x1.dtype())).unsqueeze(0).unsqueeze(0);
    torch::Tensor c = torch::arange(h, torch::TensorOptions().device(masks.device()).dtype(x1.dtype())).unsqueeze(0).unsqueeze(2); 
    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2));
}

cv::Mat tensor2mat(const torch::Tensor& tensor)
 {
	torch::Tensor tensor_float = tensor.to(torch::kCPU).squeeze().to(torch::kFloat32); 
    cv::Mat mat(tensor.size(0), tensor.size(1), CV_32FC1, tensor_float.data_ptr<float>());
    return mat;
}

cv::Mat scale_mask(const cv::Mat& mask, const cv::Size& input_shape, const cv::Size& output_shape)
{
	float gain = std::min(float(input_shape.width) / output_shape.width, float(input_shape.height) / output_shape.height); 
	float pad[2] = {(input_shape.width - output_shape.width * gain) / 2, (input_shape.height - output_shape.height * gain) / 2};
	cv::Mat mask_scaled = mask(cv::Rect(int(pad[0]), int(pad[1]), mask.cols - 2 * int(pad[0]), mask.rows - 2 * int(pad[1])));
	cv::Mat mask_resized;
	cv::resize(mask_scaled, mask_resized, cv::Size(output_shape.width, output_shape.height), cv::INTER_LINEAR);
	return mask_resized;
}

void YOLO_Libtorch_Segment::post_process()
{
	torch::Tensor output = torch::squeeze(m_output0);
	torch::Tensor cls_scores;
	if (m_algo_type == YOLOv5)
	{
		torch::Tensor obj_conf = output.index({torch::indexing::Ellipsis, 4});
        torch::Tensor conf_mask = obj_conf > m_confidence_threshold;
		output = output.index({conf_mask});
		cls_scores = output.index({torch::indexing::Ellipsis, torch::indexing::Slice(5, 5 + m_class_num)});
	}  
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		cls_scores = output.index({torch::indexing::Ellipsis, torch::indexing::Slice(4, 4 + m_class_num)});
	}	

	output.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}) = xywh2xyxy(output.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 4)}));
	torch::Tensor xc = torch::amax(cls_scores, 1) > m_score_threshold;
	torch::Tensor output_filtered = output.index({xc});

	torch::Tensor box, obj, cls, mask;
	if (m_algo_type == YOLOv5)
	{
		std::vector<torch::Tensor> split_result = torch::split(output_filtered, {4, 1, m_class_num, 32}, 1);
		box = split_result[0];
		obj = split_result[1];
		cls = split_result[2]; 
		mask = split_result[3]; 
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		std::vector<torch::Tensor> split_result = torch::split(output_filtered, {4, m_class_num, 32}, 1);
		box = split_result[0];
		cls = split_result[1]; 
		mask = split_result[2]; 
	}		
	
  	auto [scores_tensor, j] = torch::max(cls, 1, true);

	std::vector<cv::Rect> boxes = tensor2rects(box);
	scale_boxes(boxes, m_image.size());

	std::vector<float> scores;
	if(obj.defined())
		scores = tensor2vec<float>(scores_tensor * obj);
	else
		scores = tensor2vec<float>(scores_tensor);
		
	std::vector<int> class_ids = tensor2vec<int>(j.to(torch::kInt32));

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	box = box.index_select(0, torch::tensor(indices, torch::kLong).to(m_device)).to(torch::kFloat32).contiguous();
	torch::Tensor masks_in = mask.index_select(0, torch::tensor(indices, torch::kLong).to(m_device)).to(torch::kFloat32).contiguous();
	
	torch::Tensor proto = torch::squeeze(m_output1).to(torch::kFloat32).contiguous();
    int64_t c = proto.size(0), mh = proto.size(1), mw = proto.size(2);

	torch::Tensor masks;
	auto proto_reshaped = proto.reshape({c, -1});  
	if (m_algo_type == YOLOv5) 
		masks = torch::sigmoid(masks_in.matmul(proto_reshaped)).reshape({-1, mh, mw});
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12) 
		masks = masks_in.matmul(proto_reshaped).reshape({-1, mh, mw});

	torch::Tensor downsampled_bboxes = box.clone();
	float scale_w = static_cast<float>(mw) / m_input_size.width;   
	float scale_h = static_cast<float>(mh) / m_input_size.height;  
	downsampled_bboxes.index({torch::indexing::Ellipsis, 0}) *= scale_w;  
	downsampled_bboxes.index({torch::indexing::Ellipsis, 2}) *= scale_w; 
	downsampled_bboxes.index({torch::indexing::Ellipsis, 1}) *= scale_h;  
	downsampled_bboxes.index({torch::indexing::Ellipsis, 3}) *= scale_h;  

	masks = crop_mask(masks, downsampled_bboxes);

	std::vector<int64_t> inputs_shape = {m_input_size.height, m_input_size.width}; 
    masks = torch::nn::functional::interpolate(masks.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(inputs_shape).mode(torch::kBilinear).align_corners(false)).index({0}); 

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
		cv::Mat mask_mat = tensor2mat(masks[i]);
		cv::Mat resized_mask = scale_mask(mask_mat, m_input_size, m_image.size());
		if (m_algo_type == YOLOv5) 
			output.mask= (resized_mask > 0.5f);
		else if(m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
			output.mask =(resized_mask > 0.0f);
		m_output_seg[i] = output;
	}

	if(m_draw_result)
	{
		cv::Mat image_copy = m_image.clone();
		srand(time(0));
		for (int i = 0; i < m_output_seg.size(); ++i)
		{
			cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
			image_copy.setTo(random_color, m_output_seg[i].mask);
		}

		cv::addWeighted(m_image, 0.5, image_copy, 0.5, 0.0, m_result);

		for (int i = 0; i < m_output_seg.size(); ++i)
		{
			cv::Rect bbox = m_output_seg[i].box & cv::Rect(0, 0, m_image.cols, m_image.rows);
			cv::rectangle(m_result, bbox, cv::Scalar(255, 0, 0), 1);
			std::string label = "class" + std::to_string(m_output_seg[i].id) + ":" + cv::format("%.2f", m_output_seg[i].score);
			cv::putText(m_result, label, cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}
	}
}