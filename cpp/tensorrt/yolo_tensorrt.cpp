/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-09-09 10:01:30
 * @FilePath: \cpp\tensorrt\yolo_tensorrt.cpp
 * @Description: tensorrt inference source file for YOLO algorithm
 */

#include "yolo_tensorrt.h"
#include "preprocess.cuh"
#include "decode.cuh"

class TRTLogger : public nvinfer1::ILogger
{
public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
	{
	}
} logger;

void YOLO_TensorRT::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if (device_type != GPU)
	{
		std::cerr << "TensorRT only support GPU!" << std::endl;
		std::exit(-1);
	}

	m_model_type = model_type;
	
	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}
	std::ifstream file(model_path, std::ios::binary);
	if (!file.good())
	{
		std::cerr << "read model error!" << std::endl;
		std::exit(-1);
	}

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string stream_model(buffer.str());

	TRTLogger logger;
	m_runtime = nvinfer1::createInferRuntime(logger);
	m_engine = m_runtime->deserializeCudaEngine(stream_model.data(), stream_model.size());

	cudaStreamCreate(&m_stream);
	m_execution_context = m_engine->createExecutionContext();
}

void YOLO_TensorRT_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_input_width = 224;
		m_input_height = 224;
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}

	m_task_type = Classify;

	cudaMallocHost(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output_host, sizeof(float) * m_class_num);

	cudaMalloc(&m_input_device, m_max_input_size);
	cudaMalloc(&m_output_device, sizeof(float) * m_class_num);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output_device;
}

void YOLO_TensorRT_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11 && algo_type != YOLOv12  && algo_type != YOLOv13)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		m_output_numprob = 5 + m_class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
	}
	if (m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
	{
		m_output_numprob = 4 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}

	m_output_numdet = 1 * m_output_numprob * m_output_numbox;

	m_task_type = Detect;

	cudaMallocHost(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output_host, sizeof(float) * m_output_numdet);

	cudaMalloc(&m_input_device, m_max_input_size);
	cudaMalloc(&m_output_device, sizeof(float) * m_output_numdet);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output_device;

#ifdef _CUDA_PREPROCESS
	cudaMallocHost(&m_d2s_host, sizeof(float) * 6);
	cudaMalloc(&m_d2s_device, sizeof(float) * 6);
	cudaMallocHost(&m_s2d_host, sizeof(float) * 6);
	cudaMalloc(&m_s2d_device, sizeof(float) * 6);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaMallocHost(&m_output_box_host, sizeof(float) * (m_num_box_element * m_max_box + 1));
	cudaMalloc(&m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1));
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv5)
	{
		m_output_numprob = 37 + m_class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_output_numprob = 36 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;

	m_task_type = Segment;

	cudaMallocHost(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output0_host, sizeof(float) * m_output_numdet);
	cudaMallocHost(&m_output1_host, sizeof(float) * m_output_numseg);

	cudaMalloc(&m_input_device, m_max_input_size);
	cudaMalloc(&m_output0_device, sizeof(float) * m_output_numdet);
	cudaMalloc(&m_output1_device, sizeof(float) * m_output_numseg);

#if NV_TENSORRT_MAJOR < 10
	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output1_device;
	m_bindings[2] = m_output0_device;
#else
	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output0_device;
	m_bindings[2] = m_output1_device;
#endif 

#ifdef _CUDA_PREPROCESS
	cudaMallocHost(&m_d2s_host, sizeof(float) * 6);
	cudaMalloc(&m_d2s_device, sizeof(float) * 6);
	cudaMallocHost(&m_s2d_host, sizeof(float) * 6);
	cudaMalloc(&m_s2d_device, sizeof(float) * 6);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaMallocHost(&m_output_box_host, sizeof(float) * (m_num_box_element * m_max_box + 1));
	cudaMalloc(&m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1));
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo_type == YOLOv5)
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
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_height * m_image.cols / m_image.rows, m_input_height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_width, m_input_width * m_image.rows / m_image.cols));

		//CenterCrop
		int crop_size = std::min(crop_image.cols, crop_image.rows);
		int  left = (crop_image.cols - crop_size) / 2, top = (crop_image.rows - crop_size) / 2;
		crop_image = crop_image(cv::Rect(left, top, crop_size, crop_size));
		cv::resize(crop_image, crop_image, cv::Size(m_input_width, m_input_height));

		//Normalize
		crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);
	}

	int image_area = crop_image.cols * crop_image.rows;
	float* pimage = (float*)crop_image.data;
	float* phost_r = m_input_host + image_area * 0;
	float* phost_g = m_input_host + image_area * 1;
	float* phost_b = m_input_host + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3)
	{
		*phost_r++ = pimage[2];
		*phost_g++ = pimage[1];
		*phost_b++ = pimage[0];
	}

	cudaMemcpyAsync(m_input_device, m_input_host, sizeof(float) * m_input_numel, cudaMemcpyHostToDevice, m_stream);
}

void YOLO_TensorRT_Detect::pre_process()
{
#ifdef _CUDA_PREPROCESS
	cudaMemcpyAsync(m_input_host, m_image.data, sizeof(uint8_t) * 3 * m_image.cols * m_image.rows, cudaMemcpyHostToDevice, m_stream);
	preprocess_kernel_img(m_input_host, m_image.cols, m_image.rows, m_input_device, m_input_width, m_input_height, m_d2s_host, m_s2d_host, m_stream);
	cudaMemcpyAsync(m_d2s_device, m_d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_s2d_device, m_s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
#else
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	int image_area = letterbox.cols * letterbox.rows;
	uchar* pimage = letterbox.data;
	float* phost_r = m_input_host + image_area * 0;
	float* phost_g = m_input_host + image_area * 1;
	float* phost_b = m_input_host + image_area * 2;
	for (int i = 0; i < letterbox.cols * letterbox.rows; ++i, pimage += 3)
	{
		*phost_r++ = pimage[2] / 255.0f;
		*phost_g++ = pimage[1] / 255.0f;
		*phost_b++ = pimage[0] / 255.0f;
	}

	cudaMemcpyAsync(m_input_device, m_input_host, sizeof(float) * m_input_numel, cudaMemcpyHostToDevice, m_stream);
#endif // _CUDA_PREPROCESS
}

void YOLO_TensorRT_Segment::pre_process()
{
#ifdef _CUDA_PREPROCESS
	cudaMemcpyAsync(m_input_host, m_image.data, sizeof(uint8_t) * 3 * m_image.cols * m_image.rows, cudaMemcpyHostToDevice, m_stream);
	preprocess_kernel_img(m_input_host, m_image.cols, m_image.rows, m_input_device, m_input_width, m_input_height, m_d2s_host, m_s2d_host, m_stream);
	cudaMemcpyAsync(m_d2s_device, m_d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_s2d_device, m_s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
#else
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));
	int image_area = letterbox.cols * letterbox.rows;
	uchar* pimage = letterbox.data;
	float* phost_r = m_input_host + image_area * 0;
	float* phost_g = m_input_host + image_area * 1;
	float* phost_b = m_input_host + image_area * 2;
	for (int i = 0; i < letterbox.cols * letterbox.rows; ++i, pimage += 3)
	{
		*phost_r++ = pimage[2] / 255.0f;
		*phost_g++ = pimage[1] / 255.0f;
		*phost_b++ = pimage[0] / 255.0f;
	}

	cudaMemcpyAsync(m_input_device, m_input_host, sizeof(float) * m_input_numel, cudaMemcpyHostToDevice, m_stream);
#endif // _CUDA_PREPROCESS
}

void YOLO_TensorRT_Classify::process()
{
#if NV_TENSORRT_MAJOR < 10
	m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
#else
	m_execution_context->executeV2((void**)m_bindings);
#endif 

	cudaMemcpyAsync(m_output_host, m_output_device, sizeof(float) * m_class_num, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
}

void YOLO_TensorRT_Detect::process()
{
#if NV_TENSORRT_MAJOR < 10
	m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
#else
	m_execution_context->executeV2((void**)m_bindings);
#endif 

#ifndef _CUDA_POSTPROCESS
	cudaMemcpyAsync(m_output_host, m_output_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
#endif // !_CUDA_POSTPROCESS
}

void YOLO_TensorRT_Segment::process()
{
#if NV_TENSORRT_MAJOR < 10
	m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
#else
	m_execution_context->executeV2((void**)m_bindings);
#endif 

	cudaMemcpyAsync(m_output0_host, m_output0_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
	cudaMemcpyAsync(m_output1_host, m_output1_device, sizeof(float) * m_output_numseg, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
}

void YOLO_TensorRT_Classify::post_process()
{
	std::vector<float> scores(m_class_num); 
	float sum = 0.0f;
	for (size_t i = 0; i < m_class_num; i++)
	{
		scores[i] = m_output_host[i];
		sum += exp(m_output_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo_type == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
}

void YOLO_TensorRT_Detect::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

#ifdef _CUDA_POSTPROCESS
	cudaMemset(m_output_box_device, 0, sizeof(float) * (m_num_box_element * m_max_box + 1));	
	decode_kernel_invoker(m_output_device, m_output_numbox, m_class_num, m_confidence_threshold, m_d2s_device, m_output_box_device, m_max_box, m_num_box_element, m_stream, m_algo_type, m_task_type);
	nms_kernel_invoker(m_output_box_device, m_nms_threshold, m_max_box, m_num_box_element, m_stream);
	cudaMemcpyAsync(m_output_box_host, m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	for (size_t i = 0; i < m_max_box; i++)
	{
		if (m_output_box_host[7 * i + 7])
		{
			float x1 = m_output_box_host[7 * i + 1];
			float y1 = m_output_box_host[7 * i + 2];
			float x2 = m_output_box_host[7 * i + 3];
			float y2 = m_output_box_host[7 * i + 4];

			int left = int(x1) > 0 ? int(x1) : 0;
			int top = int(y1) > 0 ? int(y1) : 0;
			int width = int(x2 - x1) > 0 ? int(x2 - x1) : 0;
			int height = int(y2 - y1)> 0 ? int(y2 - y1) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);

			boxes.push_back(cv::Rect(left, top, width, height));
			scores.push_back(m_output_box_host[7 * i + 5]);
			class_ids.push_back(m_output_box_host[7 * i + 6]);
		}
	}

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

	if(m_draw_result)
		draw_result(m_output_det);

#else
	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo_type == YOLOv6 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if(m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
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
		if (m_algo_type == YOLOv10)
		{
			int left = int(ptr[0]) > 0 ? int(ptr[0]) : 0;
			int top = int(ptr[1]) > 0 ? int(ptr[1]) : 0;
			int width = int(ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			int height = int(ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
		}

		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

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
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<cv::Mat> masks;
	std::vector<std::vector<float>> picked_proposals;

#ifdef _CUDA_POSTPROCESS
	cudaMemset(m_output_box_device, 0, sizeof(float) * (m_num_box_element * m_max_box + 1));	
	decode_kernel_invoker(m_output0_device, m_output_numbox, m_class_num, m_confidence_threshold, m_d2s_device, m_output_box_device, m_max_box, m_num_box_element, m_stream, m_algo_type, m_task_type);
	nms_kernel_invoker(m_output_box_device, m_nms_threshold, m_max_box, m_num_box_element, m_stream);
	cudaMemcpyAsync(m_output_box_host, m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	for (size_t i = 0; i < m_max_box; i++)
	{
		if (m_output_box_host[8 * i + 7])
		{
			float x1 = m_output_box_host[8 * i + 1];
			float y1 = m_output_box_host[8 * i + 2];
			float x2 = m_output_box_host[8 * i + 3];
			float y2 = m_output_box_host[8 * i + 4];

			int left = int(x1) > 0 ? int(x1) : 0;
			int top = int(y1) > 0 ? int(y1) : 0;
			int width = int(x2 - x1) > 0 ? int(x2 - x1) : 0;
			int height = int(y2 - y1)> 0 ? int(y2 - y1) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);

			boxes.push_back(cv::Rect(left, top, width, height));
			scores.push_back(m_output_box_host[8 * i + 5]);
			class_ids.push_back(m_output_box_host[8 * i + 6]);

			int row_index = m_output_box_host[8 * i + 8];
			float* mask_weights;
			if (m_algo_type == YOLOv5)
			{
				mask_weights = m_output0_device + row_index * m_output_numprob + m_class_num + 5;
			}
			if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
			{
				mask_weights = m_output0_device + row_index * m_output_numprob + m_class_num + 4;
			}

			float l, t, r, b;
			affine_project(m_s2d_host, x1, y1, &l, &t);
			affine_project(m_s2d_host, x2, y2, &r, &b);
			float x_ratio = m_mask_params.seg_width / (float)m_mask_params.net_width;
			float y_ratio = m_mask_params.seg_height / (float)m_mask_params.net_height;
			int mask_out_width = ceil((r - l) * x_ratio + 0.5f);
			int mask_out_height = ceil((b - t) * y_ratio + 0.5f);

			if (mask_out_width > 0 && mask_out_height > 0)
			{ 
				unsigned char* mask_out_device;
				cudaMalloc(&mask_out_device, sizeof(unsigned char) * (mask_out_width*mask_out_height));
				cudaMemset(mask_out_device, 0, sizeof(unsigned char) * (mask_out_width*mask_out_height));	
				decode_single_mask(l * x_ratio, t * y_ratio, mask_weights, m_output1_device, m_mask_params.seg_height, m_mask_params.seg_width, mask_out_device, 32, mask_out_width, mask_out_height, m_stream);
				unsigned char* mask_out_host;
				cudaMallocHost(&mask_out_host, sizeof(unsigned char) * (mask_out_width*mask_out_height));
				cudaMemcpyAsync(mask_out_host, mask_out_device, sizeof(unsigned char) * (mask_out_width*mask_out_height), cudaMemcpyDeviceToHost, m_stream);
				cudaStreamSynchronize(m_stream);
				cv::Mat temp(mask_out_height, mask_out_width, CV_8UC1, mask_out_host), mask;
				cv::resize(temp, mask, cv::Size(width, height), cv::INTER_LINEAR);
				cv::Rect temp_rect = cv::Rect(left, top, width, height);
				mask = mask(temp_rect - cv::Point(left, top)) > m_mask_params.mask_threshold*255;
				masks.push_back(mask);
			}
		}
	}

	m_output_seg.clear();
	m_output_seg.resize(boxes.size());
	for (int i = 0; i < boxes.size(); i++)
	{
		OutputSeg output;
		output.id = class_ids[i];
		output.score = scores[i];
		output.box = boxes[i];
		output.mask = masks[i];
		m_output_seg[i] = output;
	}

	if(m_draw_result)
		draw_result(m_output_seg);

#else
	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;
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
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}

		if (score < m_score_threshold)
			continue;

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
		cv::Rect box = cv::Rect(left, top, width, height);

		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);

		if (m_algo_type == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 5, ptr + m_class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 4, ptr + m_class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
	}

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	m_output_seg.clear();
	m_output_seg.resize(indices.size());
	std::vector<std::vector<float>> temp_mask_proposals;
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

#ifdef _CUDA_PREPROCESS
	m_mask_params.params = cv::Vec4d(1 / m_d2s_host[0], 1 / m_d2s_host[4], -m_d2s_host[2] / m_d2s_host[0]);
#else
	m_mask_params.params = m_params;
#endif // _CUDA_PREPROCESS

	m_mask_params.input_shape = m_image.size();
	int shape[4] = { 1, m_mask_params.seg_channels, m_mask_params.seg_width, m_mask_params.seg_height};
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1_host, m_output1_host + m_output_numseg, (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, m_output_seg[i], m_mask_params, m_algo_type);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT::release()
{
	cudaStreamDestroy(m_stream);
	cudaFree(m_input_device);

#if NV_TENSORRT_MAJOR < 10
	m_execution_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
#endif 
}

void YOLO_TensorRT_Classify::release()
{
	YOLO_TensorRT::release();
	
	cudaFree(m_output_device);
}

void YOLO_TensorRT_Detect::release()
{
	YOLO_TensorRT::release();

	cudaFree(m_output_device);

#ifdef _CUDA_PREPROCESS
	cudaFree(m_d2s_device);
	cudaFreeHost(m_d2s_host);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaFree(m_output_box_device);
	cudaFreeHost(m_output_box_host);
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT_Segment::release()
{
	YOLO_TensorRT::release();

	cudaFree(m_output0_device);
	cudaFree(m_output1_device);

#ifdef _CUDA_PREPROCESS
	cudaFree(m_d2s_device);
	cudaFreeHost(m_d2s_host);
#endif // _CUDA_PREPROCESS
}