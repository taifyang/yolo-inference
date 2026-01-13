/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-04 23:48:17
 * @Description: source file for YOLO tensorrt detection
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

void YOLO_TensorRT_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv3 && algo_type != YOLOv4 && algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11 && algo_type != YOLOv12  && algo_type != YOLOv13)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_Detect::init(algo_type, device_type, model_type, model_path);

	m_task_type = Detect;

	cudaMallocHost(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output0_host, sizeof(float) * m_output_numdet);

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

void YOLO_TensorRT_Detect::pre_process()
{
#ifdef _CUDA_PREPROCESS
	cudaMemcpyAsync(m_input_host, m_image.data, sizeof(uint8_t) * 3 * m_image.cols * m_image.rows, cudaMemcpyHostToDevice, m_stream);
	cuda_preprocess_img(m_input_host, m_image.cols, m_image.rows, m_input_device, m_input_size.width, m_input_size.height, m_d2s_host, m_s2d_host, m_stream);
	cudaMemcpyAsync(m_d2s_device, m_d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_s2d_device, m_s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
#else
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));
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

void YOLO_TensorRT_Detect::process()
{
	m_execution_context->executeV2((void**)m_bindings);

#ifndef _CUDA_POSTPROCESS
	cudaMemcpyAsync(m_output0_host, m_output_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
#endif // !_CUDA_POSTPROCESS
}

void YOLO_TensorRT_Detect::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

#ifdef _CUDA_POSTPROCESS
	cudaMemset(m_output_box_device, 0, sizeof(float) * (m_num_box_element * m_max_box + 1));	
	cuda_decode(m_output_device, m_output_numbox, m_class_num, m_confidence_threshold, m_score_threshold, m_d2s_device,
		 m_output_box_device, m_max_box, m_num_box_element, m_input_size, m_stream, m_algo_type, m_task_type);
	cuda_nms(m_output_box_device, m_nms_threshold, m_max_box, m_num_box_element, m_stream);
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

#else
	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;
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

		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if(m_algo_type == YOLOv3 || m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLOv13)
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
		else if (m_algo_type == YOLOv4)
		{
			float x1 = ptr[0] * m_input_size.width;
			float y1 = ptr[1] * m_input_size.height;
			float x2 = ptr[2] * m_input_size.width;
			float y2 = ptr[3] * m_input_size.height;
			int left = int(x1) > 0 ? int(x1) : 0;
			int top = int(y1) > 0 ? int(y1) : 0;
			int width = int(x2 - x1) > 0 ? int(x2 - x1) : 0;
			int height = int(y2 - y1)> 0 ? int(y2 - y1) : 0;
			box = cv::Rect(left, top, width, height);
		}
		else if (m_algo_type == YOLOv10)
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
	}

	scale_boxes(boxes, m_image.size());

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

#endif // _CUDA_POSTPROCESS

	if(m_draw_result)
		draw_result(m_output_det);
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
