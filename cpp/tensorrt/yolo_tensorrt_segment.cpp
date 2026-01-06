/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-05 00:18:42
 * @Description: source file for YOLO tensorrt segmentation
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

void YOLO_TensorRT_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_Segment::init(algo_type, device_type, model_type, model_path);

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

void YOLO_TensorRT_Segment::pre_process()
{	
	YOLO_TensorRT_Detect::pre_process();
}

void YOLO_TensorRT_Segment::process()
{
	m_execution_context->executeV2((void**)m_bindings);

#ifndef _CUDA_POSTPROCESS
	cudaMemcpyAsync(m_output0_host, m_output0_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
	cudaMemcpyAsync(m_output1_host, m_output1_device, sizeof(float) * m_output_numseg, cudaMemcpyDeviceToHost, m_stream);
#endif // !_CUDA_POSTPROCESS
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
	decode_kernel_invoker(m_output0_device, m_output_numbox, m_class_num, m_confidence_threshold, m_score_threshold, m_d2s_device, 
		m_output_box_device, m_max_box, m_num_box_element, m_input_size, m_stream, m_algo_type, m_task_type);
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
			cv::Rect box(left, top, width, height);

			boxes.push_back(box);
			scores.push_back(m_output_box_host[8 * i + 5]);
			class_ids.push_back(m_output_box_host[8 * i + 6]);

			int row_index = m_output_box_host[8 * i + 8];
			float* mask_weights;
			if (m_algo_type == YOLOv5)
			{
				mask_weights = m_output0_device + row_index * m_output_numprob + m_class_num + 5;
			}
			else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
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
				uint8_t* mask_device, *mask_resized_device, *mask_host;
				cudaMalloc(&mask_device, sizeof(uint8_t) * mask_out_width * mask_out_height);
				cudaMemset(mask_device, 0, sizeof(uint8_t) * mask_out_width * mask_out_height);	
				decode_single_mask(l * x_ratio, t * y_ratio, mask_weights, m_output1_device, m_mask_params.seg_height, m_mask_params.seg_width, mask_device, 32, mask_out_width, mask_out_height, m_stream);
				cudaMalloc(&mask_resized_device, sizeof(uint8_t) * (width*height));
				resize_cuda(mask_device, mask_resized_device, cv::Size(mask_out_width, mask_out_height), cv::Size(width, height), m_stream);
				cudaMallocHost(&mask_host, sizeof(uint8_t) * width * height);
				cudaMemcpyAsync(mask_host, mask_resized_device, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost, m_stream);
				cudaStreamSynchronize(m_stream);
				cv::Mat mask(height, width, CV_8UC1, mask_host);
				cv::Rect temp_rect = cv::Rect(left, top, width, height);
				mask = mask(temp_rect - cv::Point(left, top)) > m_mask_params.mask_threshold * 255;
				masks.push_back(mask.clone());
				cudaFree(mask_resized_device);
				cudaFree(mask_device);
				cudaFreeHost(mask_host);
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
		else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
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
	}

	scale_boxes(boxes, m_image.size());

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
	m_mask_params.params = YOLO_TensorRT_Detect::m_params;
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
