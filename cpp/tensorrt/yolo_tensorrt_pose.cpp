/* 
 * @Author: taifyang
 * @Date: 2026-01-03 21:57:36
 * @LastEditTime: 2026-01-17 21:40:00
 * @Description: source file for YOLO tensorrt pose
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

void YOLO_TensorRT_Pose::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{	
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_Pose::init(algo_type, device_type, model_type, model_path);

	m_task_type = Pose;

	cudaMalloc(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output0_host, sizeof(float) * m_output_numdet);

	cudaMalloc(&m_input_device, sizeof(float) * m_input_numel);
	cudaMalloc(&m_output0_device, sizeof(float) * m_output_numdet);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output0_device;

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

void YOLO_TensorRT_Pose::pre_process()
{	
	YOLO_TensorRT_Detect::pre_process();
}

void YOLO_TensorRT_Pose::process()
{
	m_execution_context->executeV2((void**)m_bindings);

#ifndef _CUDA_POSTPROCESS
	cudaMemcpy(m_output0_host, m_output0_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost);
#endif // !_CUDA_POSTPROCESS
}

void YOLO_TensorRT_Pose::post_process()
{	
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> keypoints;

#ifdef _CUDA_POSTPROCESS
	cudaMemset(m_output_box_device, 0, sizeof(float) * (m_num_box_element * m_max_box + 1));	
	cuda_decode(m_output0_device, m_output_numbox, m_class_num, m_confidence_threshold, m_score_threshold,
		m_d2s_device, m_output_box_device, m_max_box, m_num_box_element, m_input_size, m_algo_type, m_task_type);
	cuda_nms(m_output_box_device, m_nms_threshold, m_max_box, m_num_box_element);
	cudaMemcpy(m_output_box_host, m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < m_max_box; i++)
	{
		if (m_output_box_host[58 * i + 7])
		{
			float x1 = m_output_box_host[58 * i + 1];
			float y1 = m_output_box_host[58 * i + 2];
			float x2 = m_output_box_host[58 * i + 3];
			float y2 = m_output_box_host[58 * i + 4];
			int left = int(x1) > 0 ? int(x1) : 0;
			int top = int(y1) > 0 ? int(y1) : 0;
			int width = int(x2 - x1) > 0 ? int(x2 - x1) : 0;
			int height = int(y2 - y1)> 0 ? int(y2 - y1) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);

			std::vector<float> keypoint(51);
			for (int j = 0; j < keypoint.size(); j++)
			{
				keypoint[j] = m_output_box_host[58 * i + 8 + j];
			}

			boxes.push_back(cv::Rect(left, top, width, height));
			scores.push_back(m_output_box_host[58 * i + 5]);
			keypoints.push_back(keypoint);
		}
	}

	m_output_pose.clear();
	m_output_pose.resize(boxes.size());
	for (int i = 0; i < boxes.size(); i++)
	{
		OutputPose output;
		output.id = 0;
		output.score = scores[i];
		output.box = boxes[i];
		output.keypoint = keypoints[i];
		m_output_pose[i] = output;
	}

#else
	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;

		float score = ptr[4];
		if (score < m_score_threshold)
			continue;
		
		std::vector<float> keypoint(51);
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
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
			for (int j = 0; j < keypoint.size(); j++)
			{
				keypoint[j] = ptr[5 + j];
			}
		}
		else if(m_algo_type == YOLO26)
		{
			int left = int(ptr[0]) > 0 ? int(ptr[0]) : 0;
			int top = int(ptr[1]) > 0 ? int(ptr[1]) : 0;
			int width = int(ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			int height = int(ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
			for (int j = 0; j < keypoint.size(); j++)
			{
				keypoint[j] = ptr[6 + j];
			}
		}

		boxes.push_back(box);
		scores.push_back(score);
		keypoints.push_back(keypoint);
	}

	scale_boxes(boxes, keypoints, m_image.size());

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		std::vector<int> indices;
		nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);
		m_output_pose.clear();
		m_output_pose.resize(indices.size());
		for (int i = 0; i < indices.size(); i++)
		{
			int idx = indices[i];
			OutputPose output;
			output.id = 0;
			output.score = scores[idx];
			output.box = boxes[idx];
			output.keypoint = keypoints[idx];
			m_output_pose[i] = output;
		}
	}
	else if(m_algo_type == YOLO26)
	{
		m_output_pose.clear();
		m_output_pose.resize(boxes.size());
		for (int i = 0; i < boxes.size(); i++)
		{
			OutputPose output;
			output.id = 0;
			output.score = scores[i];
			output.box = boxes[i];
			output.keypoint = keypoints[i];
			m_output_pose[i] = output;
		}
	}   
#endif // _CUDA_POSTPROCESS

	if(m_draw_result)
		draw_result(m_output_pose);
}

void YOLO_TensorRT_Pose::release()
{
	YOLO_TensorRT::release();

	cudaFree(m_output0_device);

#ifdef _CUDA_PREPROCESS
	cudaFree(m_d2s_device);
	cudaFreeHost(m_d2s_host);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaFree(m_output_box_device);
	cudaFreeHost(m_output_box_host);
#endif // _CUDA_POSTPROCESS
}