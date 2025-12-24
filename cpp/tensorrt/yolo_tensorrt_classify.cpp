/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2025-12-21 23:24:57
 * @Description: tensorrt _classify source file for YOLO algorithm
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

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
		m_input_size.width = 224;
		m_input_size.height = 224;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}

	m_task_type = Classify;

	cudaMallocHost(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output_host, sizeof(float) * m_class_num);

	cudaMalloc(&m_input_device, m_max_input_size);
	cudaMalloc(&m_output_device, sizeof(float) * m_class_num);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output_device;
}

void YOLO_TensorRT_Classify::pre_process()
{
	cv::Mat crop_image;
	if (m_algo_type == YOLOv5)
	{
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));

		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
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

void YOLO_TensorRT_Classify::process()
{
	m_execution_context->executeV2((void**)m_bindings);
	cudaMemcpyAsync(m_output_host, m_output_device, sizeof(float) * m_class_num, cudaMemcpyDeviceToHost, m_stream);
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
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
}

void YOLO_TensorRT_Classify::release()
{
	YOLO_TensorRT::release();
	
	cudaFree(m_output_device);
}
