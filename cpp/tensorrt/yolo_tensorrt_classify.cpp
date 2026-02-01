/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-31 08:39:23
 * @Description: source file for YOLO tensorrt classification
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

void YOLO_TensorRT_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_Classify::init(algo_type, device_type, model_type, model_path);

	m_task_type = Classify;

#ifdef _CUDA_PREPROCESS
	cudaMalloc(&m_input, m_max_input_size);
	cudaMalloc(&m_image_crop, m_max_input_size);
	cudaMalloc(&m_image_centercrop_device, m_max_input_size);
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
		cudaMalloc(&m_image_resize_device, m_max_input_size);
#else
	cudaMallocHost(&m_input, m_max_input_size);
#endif // _CUDA_PREPROCESS
	cudaMallocHost(&m_output0_host, sizeof(float) * m_class_num);

	cudaMalloc(&m_input_device, sizeof(float) * m_input_numel);
	cudaMalloc(&m_output0_device, sizeof(float) * m_class_num);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output0_device;
}

void YOLO_TensorRT_Classify::pre_process()
{
#ifdef _CUDA_PREPROCESS
	cudaMemcpy(m_input, m_image.data, sizeof(uint8_t)*3*m_image.cols*m_image.rows, cudaMemcpyHostToDevice);

	if (m_algo_type == YOLOv5)
	{
		cuda_centercrop(m_input, m_image_crop, m_image_centercrop_device, m_image.size(), m_input_size, m_image.channels());
		cuda_normalize(m_image_centercrop_device, m_input_device, m_input_size, m_algo_type);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
	{
		cv::Size image_resize;
		if (m_image.cols > m_image.rows)
		{
			image_resize = cv::Size (m_input_size.height * m_image.cols / m_image.rows, m_input_size.height);		
			cuda_resize_linear(m_input, m_image_resize_device, m_image.size(), image_resize, m_image.channels());
		}
		else
		{
			image_resize = cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols);
			cuda_resize_linear(m_input, m_image_resize_device, m_image.size(), image_resize, m_image.channels());
		}		

		cuda_centercrop(m_image_resize_device, m_image_crop, m_image_centercrop_device, image_resize, m_input_size, m_image.channels());
		cuda_normalize(m_image_centercrop_device, m_input_device, m_input_size, m_algo_type);
	}
#else
	cv::Mat image_resize;
	if (m_algo_type == YOLOv5)
	{
		CenterCrop(m_image, image_resize);
		Normalize(image_resize, image_resize, m_algo_type);
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
	{
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, image_resize, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, image_resize, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));
		CenterCrop(image_resize, image_resize);
		Normalize(image_resize, image_resize, m_algo_type);
	}

	int image_area = image_resize.cols * image_resize.rows;
	float* pimage = (float*)image_resize.data;
	float* phost_r = m_input_host + image_area * 0;
	float* phost_g = m_input_host + image_area * 1;
	float* phost_b = m_input_host + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3)
	{
		*phost_r++ = pimage[2];
		*phost_g++ = pimage[1];
		*phost_b++ = pimage[0];
	}

	cudaMemcpy(m_input_device, m_input_host, sizeof(float) * m_input_numel, cudaMemcpyHostToDevice);
#endif // _CUDA_PREPROCESS
}

void YOLO_TensorRT_Classify::process()
{
	m_execution_context->executeV2((void**)m_bindings);
	cudaMemcpy(m_output0_host, m_output0_device, sizeof(float) * m_class_num, cudaMemcpyDeviceToHost);
}

void YOLO_TensorRT_Classify::post_process()
{
	std::vector<float> scores(m_class_num); 
	float sum = 0.0f;
	for (size_t i = 0; i < m_class_num; i++)
	{
		scores[i] = m_output0_host[i];
		sum += exp(m_output0_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo_type == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
}

void YOLO_TensorRT_Classify::release()
{
	YOLO_TensorRT::release();
	
	cudaFree(m_output0_device);
	cudaFree(m_input);
#ifdef _CUDA_PREPROCESS
	cudaFree(m_image_crop);
	cudaFree(m_image_centercrop_device);
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12 || m_algo_type == YOLO26)
		cudaFree(m_image_resize_device);
#endif // _CUDA_PREPROCESS
}
