/* 
 * @Author: taifyang
 * @Date: 2026-01-03 21:57:36
 * @LastEditTime: 2026-08-25 22:25:30
 * @Description: source file for YOLO tensorrt depth estimation
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/postprocess.cuh"

void YOLO_TensorRT_Depth::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{	
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_Depth::init(algo_type, device_type, model_type, model_path);

	m_task_type = Depth;

#ifdef _CUDA_PREPROCESS
	cudaMalloc(&m_input, m_max_input_size);
#else
	cudaMallocHost(&m_input, m_max_input_size);
#endif // _CUDA_PREPROCESS
	cudaMallocHost(&m_output0_host, sizeof(float) * m_output_numdet);

	cudaMalloc(&m_input_device, sizeof(float) * m_input_numel);
	cudaMalloc(&m_output0_device, sizeof(float) * m_output_numdet);

	m_bindings.push_back(m_input_device);
	m_bindings.push_back(m_output0_device);
}

void YOLO_TensorRT_Depth::pre_process()
{	
	YOLO_TensorRT_Detect::pre_process();
}

void YOLO_TensorRT_Depth::process()
{
	m_execution_context->executeV2(m_bindings.data());

//#ifndef _CUDA_POSTPROCESS
	cudaMemcpy(m_output0_host, m_output0_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost);
//#endif // !_CUDA_POSTPROCESS
}

void YOLO_TensorRT_Depth::post_process()
{		
	m_depth = cv::Mat::zeros(m_image.size(), CV_32FC1);

#ifdef _CUDA_POSTPROCESS
	cudaMalloc(&m_depth_device, sizeof(float) * m_image.cols * m_image.rows);
	cuda_scale_mask(m_output0_device, m_depth_device, m_input_size, m_image.size());
	cudaMemcpy(m_depth.data, m_depth_device, sizeof(float) * m_image.cols * m_image.rows, cudaMemcpyDeviceToHost);
	cudaFree(m_depth_device);
#else
	cv::Mat depth = cv::Mat::zeros(m_input_size, CV_32FC1);
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	std::copy(m_output0_host, m_output0_host + m_output_numdet, (float*)depth.data);
	scale_mask(depth, m_depth, m_input_size, m_image.size());
#endif // !_CUDA_POSTPROCESS

	if(m_draw_result)
		draw_result();
}

void YOLO_TensorRT_Depth::release()
{
	YOLO_TensorRT::release();

	cudaFree(m_output0_device);

#ifdef _CUDA_PREPROCESS
	cudaFree(m_input);
#else
	cudaFreeHost(m_input);
#endif // _CUDA_PREPROCESS
}