/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2025-12-21 23:27:01
 * @Description: opencv classify source file for YOLO algorithm
 */

#include "yolo_opencv.h"

void YOLO_OpenCV_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		m_input_size.width = 224;
		m_input_size.height = 224;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}
}

void YOLO_OpenCV_Classify::pre_process()
{
	cv::Mat crop_image;

	if (m_algo_type == YOLOv5)
	{
#ifndef OPENCV_WITH_CUDA
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
#else
		cv::cuda::GpuMat gpu_image, gpu_crop_image, gpu_cvt_image;
    	gpu_image.upload(m_image);
		int crop_size = std::min(m_image.cols, m_image.rows);
		int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
		gpu_crop_image = gpu_image(cv::Rect(left, top, crop_size, crop_size));
		cv::cuda::resize(gpu_crop_image, gpu_crop_image, cv::Size(m_input_size.width, m_input_size.height));
		gpu_crop_image.convertTo(gpu_cvt_image, CV_32FC3, 1. / 255.);
		cv::cuda::subtract(gpu_cvt_image, cv::Scalar(0.406, 0.456, 0.485), gpu_cvt_image);
		cv::cuda::divide(gpu_cvt_image, cv::Scalar(0.225, 0.224, 0.229), gpu_cvt_image);
		gpu_cvt_image.download(crop_image);
#endif
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{	
#ifndef OPENCV_WITH_CUDA	
		if (m_image.cols > m_image.rows)
			cv::resize(m_image, crop_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::resize(m_image, crop_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));
		CenterCrop(m_image, crop_image);
		Normalize(crop_image, crop_image, m_algo_type);
#else
		cv::cuda::GpuMat gpu_image, gpu_crop_image, gpu_cvt_image;
		gpu_image.upload(m_image);
		if (m_image.cols > m_image.rows)
			cv::cuda::resize(gpu_image, gpu_image, cv::Size(m_input_size.height * m_image.cols / m_image.rows, m_input_size.height));
		else
			cv::cuda::resize(gpu_image, gpu_image, cv::Size(m_input_size.width, m_input_size.width * m_image.rows / m_image.cols));
		int crop_size = std::min(crop_image.cols, crop_image.rows);
		int left = (crop_image.cols - crop_size) / 2, top = (crop_image.rows - crop_size) / 2;
		gpu_crop_image = gpu_image(cv::Rect(left, top, crop_size, crop_size));
		cv::cuda::resize(gpu_crop_image, gpu_crop_image, cv::Size(m_input_size.width, m_input_size.height));
		gpu_crop_image.convertTo(gpu_cvt_image, CV_32FC3, 1. / 255.);
		gpu_cvt_image.download(crop_image);
#endif
	}

	cv::dnn::blobFromImage(crop_image, m_input, 1, cv::Size(crop_image.cols, crop_image.rows), cv::Scalar(), true, false);
}

void YOLO_OpenCV_Classify::post_process()
{
	m_output_host = (float*)m_output[0].data;

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
