/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2025-12-21 23:26:32
 * @Description: openvino classify source file for YOLO algorithm
 */

#include "yolo_openvino.h"

void YOLO_OpenVINO_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenVINO::init(algo_type, device_type, model_type, model_path);

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		m_input_size.width = 224;
		m_input_size.height = 224;
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
	}
}

void YOLO_OpenVINO_Classify::pre_process()
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

	cv::dnn::blobFromImage(crop_image, m_input, 1, cv::Size(crop_image.cols, crop_image.rows), cv::Scalar(), true, false);
}

void YOLO_OpenVINO_Classify::process()
{
	ov::Tensor input_tensor(m_input_port.get_element_type(), m_input_port.get_shape(), m_input.ptr(0)); 
	m_infer_request.set_input_tensor(input_tensor); 
	m_infer_request.infer(); 
	m_output_host = (float*)m_infer_request.get_output_tensor(0).data();
}

void YOLO_OpenVINO_Classify::post_process()
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
