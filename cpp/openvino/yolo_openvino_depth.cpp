/* 
 * @Author: taifyang
 * @Date: 2026-01-03 21:31:46
 * @LastEditTime: 2026-08-26 00:01:55
* @Description: source file for YOLO openvino depth estimation
 */

 #include "yolo_openvino.h"

 void YOLO_OpenVINO_Depth::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenVINO::init(algo_type, device_type, model_type, model_path);
	YOLO_Depth::init(algo_type, device_type, model_type, model_path);
}

void YOLO_OpenVINO_Depth::pre_process()
{
	YOLO_OpenVINO_Detect::pre_process();
}

void YOLO_OpenVINO_Depth::process()
{
	YOLO_OpenVINO_Detect::process();
}

void YOLO_OpenVINO_Depth::post_process()
{
	m_output0_host = (float*)m_infer_request.get_output_tensor(0).data();

	cv::Mat depth = cv::Mat::zeros(m_input_size, CV_32FC1);
	m_depth = cv::Mat::zeros(m_image.size(), CV_32FC1);		
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	std::copy(m_output0_host, m_output0_host + m_output_numdet, (float*)depth.data);
	scale_mask(depth, m_depth, m_input_size, m_image.size());
	
	if(m_draw_result)
		draw_result();
}