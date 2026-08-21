/* 
 * @Author: taifyang
 * @Date: 2026-01-03 20:51:36
 * @LastEditTime: 2026-08-09 23:04:31
 * @Description: source file for YOLO opencv depth estimation
 */

 #include "yolo_opencv.h"

 void YOLO_OpenCV_Depth::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);
	YOLO_Depth::init(algo_type, device_type, model_type, model_path);
}

void YOLO_OpenCV_Depth::pre_process()
{
	YOLO_OpenCV_Detect::pre_process();
}

void YOLO_OpenCV_Depth::post_process()
{
	m_output0_host = (float*)m_output[0].data;

	m_depth = cv::Mat::zeros(m_input_size.width, m_input_size.height, CV_32FC1);	
	m_result = cv::Mat::zeros(m_image.size(), CV_16UC1);
	std::copy(m_output0_host, m_output0_host + m_output_numdet, (float*)m_depth.data);
	scale_mask(m_depth, m_input_size, m_image.size());
	
	if(m_draw_result)
		draw_result();
}