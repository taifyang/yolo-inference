/* 
 * @Author: taifyang
 * @Date: 2026-01-03 21:31:46
 * @LastEditTime: 2026-08-29 16:42:17
* @Description: source file for YOLO openvino semantic segmentation 
 */

 #include "yolo_openvino.h"

 void YOLO_OpenVINO_Semantic::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenVINO::init(algo_type, device_type, model_type, model_path);
	YOLO_Semantic::init(algo_type, device_type, model_type, model_path);
}

void YOLO_OpenVINO_Semantic::pre_process()
{
	YOLO_OpenVINO_Detect::pre_process();
}

void YOLO_OpenVINO_Semantic::process()
{
	YOLO_OpenVINO_Detect::process();
}

void YOLO_OpenVINO_Semantic::post_process()
{
	m_output0_host = (uint8_t*)m_infer_request.get_output_tensor(0).data();

	m_mask = cv::Mat::zeros(m_image.size(), CV_8UC1);
	cv::Mat output = cv::Mat::zeros(m_input_size, CV_8UC1);	
	std::copy(m_output0_host, m_output0_host + m_output_numdet, output.data);
	scale_mask(output, m_mask, m_input_size, m_image.size());

	if (m_draw_result)
	{
		draw_result();
	}
}