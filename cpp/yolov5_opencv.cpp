#include "yolov5_opencv.h"
#include "utils.h"


YOLOv5_OpenCV::YOLOv5_OpenCV(std::string model_path, Device_Type device_type, Model_Type model_type)
{
	assert(("unsupported model type!", model_type == FP32));
	m_net = cv::dnn::readNet(model_path);
	if (device_type == GPU)
	{
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);//cv::dnn::DNN_BACKEND_OPENCV;
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);  //cv::dnn::DNN_TARGET_CPU;
	}
}


void YOLOv5_OpenCV::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	cv::dnn::blobFromImage(letterbox, m_inputs, 1. / 255., cv::Size(input_width, input_height), cv::Scalar(), true, false);
}


void YOLOv5_OpenCV::process()
{
	m_net.setInput(m_inputs);
	m_net.forward(m_outputs, m_net.getUnconnectedOutLayersNames());
	m_outputs_host = (float*)m_outputs[0].data;
}

