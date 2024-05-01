#include "yolov5_openvino.h"
#include "utils.h"


void YOLOv5_OpenVINO::init(const std::string model_path, const Device_Type device_type, const Model_Type model_type)
{
	ov::Core core; //Initialize OpenVINO Runtime Core 
	auto compiled_model = core.compile_model(model_path, device_type == GPU ? "GPU" : "CPU"); //Compile the Model 
	m_infer_request = compiled_model.create_infer_request(); //Create an Inference Request 
	m_input_port = compiled_model.input(); //Get input port for model with one input
}


void YOLOv5_OpenVINO::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	cv::dnn::blobFromImage(letterbox, m_inputs, 1. / 255., cv::Size(input_width, input_height), cv::Scalar(), true, false);
}


void YOLOv5_OpenVINO::process()
{
	ov::Tensor input_tensor(m_input_port.get_element_type(), m_input_port.get_shape(), m_inputs.ptr(0)); //Create tensor from external memory
	m_infer_request.set_input_tensor(input_tensor); // Set input tensor for model with one input
	m_infer_request.infer(); //Start inference
	m_outputs = m_infer_request.get_output_tensor(0); //Get the inference result 
	m_outputs_host = (float*)m_outputs.data();
}

