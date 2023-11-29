#include "yolov5_openvino.h"
#include "utils.h"


YOLOv5_OpenVINO::YOLOv5_OpenVINO(std::string model_path, Device_Type device_type)
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
}


void YOLOv5_OpenVINO::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = (float*)m_outputs.data() + i * output_numprob;
		float objness = ptr[4];
		if (objness < confidence_threshold)
			continue;

		float* classes_scores = ptr + 5;
		int class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
		float max_class_score = classes_scores[class_id];
		float score = max_class_score * objness;
		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_boxes(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		std::string label = class_names[class_ids[idx]] + ":" + cv::format("%.2f", scores[idx]); //class_ids[idx]ÊÇclass_id
		draw_result(m_result, label, box);
	}
}

