/* 
 * @Author: taifyang
 * @Date: 2026-01-12 10:30:09
 * @LastEditTime: 2026-01-12 10:45:18
* @Description: source file for YOLO openvino obb
 */

 #include "yolo_openvino.h"

 void YOLO_OpenVINO_OBB::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenVINO::init(algo_type, device_type, model_type, model_path);
	YOLO_OBB::init(algo_type, device_type, model_type, model_path);
}

void YOLO_OpenVINO_OBB::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);

	std::vector<cv::Mat> split_images;
	cv::split(letterbox, split_images);
	for(auto& split_image : split_images)
		split_image = split_image.reshape(1, 1);
	cv::hconcat(split_images, m_input);
}

void YOLO_OpenVINO_OBB::process()
{
	ov::Tensor input_tensor(m_input_port.get_element_type(), m_input_port.get_shape(), m_input.ptr(0)); 
	m_infer_request.set_input_tensor(input_tensor); 
	m_infer_request.infer(); 	
}

void YOLO_OpenVINO_OBB::post_process()
{
	m_output0_host = (float*)m_infer_request.get_output_tensor(0).data(); 

	std::vector<std::vector<float>> boxes_rotated;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host+ i * m_output_numprob;

		float* classes_scores = ptr + 4;
		int class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
		float score = classes_scores[class_id];
		if (score < m_score_threshold)
			continue;

		float angle = ptr[m_class_num + 4];
		boxes_rotated.push_back({ptr[0], ptr[1], ptr[2], ptr[3], score, class_id, angle});
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms_rotated(boxes_rotated, scores, m_score_threshold, m_nms_threshold, indices);

	std::vector<std::vector<float>> boxes_nms(indices.size());
	for(int i=0; i<indices.size(); i++)	
		boxes_nms[i] = boxes_rotated[indices[i]];

	regularize_rboxes(boxes_nms);
	
	scale_rboxes(boxes_nms, m_image.size());

	m_output_obb.clear();
	m_output_obb.resize(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		OutputOBB output;
		output.score = boxes_nms[i][4];
		output.id = boxes_nms[i][5];
		float angle = boxes_nms[i][6] * 180 / CV_PI;
		output.box_rotate = cv::RotatedRect(cv::Point2f(boxes_nms[i][0], boxes_nms[i][1]), cv::Size2f(boxes_nms[i][2], boxes_nms[i][3]), angle);
		m_output_obb[i] = output;
	}

	if(m_draw_result)
		draw_result(m_output_obb);
}