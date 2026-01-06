/* 
 * @Author: taifyang
 * @Date: 2026-01-03 20:22:46
 * @LastEditTime: 2026-01-06 11:07:16
 * @Description: source file for YOLO libtorch pose
 */

 #include "yolo_libtorch.h"

 void YOLO_Libtorch_Pose::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
	YOLO_Pose::init(algo_type, device_type, model_type, model_path);
}

void YOLO_Libtorch_Pose::pre_process()
{
	YOLO_Libtorch_Detect::pre_process();
}

void YOLO_Libtorch_Pose::process()
{
	m_output = m_module.forward(m_input);
	torch::Tensor pred;
	pred = m_output.toTensor().to(at::kCPU);
	m_output0.assign(pred.data_ptr<float>(), pred.data_ptr<float>() + m_output_numdet);
}

void YOLO_Libtorch_Pose::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> keypoints;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0.data() + i * m_output_numprob;

		float score = ptr[4];
		if (score < m_score_threshold)
			continue;
		
		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w) > 0 ? int(x - 0.5 * w) : 0;
		int top = int(y - 0.5 * h) > 0 ? int(y - 0.5 * h) : 0;
		int width = int(w) > 0 ? int(w) : 0;
		int height = int(h)> 0 ? int(h) : 0;
		width = (left + width) < m_image.cols ? width : (m_image.cols - left);
		height = (top + height) < m_image.rows ? height : (m_image.rows - top);
		cv::Rect box = cv::Rect(left, top, width, height);

		std::vector<float> keypoint(51);
		for (int j = 0; j < keypoint.size(); j++)
		{
			keypoint[j] = ptr[5 + j];
		}

		boxes.push_back(box);
		scores.push_back(score);
		keypoints.push_back(keypoint);
	}

	scale_boxes(boxes, keypoints, m_image.size());

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	m_output_pose.clear();
	m_output_pose.resize(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		OutputPose output;
		output.id = 0;
		output.score = scores[idx];
		output.box = boxes[idx];
		output.keypoint = keypoints[idx];
		m_output_pose[i] = output;
	}

	if(m_draw_result)
		draw_result(m_output_pose);
}