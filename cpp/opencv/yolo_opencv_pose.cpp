/* 
 * @Author: taifyang
 * @Date: 2026-01-03 20:51:36
 * @LastEditTime: 2026-01-19 23:56:42
 * @Description: source file for YOLO opencv pose
 */

 #include "yolo_opencv.h"

 void YOLO_OpenCV_Pose::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12 && algo_type != YOLO26)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_OpenCV::init(algo_type, device_type, model_type, model_path);
	YOLO_Pose::init(algo_type, device_type, model_type, model_path);
}

void YOLO_OpenCV_Pose::pre_process()
{
	YOLO_OpenCV_Detect::pre_process();
}

void YOLO_OpenCV_Pose::post_process()
{
	m_output0_host = (float*)m_output[0].data;

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> keypoints;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;

		float score = ptr[4];
		if (score < m_score_threshold)
			continue;
		
		cv::Rect box;
		std::vector<float> keypoint(51);
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
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
			box = cv::Rect(left, top, width, height);
			for (int j = 0; j < keypoint.size(); j++)
			{
				keypoint[j] = ptr[5 + j];
			}
		}
		else if(m_algo_type == YOLO26)
		{
			int left = int(ptr[0]) > 0 ? int(ptr[0]) : 0;
			int top = int(ptr[1]) > 0 ? int(ptr[1]) : 0;
			int width = int(ptr[2] - ptr[0]) > 0 ? int(ptr[2] - ptr[0]) : 0;
			int height = int(ptr[3] - ptr[1])> 0 ? int(ptr[3] - ptr[1]) : 0;
			width = (left + width) < m_image.cols ? width : (m_image.cols - left);
			height = (top + height) < m_image.rows ? height : (m_image.rows - top);
			box = cv::Rect(left, top, width, height);
			for (int j = 0; j < keypoint.size(); j++)
			{
				keypoint[j] = ptr[6 + j];
			}
		}

		boxes.push_back(box);
		scores.push_back(score);
		keypoints.push_back(keypoint);
	}

	scale_boxes(boxes, keypoints, m_image.size());

	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
	{
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, scores, m_score_threshold, m_nms_threshold, indices);
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
	}
	else if(m_algo_type == YOLO26)
	{
		m_output_pose.clear();
		m_output_pose.resize(boxes.size());
		for (int i = 0; i < boxes.size(); i++)
		{
			OutputPose output;
			output.id = 0;
			output.score = scores[i];
			output.box = boxes[i];
			output.keypoint = keypoints[i];
			m_output_pose[i] = output;
		}
	}

	if(m_draw_result)
		draw_result(m_output_pose);
}