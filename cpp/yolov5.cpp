#include "yolov5.h"
#include "utils.h"


void YOLOv5::infer(const std::string file_path)
{
	std::string suffix = file_path.substr(file_path.size() - 4);
	if (suffix == ".bmp" || suffix == ".jpg" || suffix == ".png")
	{
		m_image = cv::imread(file_path);
		m_result = m_image.clone();
		pre_process();
		process();
		post_process();
		cv::imwrite("result.jpg", m_result);
		cv::imshow("result", m_result);
		cv::waitKey(0);
	}
	else if (suffix == ".mp4")
	{
		cv::VideoCapture cap(file_path);
		while (cv::waitKey(1) < 0)
		{
			clock_t start = clock();

			cap.read(m_image);
			if (m_image.empty())
				break;

			m_result = m_image.clone();
			pre_process();
			process();
			post_process();
			cv::imshow("result", m_result);

			clock_t end = clock();
			std::cout << end - start << "ms" << std::endl;
		}
	}
}

	
void YOLOv5::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = m_outputs_host + i * output_numprob;
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
		scale_box(box, m_image.size());
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
		std::string label = class_names[class_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);
		draw_result(m_result, label, box);
	}
}

