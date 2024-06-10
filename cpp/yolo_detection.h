#pragma once

#include "yolo.h"
#include "utils.h"

class YOLO_Detection : virtual public YOLO
{
protected:
	void LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Vec4d& params, cv::Size shape = cv::Size(640, 640), cv::Scalar color = cv::Scalar(114, 114, 114))
	{
		float r = std::min((float)shape.height / (float)input_image.rows, (float)shape.width / (float)input_image.cols);
		float ratio[2]{ r, r };
		int new_un_pad[2] = { (int)std::round((float)input_image.cols * r),(int)std::round((float)input_image.rows * r) };

		auto dw = (float)(shape.width - new_un_pad[0]) / 2;
		auto dh = (float)(shape.height - new_un_pad[1]) / 2;

		if (input_image.cols != new_un_pad[0] && input_image.rows != new_un_pad[1])
			cv::resize(input_image, output_image, cv::Size(new_un_pad[0], new_un_pad[1]));
		else
			output_image = input_image.clone();

		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));

		params[0] = ratio[0];
		params[1] = ratio[1];
		params[2] = left;
		params[3] = top;

		cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
	}

	void nms(std::vector<cv::Rect> & boxes, std::vector<float> & scores, float score_threshold, float nms_threshold, std::vector<int> & indices)
	{
		assert(boxes.size() == scores.size());

		struct BoxScore
		{
			cv::Rect box;
			float score;
			int id;
		};
		std::vector<BoxScore> boxes_scores;
		for (size_t i = 0; i < boxes.size(); i++)
		{
			BoxScore box_conf;
			box_conf.box = boxes[i];
			box_conf.score = scores[i];
			box_conf.id = i;
			if (scores[i] > score_threshold)	boxes_scores.push_back(box_conf);
		}

		std::sort(boxes_scores.begin(), boxes_scores.end(), [](BoxScore a, BoxScore b) { return a.score > b.score; });

		std::vector<float> area(boxes_scores.size());
		for (size_t i = 0; i < boxes_scores.size(); ++i)
		{
			area[i] = boxes_scores[i].box.width * boxes_scores[i].box.height;
		}

		std::vector<bool> isSuppressed(boxes_scores.size(), false);
		for (size_t i = 0; i < boxes_scores.size(); ++i)
		{
			if (isSuppressed[i])  continue;
			for (size_t j = i + 1; j < boxes_scores.size(); ++j)
			{
				if (isSuppressed[j])  continue;

				float x1 = (std::max)(boxes_scores[i].box.x, boxes_scores[j].box.x);
				float y1 = (std::max)(boxes_scores[i].box.y, boxes_scores[j].box.y);
				float x2 = (std::min)(boxes_scores[i].box.x + boxes_scores[i].box.width, boxes_scores[j].box.x + boxes_scores[j].box.width);
				float y2 = (std::min)(boxes_scores[i].box.y + boxes_scores[i].box.height, boxes_scores[j].box.y + boxes_scores[j].box.height);
				float w = (std::max)(0.0f, x2 - x1);
				float h = (std::max)(0.0f, y2 - y1);
				float inter = w * h;
				float ovr = inter / (area[i] + area[j] - inter);

				if (ovr >= nms_threshold)  isSuppressed[j] = true;
			}
		}

		for (int i = 0; i < boxes_scores.size(); ++i)
		{
			if (!isSuppressed[i])	indices.push_back(boxes_scores[i].id);
		}
	}

	void scale_box(cv::Rect& box, cv::Size size)
	{
		float gain = std::min(m_input_width * 1.0 / size.width, m_input_height * 1.0 / size.height);
		int pad_w = (m_input_width - size.width * gain) / 2;
		int pad_h = (m_input_height - size.height * gain) / 2;
		box.x -= pad_w;
		box.y -= pad_h;
		box.x /= gain;
		box.y /= gain;
		box.width /= gain;
		box.height /= gain;
	}

	void draw_result(std::string label, cv::Rect box)
	{
		cv::rectangle(m_result, box, cv::Scalar(255, 0, 0), 1);
		cv::putText(m_result, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
	}

	int class_num = 80;

	float score_threshold = 0.2;

	float nms_threshold = 0.5;

	float confidence_threshold = 0.2;

	cv::Vec4d m_params;

	int m_output_numprob;

	int m_output_numbox;

	int m_output_numdet;
};
