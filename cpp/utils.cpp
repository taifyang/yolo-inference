#include "utils.h"


void LetterBox(cv::Mat & input_image, cv::Mat & output_image, cv::Size & shape, cv::Scalar & color)
{
	cv::Size shape = input_image.size();
	float r = std::min((float)shape.height / (float)shape.height, (float)shape.width / (float)shape.width);
	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(shape.width - new_un_pad[0]) / 2;
	auto dh = (float)(shape.height - new_un_pad[1]) / 2;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		cv::resize(input_image, output_image, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		output_image = input_image.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

	cv::Vec4d params;
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


void scale_box(cv::Rect & box, cv::Size size)
{
	float gain = std::min(input_width * 1.0 / size.width, input_height * 1.0 / size.height);
	int pad_w = (input_width - size.width * gain) / 2;
	int pad_h = (input_height - size.height * gain) / 2;
	box.x -= pad_w;
	box.y -= pad_h;
	box.x /= gain;
	box.y /= gain;
	box.width /= gain;
	box.height /= gain;
}


void draw_result(cv::Mat & image, std::string label, cv::Rect box)
{
	cv::rectangle(image, box, cv::Scalar(255, 0, 0), 1);
	int baseLine;
	cv::Size label_size = cv::getTextSize(label, 1, 1, 1, &baseLine);
	cv::Point tlc = cv::Point(box.x, box.y);
	cv::Point brc = cv::Point(box.x, box.y + label_size.height + baseLine);
	cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
}
