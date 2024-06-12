#pragma once

#include "yolo.h"
#include "utils.h"

class YOLO_Classify : virtual public YOLO
{
protected:
	void draw_result(std::string label)
	{
		int baseLine;
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
		cv::putText(m_result, label, cv::Point(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
	}

	int class_num = 1000;
};
